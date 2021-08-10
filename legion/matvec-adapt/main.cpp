#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionRows(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces);
LogicalPartition partitionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t pieces);
void computeLegionRows(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition aPart, LogicalPartition bPart);
void computeLegionCols(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition aPart, LogicalPartition bPart, LogicalPartition cPart);

// TODO (rohany): This code could be generated by TACO, but I haven't wired up the partitioning
//  logic to handle not in-order distributions yet. I'll hold off to try and do a full refactoring
//  of this code to have a "partition" pass, and a "compute" pass that operates on the partitions.
LogicalPartition partitionCols(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces) {
  long long A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  long long A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<2> AStart = Point<2>(0, (in * ((A2_dimension + (pieces - 1)) / pieces) + 0 / pieces));
    Point<2> AEnd = Point<2>(std::min(A1_dimension, ADomain.hi()[0]), std::min((in * ((A2_dimension + (pieces - 1)) / pieces) + ((A1_dimension + (pieces - 1)) / pieces - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int pieces = -1;
  bool match = false;
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-pieces") == 0) {
      pieces = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-match") == 0) {
      match = true;
      continue;
    }
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (pieces == -1) {
    std::cout << "Please provide a number of pieces with -pieces." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);

  auto matSpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto vecSpace = runtime->create_index_space(ctx, Rect<1>({0, n - 1}));
  auto A = runtime->create_logical_region(ctx, matSpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, matSpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, vecSpace, fspace); runtime->attach_name(C, "C");

  // Create the row-wise partition of the matrices and fill them.
  auto aPartRows = partitionRows(ctx, runtime, A, pieces);
  auto bPartRows = partitionRows(ctx, runtime, B, pieces);
  auto aPartCols = partitionCols(ctx, runtime, A, pieces);
  auto bPartCols = partitionCols(ctx, runtime, B, pieces);
  // Make a partition of C for the column-wise operation.
  auto cPartCols = partitionC(ctx, runtime, C, pieces);

  // Just fill C with the runtime.
  runtime->fill_field(ctx, C, C, FID_VAL, valType(1));

  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    // Fill the row wise partitions.
    tacoFill<valType>(ctx, runtime, A, aPartRows, 0);
    tacoFill<valType>(ctx, runtime, B, bPartRows, 1);

    // Wait for the fill to complete so that it isn't included as part of the execution time.
    runtime->issue_execution_fence(ctx).wait();

    // If we're supposed to match the partition, then use the row-wise compute. Otherwise
    // use the column wise compute.
    benchmark(ctx, runtime, times, [&]() {
      if (match) {
        computeLegionRows(ctx, runtime, A, B, C, aPartRows, bPartRows);
      } else {
        computeLegionCols(ctx, runtime, A, B, C, aPartCols, bPartCols, cPartCols);
      }
    });
  }

  // Calculate the total bandwidth.
  size_t elems = [](size_t n) { return 2 * n * n + n; }(n);
  size_t bytes = elems * sizeof(valType);
  double gbytes = double(bytes) / 1e9;
  auto avgTimeS = double(average(times)) / 1e3;
  double bw = gbytes / (avgTimeS);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GB/s BW per node: %lf.\n", nodes, bw / double(nodes));
}

TACO_MAIN(valType)