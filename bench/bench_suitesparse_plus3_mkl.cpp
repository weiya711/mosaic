#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"

using namespace taco;


static void bench_suitesparse_plus3_mkl(benchmark::State& state, bool gen=true, int fill_value=0) {
  bool GEN_OTHER = (getEnvVar("GEN") == "ON" && gen);

  // Counters must be present in every run to get reported to the CSV.
  state.counters["dimx"] = 0;
  state.counters["dimy"] = 0;
  state.counters["nnz"] = 0;
  state.counters["other_sparsity1"] = 0;
  state.counters["other_sparsity1"] = 0;

  auto tensorPath = getEnvVar("SUITESPARSE_TENSOR_PATH");
  if (tensorPath == "") {
    std::cout << "BENCHMARK ERROR" << std::endl;
    state.error_occurred();
    return;
  }
  std::cout << tensorPath << std::endl;
  auto pathSplit = taco::util::split(tensorPath, "/");
  auto filename = pathSplit[pathSplit.size() - 1];
  auto tensorName = taco::util::split(filename, ".")[0];
  state.SetLabel(tensorName);

  taco::Tensor<float> ssTensor, otherShifted;
  try {
    taco::Format format = CSR;
    std::tie(ssTensor, otherShifted) = inputCacheFloat.getTensorInput(tensorPath, tensorName, format, true /* countNNZ */,
                                                                 true /* includeThird */, true, false, GEN_OTHER, true);
  } catch (TacoException &e) {
    // Counters don't show up in the generated CSV if we used SkipWithError, so
    // just add in the label that this run is skipped.
    std::cout << e.what() << std::endl;
    state.SetLabel(tensorName + "/SKIPPED-FAILED-READ");
    return;
  }

  int DIM0 = ssTensor.getDimension(0);
  int DIM1 = ssTensor.getDimension(1);

  state.counters["dimx"] = DIM0;
  state.counters["dimy"] = DIM1;
  state.counters["nnz"] = inputCacheFloat.nnz;

  Tensor<float> otherShifted2 = inputCacheFloat.thirdTensor;


   IndexVar i("i");
   IndexVar j("j");
   IndexExpr accelerateExpr = ssTensor(i, j) + otherShifted(i, j);
 	 
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> res1("res1", {DIM0, DIM1}, Format{Dense, Sparse});
    Tensor<float> res2("res2", {DIM0, DIM1}, Format{Dense, Sparse});
    res1(i, j) = accelerateExpr;
    IndexExpr accelerateExpr2 = res1(i, j) + otherShifted2(i, j);
    res2(i, j) = accelerateExpr2; 

    IndexStmt stmt1 = res1.getAssignment().concretize();
    stmt1 = stmt1.accelerate(new MklAdd(), accelerateExpr, true);
    IndexStmt stmt2 = res2.getAssignment().concretize();
    stmt2 = stmt2.accelerate(new MklAdd(), accelerateExpr2, true);

    res1.compile(stmt1);
    state.ResumeTiming();
    res1.assemble();
    auto func1 = res1.compute_split();
    auto pair1 = res1.returnFuncPackedRaw(func1);
    pair1.first(func1.data());
    state.PauseTiming();

    res2.compile(stmt2);
    state.ResumeTiming();
    res2.assemble();
    auto func2 = res2.compute_split();
    auto pair2 = res2.returnFuncPackedRaw(func2);
    pair2.first(func2.data());

  }
}

TACO_BENCH_ARGS(bench_suitesparse_plus3_mkl, mat_plus3, false)->UseRealTime();

