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


static void bench_suitesparse_mattransmul_taco(benchmark::State& state, bool gen=true, int fill_value=0) {
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

 //   std::cout << ssTensor << std::endl;
 //   std::cout << otherVec << std::endl;

   // actual computation
   // int dim = state.range(0);
   std::cout << "Before get vec" << std::endl;
   Tensor<float> otherVecj = inputCacheFloat.otherVecFirstMode;

   Tensor<float> otherVecjDense("d", {DIM1}, Format{Dense});
   std::vector<int> coordsj(otherVecj.getOrder());
   for (auto &value: taco::iterate<float>(otherVecj)) {
   	for (int ii = 0; ii < otherVecj.getOrder(); ii++) {
		coordsj[ii] = value.first[ii];
        }
        otherVecjDense.insert(coordsj, (float)value.second);

   }

   Tensor<float> otherVeci = inputCacheFloat.otherVecLastMode;

   Tensor<float> otherVeciDense("b", {DIM0}, Format{Dense});
   std::vector<int> coordsi(otherVeci.getOrder());
   for (auto &value: taco::iterate<float>(otherVeci)) {
   	for (int ii = 0; ii < otherVeci.getOrder(); ii++) {
		coordsi[ii] = value.first[ii];
        }
        otherVeciDense.insert(coordsi, (float)value.second);

   }

   taco::Tensor<float> s1("s1"), s2("s2");
   s1.insert({}, float(2));
   s2.insert({}, float(2));

   IndexVar i("i");
   IndexVar j("j");
   IndexExpr accelerateExpr = s1() * ssTensor(j, i) * otherVecjDense(j) + s2() * otherVeciDense(i);
   
   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> res("res", {DIM1}, Format{Dense});
    res(i) = accelerateExpr;
   
    IndexStmt stmt = res.getAssignment().concretize();

    res.compile(stmt);
    state.ResumeTiming();
    res.assemble();
    auto func = res.compute_split();
    auto pair = res.returnFuncPackedRaw(func);
    pair.first(func.data());
  }

}

TACO_BENCH_ARGS(bench_suitesparse_mattransmul_taco, mat_mattransmul, false)->UseRealTime();
