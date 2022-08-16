#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lowerer_impl_imperative.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/lower/lower.h"
#include "taco/ir_tags.h"
#include "taco/error/error_messages.h"

#include "taco/accelerator_interface/cblas_saxpy.h"
#include "taco/accelerator_interface/test_interface.h"


using namespace taco;


bool trivialkernelChecker(IndexStmt expr){
   return true;
}


TEST(interface, pluginInterface) {

   TensorVar a("a", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar b("b", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   // should basically call a C function 
   // that can be included in header
   TransferLoad load_test("load_test", "void");
   TransferStore store_test("store_test", "void");

   TransferType kernelTransfer("test", load_test, store_test);

   ForeignFunctionDescription kernel1("kernel1", "void", a(i),  a(i) + b(i), {}, trivialkernelChecker);
   ForeignFunctionDescription kernel2( "kernel2", "void", a(i), b(i), {}, trivialkernelChecker);

   AcceleratorDescription accelDesc(kernelTransfer, 
            {  kernel1(load_test(a)),
               kernel2(load_test(a, load_test(b)), load_test(b))
            });

   cout << load_test(a, load_test(a), load_test(a, load_test(a)), b, Dim(i)) << endl;

   Tensor<double> A("A", {16}, Format{Dense});

}


TEST(interface, concretepluginInterface) {

   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexVar i("i");
   IndexVar iw("iw");
   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();

   //due the way rewrite indexstmt works, need the same object
   ConcreteAccelerateCodeGenerator concrete_cblas_saxpy("cblas_saxpy", "void",  C(i), accelerateExpr, {});
   cout << concrete_cblas_saxpy(Dim(i), 1, A, 1, B, 1) << endl;

   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);

   stmt = stmt.accelerate(concrete_cblas_saxpy(Dim(i), 1, B, 1, C, 1), i, iw, accelWorkspace);
   
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);


}

TEST(interface, endToEndPlugin) {

   TensorVar x("x", Type(taco::Float32, {Dimension()}), taco::dense);
   TensorVar y("y", Type(taco::Float32, {Dimension()}), taco::dense);
   IndexVar i("i");

   ForeignFunctionDescription cblas_saxpy("cblas_saxpy", "void", x(i) <=  x(i) + y(i), {}, trivialkernelChecker);

   AcceleratorDescription accelDesc({cblas_saxpy(Dim(i), 1, y, 1, x, 1)});

   // actual computation
   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   A(i) = B(i) + C(i);

   // register the description
   // A.registerAccelerator(accelDesc);
   // enable targeting
   // A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   Tensor<float> expected("expected", {16}, Format{Dense});
   expected(i) = B(i) + C(i);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, interfaceClass1) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new Saxpy(), accelerateExpr);

    
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, interfaceClass2) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   stmt = stmt.accelerate(new DotProduct(), accelerateExpr);

    
   A.compile(stmt);
   A.assemble();
   A.compute();

   expected(i) = accelerateExpr;
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);
}

TEST(interface, endToEndPluginInterfaceClass) {

   // actual computation
   Tensor<float> A("A", {16}, Format{Dense});
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   IndexVar i("i");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   C.pack();
   B.pack();

   A(i) = B(i) + C(i) + B(i);

   // register the description
   A.registerAccelerator(new Saxpy());
   // enable targeting
   A.accelerateOn();
   
   A.compile();
   A.assemble();
   A.compute();

   Tensor<float> expected("expected", {16}, Format{Dense});
   expected(i) = B(i) + C(i) + B(i);
   expected.compile();
   expected.assemble();
   expected.compute();

   ASSERT_TENSOR_EQ(expected, A);

}

TEST(interface, mismatchInterfaceClass) {


   Tensor<float> A("A", {16}, Format{Dense}, 0);
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});
   Tensor<float> expected("expected", {16}, Format{Dense});
   TensorVar accelWorkspace("accelWorkspace", Type(taco::Float32, {16}), taco::dense);
   IndexVar i("i");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   // Test1 test;

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) + C(i);
   A(i) = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();

   // ASSERT_THROW(stmt.accelerate(new Test1(), accelerateExpr, i, iw, accelWorkspace), taco::TacoException);

}

TEST(interface, classInterfaceSdsdot) {


   Tensor<float> A("A");
   Tensor<float> B("B", {16}, Format{Dense});
   Tensor<float> C("C", {16}, Format{Dense});

   Tensor<float> expected("expected");
   TensorVar accelWorkspace((Type(taco::Float32)));

   IndexVar i("i");
   IndexVar iw("iw");

   for (int i = 0; i < 16; i++) {
      C.insert({i}, (float) i);
      B.insert({i}, (float) i);
   }

   // Test1 test;

   C.pack();
   B.pack();

   IndexExpr accelerateExpr = B(i) * C(i);
   A = accelerateExpr;

   IndexStmt stmt = A.getAssignment().concretize();
   // stmt = stmt.accelerate(new Sdsdot(), accelerateExpr, i, iw, accelWorkspace);
// 
   // IndexExpr accelerateExpr = B(i) + C(i);
   // A(i) = accelerateExpr;

   // IndexStmt stmt = A.getAssignment().concretize();

   // ASSERT_THROW(stmt.accelerate(new Test1(), accelerateExpr, i, iw, accelWorkspace), taco::TacoException);

}