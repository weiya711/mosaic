#ifndef TBLIS_INTERFACE_H
#define TBLIS_INTERFACE_H

using namespace taco;

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"

class TestInterface : public AbstractFunctionInterface{
    public: 
        TestInterface() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()) {};

        AcceleratorStmt getStmt() const override {return z(i, k) = x(i, j) * y(j, k);}
        std::vector<Argument> getArguments() const override {
                                                taco::TransferLoad load1("test1", "int");
                                                taco::TransferLoad load2("test2", "int");
                                                return 
                                                {
                                                    new DimArg(i),
                                                    load1(load2(Dim(i)), x, y)
                                                }; }

        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_init_tensor_d";}
        std::vector<Argument>  callBefore() const override {
                                taco::TransferLoad call("callBefore", "void");
                                return { call(x, y) };
                            }

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
};

class TestInterfaceIncorrect : public AbstractFunctionInterface{
    public: 
        TestInterfaceIncorrect() : 
                    x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    y(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    z(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}),  Format{Dense, Dense})),
                    i(IndexVar()),
                    j(IndexVar()),
                    k(IndexVar()) {};

        AcceleratorStmt getStmt() const override {return z(i, k) = x(i, j) * y(j, k);}
        std::vector<Argument> getArguments() const override {
                                                taco::TransferLoad load("test", "void");
                                                return {
                                                    new DimArg(i), 
                                                    load(Dim(i), x, y)};
                                            }
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "tblis_init_tensor_d";}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject z;
        IndexVar i;
        IndexVar j;
        IndexVar k;
};

#endif
