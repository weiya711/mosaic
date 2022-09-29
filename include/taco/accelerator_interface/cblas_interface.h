#ifndef CBLAS_INTERFACE_H
#define CBLAS_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/type.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accel_interface.h"

using namespace taco;

class Saxpy : public AbstractFunctionInterface{
    public: 
        Saxpy() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()) {};

        taco::AcceleratorStmt getStmt() const override{ return x(i) = x(i) + y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(x), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override{return "cblas_saxpy";}
        bool checkerFunction(IndexStmt stmt) const override{return true;}

    private: 
        TensorObject x;
        TensorObject y;
        IndexVar i;
};

class Sdot : public AbstractFunctionInterface{
    public: 
        Sdot() : x(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32))),
                  i(IndexVar()) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return s = x(i) * y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new DimArg(i), 
                                                    new TensorObjectArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(x), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)

                                                };}
        std::string getReturnType() const override {return "float";}
        std::string getFunctionName() const override {return "cblas_sdot";}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
};

class Sgemv : public AbstractFunctionInterface{
    public: 
        Sgemv() : x(TensorObject(Type(taco::Float32, {Dimension(), Dimension()}), Format{Dense, Dense})), 
                  y(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  s(TensorObject(Type(taco::Float32, {Dimension()}), dense)),
                  i(IndexVar()),
                  j(IndexVar()) {};

        // IndexExpr getRHS() const override {return x(i);}
        // IndexExpr getLHS() const override {return x(i);}
        AcceleratorStmt getStmt() const override {return y(i) = x(i, j)*s(j) + y(i);}
        std::vector<Argument> getArguments() const override {return 
                                                {
                                                    new StringLiteral("CblasRowMajor"), 
                                                    new StringLiteral("CblasNoTrans"),
                                                    new DimArg(i), 
                                                    new DimArg(j), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(x), 
                                                    new DimArg(i),
                                                    new TensorObjectArg(s), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new LiteralArg(Datatype(taco::UInt32), 1),
                                                    new TensorObjectArg(y), 
                                                    new LiteralArg(Datatype(taco::UInt32), 1)
                                                };}
        std::string getReturnType() const override {return "void";}
        std::string getFunctionName() const override {return "cblas_sgemv";}

    private: 
        TensorObject x;
        TensorObject y;
        TensorObject s;
        IndexVar i;
        IndexVar j;
};


#endif 