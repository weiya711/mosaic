#ifndef ACCEL_INTERFACE_H
#define ACCEL_INTERFACE_H

#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/lower/iterator.h"
#include "taco/type.h"

// #include "taco/index_notation/internal_args.h"


namespace taco {

struct TransferTypeArgs;
struct TransferWithArgs;
struct AbstractFunctionInterface;
class TensorVar;
template <typename CType>
class Tensor;


enum ArgType {DIM, TENSORVAR, TENSOR_OBJECT, TENSOR, EXPR, LITERAL, USER_DEFINED, DECLVAR, UNKNOWN, DIMLIST, DATA_ARRAY, STRING, DECLVAR_ADDR, TENSOR_ADDR, CAST};

struct TransferTypeArgs : public util::Manageable<TransferTypeArgs>{

    TransferTypeArgs() : argType(UNKNOWN) {}
    TransferTypeArgs(ArgType argType) : argType(argType) {}

    virtual ~TransferTypeArgs() = default;
    virtual void lower() const {  };

    virtual std::ostream& print(std::ostream& os) const {
        os << "Printing a TransferTypeArg" << std::endl;
        return os;
    };

    ArgType argType;
};

class Argument : public util::IntrusivePtr<const TransferTypeArgs> {
  public: 
    Argument(TransferTypeArgs * arg) : IntrusivePtr(arg) {}
    Argument() : Argument(new TransferTypeArgs())  {}

    template<typename T>
    const T* getNode() const {
      return static_cast<const T*>(ptr);
    }

    const TransferTypeArgs* getNode() const {
      return ptr;
    }

    ArgType getArgType() const;

};

std::ostream& operator<<(std::ostream&,  const Argument&);

struct TensorVarArg : public TransferTypeArgs{
    explicit TensorVarArg(const TensorVar& t) : TransferTypeArgs(TENSORVAR), t(t) {}

    std::ostream& print(std::ostream& os) const override;

    Argument operator=(Argument) const;

    TensorVar t; 
};

struct TensorObjectArg : public TransferTypeArgs{
    explicit TensorObjectArg(const TensorObject& t) : TransferTypeArgs(TENSOR_OBJECT), t(t) {}
    
    std::ostream& print(std::ostream& os) const override;
    Argument operator=(Argument) const;
    TensorObject t; 
    TensorVar tvar;
};

struct irExprArg : public TransferTypeArgs{
    explicit irExprArg(const ir::Expr& irExpr) : TransferTypeArgs(EXPR), irExpr(irExpr) {}

    std::ostream& print(std::ostream& os) const override;

    ir::Expr irExpr; 
};

struct DimList : public TransferTypeArgs{
    explicit DimList(const TensorObject& t) : TransferTypeArgs(DIMLIST), t(t) {}
    explicit DimList(const TensorVar& tvar) : TransferTypeArgs(DIMLIST), tvar(tvar) {}
    std::ostream& print(std::ostream& os) const override;
    TensorObject t; 
    TensorVar tvar;
};

struct DataArray : public TransferTypeArgs{
    explicit DataArray(const TensorObject& t) : TransferTypeArgs(DATA_ARRAY), t(t) {}
    explicit DataArray(const TensorVar& tvar) : TransferTypeArgs(DATA_ARRAY), tvar(tvar) {}
    std::ostream& print(std::ostream& os) const override;
    TensorObject t; 
    TensorVar tvar;
};

struct StringLiteral : public TransferTypeArgs{
    explicit StringLiteral(const std::string& s) : TransferTypeArgs(STRING), s(s) {}
    std::ostream& print(std::ostream& os) const override;
    std::string s; 
};

class Dim{
  public:
    explicit Dim(const IndexVar& indexVar) : indexVar(indexVar) {}
    IndexVar indexVar;
};

struct DimArg : public TransferTypeArgs{
    explicit DimArg(const Dim& dim): TransferTypeArgs(DIM), indexVar(dim.indexVar) {}
    explicit DimArg(const IndexVar& indexVar): TransferTypeArgs(DIM), indexVar(indexVar) {}

    std::ostream& print(std::ostream& os) const override;

    IndexVar indexVar;
};

struct  DeclVar {
    DeclVar(const std::string& typeString, const std::string& name) : typeString(typeString), name(name) {}
    explicit DeclVar(const std::string& typeString) : DeclVar(typeString, util::uniqueName('v')) {} 
    
    std::string getTypeString() const {return typeString;}
    std::string getName() const {return name;}
    Argument operator=(Argument) const;

    std::string typeString;
    std::string name;

};

struct  AddrTensorVar : public TransferTypeArgs{
    explicit AddrTensorVar(const TensorObject& var): TransferTypeArgs(TENSOR_ADDR), var(var) {}
    explicit AddrTensorVar(const TensorVar& tvar): TransferTypeArgs(TENSOR_ADDR), tvar(tvar) {}
    TensorObject var;
    TensorVar tvar;

};

struct AddrDeclVarArg : public TransferTypeArgs {

  explicit AddrDeclVarArg(const DeclVar& var): TransferTypeArgs(DECLVAR_ADDR), var(var) {}
  std::ostream& print(std::ostream& os) const override;
  DeclVar var; 

};

struct DeclVarArg : public TransferTypeArgs {

  explicit DeclVarArg(const DeclVar& var): TransferTypeArgs(DECLVAR), var(var) {}
  std::ostream& print(std::ostream& os) const override;
  DeclVar var; 

};

struct TensorArg : public TransferTypeArgs{
    template <typename CType>
    explicit TensorArg(const Tensor<CType>& tensor) : 
    TransferTypeArgs(TENSOR), irExpr(ir::Var::make(tensor.getName(), tensor.getComponentType(),true, true)) {}

    std::ostream& print(std::ostream& os) const override;

    ir::Expr irExpr; 
};


struct LiteralArg : public TransferTypeArgs{
    template <typename T> 
    LiteralArg(Datatype datatype, T val) 
      : TransferTypeArgs(LITERAL), datatype(datatype) {
        this->val = malloc(sizeof(T));
        *static_cast<T*>(this->val) = val;
    }

    ~LiteralArg() {
      free(val);
    }

    template <typename T> T getVal() const {
      return *static_cast<T*>(val);
    }

    std::ostream& print(std::ostream& os) const override;

    void * val; 
    Datatype datatype;
};

struct CastArg : public TransferTypeArgs{
  CastArg(const Argument& argument, const std::string& cast) : TransferTypeArgs(CAST), argument(argument), cast(cast) {}
 
  Argument argument;
  std::string cast;
};

struct TransferWithArgs : public TransferTypeArgs{
    TransferWithArgs() = default;

    TransferWithArgs(const std::string& name, const std::string& returnType , const std::vector<Argument>& args) : TransferTypeArgs(USER_DEFINED), name(name), returnType(returnType), args(args) {
    }

    TransferWithArgs(const std::string& name, const std::string& returnType , const std::vector<Argument>& args, const Argument& returnStore) 
    : TransferTypeArgs(USER_DEFINED), name(name), returnType(returnType), args(args), returnStore(returnStore) {
    }

    TransferWithArgs(const TransferWithArgs& func, const Argument& returnStore) : TransferTypeArgs(USER_DEFINED), name(func.name), returnType(func.returnType), args(func.args), returnStore(returnStore) {
    }

    std::string getName() const { return name; };
    std::string getReturnType() const { return returnType; };
    std::vector<Argument> getArgs() const { return args; };
    Argument getReturnStore() const {return returnStore;};

    void lower() const;

    std::ostream& print(std::ostream& os) const;
    
    std::string name;
    std::string returnType;
    std::vector<Argument> args; 
    Argument returnStore;

};


std::ostream& operator<<(std::ostream&, const TransferWithArgs&);

inline void addArg(std::vector<Argument>& argument, const TensorVar& t) { argument.push_back(new TensorVarArg(t)); }
inline void addArg(std::vector<Argument>& argument, const TensorObject& t) { argument.push_back(new TensorObjectArg(t)); }
inline void addArg(std::vector<Argument>& argument, const Argument&  arg) { argument.push_back(arg); }
inline void addArg(std::vector<Argument>& argument, TransferWithArgs arg) { argument.push_back(new TransferWithArgs(arg.getName(), arg.getReturnType(), arg.getArgs())); }
inline void addArg(std::vector<Argument>& argument, const Dim& dim) { argument.push_back(new DimArg(dim)); }
inline void addArg(std::vector<Argument>& argument, const int32_t& integer) { argument.push_back(new LiteralArg(Datatype(UInt32), integer)); }
inline void addArg(std::vector<Argument>& argument, const DeclVar& var) { argument.push_back(new DeclVarArg(var)); }
inline void addArg(std::vector<Argument>& argument, const DimList& var) { argument.push_back(new DimList(var)); }
inline void addArg(std::vector<Argument>& argument, const DataArray& var) { argument.push_back(new DataArray(var)); }
inline void addArg(std::vector<Argument>& argument, const StringLiteral& s) { argument.push_back(new StringLiteral(s)); }
inline void addArg(std::vector<Argument>& argument, const AddrDeclVarArg& var) { argument.push_back(new AddrDeclVarArg(var)); }
inline void addArg(std::vector<Argument>& argument, const CastArg& var) { argument.push_back(new CastArg(var)); }

template <typename CType>
inline void addArg(std::vector<Argument>& argument, const Tensor<CType>& t) { argument.push_back(new TensorArg(t)); }

template <typename T, typename ...Next>
void addArg(std::vector<Argument>& argument, T first, Next...next){
    addArg(argument, first);
    addArg(argument, (next)...); 
}


class TransferLoad{
  public:
    TransferLoad() = default;
    TransferLoad(const std::string& name, const std::string& returnType) : name(name), returnType(returnType) {};

    template <typename Exprs> 
    Argument operator()(Exprs expr)
    { 
      std::vector<Argument> argument;
      addArg(argument, expr);
      return new TransferWithArgs(name, returnType, argument);
    }

    template <typename FirstT, typename ...Args>
    Argument operator()(FirstT first, Args...remaining){
      std::vector<Argument> argument;
      addArg(argument, first, remaining...);
      return new TransferWithArgs(name, returnType, argument); 
    }

    Argument operator()(){
      std::vector<Argument> argument;
      return new TransferWithArgs(name, returnType, argument); 
    }

  private:
    std::string name;
    std::string returnType; 
};

class TransferStore{
  public:
    TransferStore() = default;
    TransferStore(const std::string& name, const std::string& returnType) : name(name), returnType(returnType){};
    

    template <typename Exprs> 
    Argument operator()(Exprs expr)
    { 
      std::vector<Argument> argument;
      addArg(argument, expr);
      return new TransferWithArgs(name, returnType, argument);
    }

    template <typename FirstT, typename ...Args>
    Argument operator()(FirstT first, Args...remaining){
      std::vector<Argument> argument;
      addArg(argument, first, remaining...);
      return new TransferWithArgs(name, returnType, argument);
    }

    Argument operator()(){
      std::vector<Argument> argument;
      return new TransferWithArgs(name, returnType, argument); 
    }

  private:
    std::string name;
    std::string returnType; 
};

class TransferType{
  public:
    TransferType() = default;
    TransferType(std::string name, taco::TransferLoad transferLoad, taco::TransferStore transferStore);
  private:
    struct Content;
    std::shared_ptr<Content> content;

};


class ForeignFunctionDescription {
  public: 
    ForeignFunctionDescription() = default;

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,  const std::vector<Argument>& args, 
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) 
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args), temporaries(temporaries), checker(checker) {};
    
    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,  const std::vector<Argument>& args, 
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker,  const  std::map<TensorVar, std::set<std::string>>& propertites)
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args), temporaries(temporaries), checker(checker), propertites(propertites) {};

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker) 
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), temporaries(temporaries), checker(checker) {};

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs,
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker,
                                const  std::map<TensorVar, std::set<std::string>>& propertites)
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), temporaries(temporaries), checker(checker), propertites(propertites) {};

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, std::pair<IndexExpr, IndexExpr> assign,
                                const std::vector<TensorVar>& temporaries, std::function<bool(taco::IndexStmt)> checker)
                                : functionName(functionName), returnType(returnType), lhs(assign.first), rhs(assign.second), temporaries(temporaries), checker(checker) {};

    ForeignFunctionDescription( const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs, const std::vector<Argument>& args)
                                : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args) {}

    taco::IndexExpr getExpr() const {return rhs;};
    std::vector<Argument> getArgs() const {return args;};
    taco::IndexExpr getLHS() const      {return lhs;};

    template <typename Exprs> 
    ForeignFunctionDescription operator()(Exprs expr)
    {  std::vector<Argument> argument;
      addArg(argument, expr);
      return ForeignFunctionDescription(functionName, returnType, lhs, rhs, argument, temporaries, checker);
    }

    template <typename FirstT, typename ...Args>
    ForeignFunctionDescription operator()(FirstT first, Args...remaining){
        std::vector<Argument> argument;
        addArg(argument, first, remaining...);
        return ForeignFunctionDescription(functionName, returnType, lhs, rhs, argument, temporaries, checker);
    }

    std::string functionName;
    std::string returnType;
    taco::IndexExpr lhs;
    taco::IndexExpr rhs;
    std::vector<Argument> args;
    std::vector<taco::TensorVar> temporaries;
    std::function<bool(taco::IndexStmt)> checker;
    std::map<TensorVar, std::set<std::string>> propertites;

};

std::ostream& operator<<(std::ostream&, const ForeignFunctionDescription&);


class AcceleratorDescription {
  public:
    AcceleratorDescription(const std::vector<ForeignFunctionDescription>& funcDescriptions) : funcDescriptions(funcDescriptions) {};
    AcceleratorDescription(const TransferType& kernelTransfer, const std::vector<ForeignFunctionDescription>& funcDescriptions) : kernelTransfer(kernelTransfer), funcDescriptions(funcDescriptions) {};
    AcceleratorDescription(const TransferType& kernelTransfer, const std::vector<ForeignFunctionDescription>& funcDescriptions, const std::string& includeFile)
                           : kernelTransfer(kernelTransfer), funcDescriptions(funcDescriptions), includeFile(includeFile) {};

    std::vector<ForeignFunctionDescription> getFuncDescriptions() const {return funcDescriptions;};
    TransferType kernelTransfer;
    std::vector<ForeignFunctionDescription> funcDescriptions;
    //files to include
    std::string includeFile;

};

class ConcreteAccelerateCodeGenerator {
  public: 
    // TODO: Add some error checking here so users 
    // dont pass in any legal IndexExpr
    ConcreteAccelerateCodeGenerator() = default;

    ConcreteAccelerateCodeGenerator(const std::string& functionName, const std::string& returnType, const taco::IndexExpr& lhs, const taco::IndexExpr& rhs, const std::vector<Argument>& args, 
                                    const std::vector<Argument>& callBefore, const std::vector<Argument>& callAfter)
                                    : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), args(args), callBefore(callBefore), callAfter(callAfter) {}

    ConcreteAccelerateCodeGenerator(const std::string& functionName, const std::string& returnType, taco::IndexExpr lhs, taco::IndexExpr rhs, const std::vector<Argument>& callBefore,
                                    const std::vector<Argument>& callAfter)
                                    : functionName(functionName), returnType(returnType), lhs(lhs), rhs(rhs), callBefore(callBefore), callAfter(callAfter) {}
  
    taco::IndexExpr getExpr() const     {return rhs;};
    taco::IndexExpr getLHS() const      {return lhs;};
    taco::IndexExpr getRHS() const      {return rhs;};
    std::vector<Argument> getArguments() const {return args;};
    std::string getReturnType() const   {return returnType;};
    std::string getFunctionName() const {return functionName;};
    std::vector<Argument> getCallBefore() const {return callBefore;};
    std::vector<Argument> getCallAfter() const {return callAfter;};

    template <typename Exprs> 
    ConcreteAccelerateCodeGenerator operator()(Exprs expr)
    {  std::vector<Argument> argument;
      addArg(argument, expr);
      return ConcreteAccelerateCodeGenerator(functionName, returnType, lhs, rhs, argument, callBefore, callAfter);
    }

    template <typename FirstT, typename ...Args>
    ConcreteAccelerateCodeGenerator operator()(FirstT first, Args...remaining){
        std::vector<Argument> argument;
        addArg(argument, first, remaining...);
        return ConcreteAccelerateCodeGenerator(functionName, returnType, lhs, rhs, argument, callBefore, callAfter);
    }

    // ConcreteAccelerateCodeGenerator& operator=(const ConcreteAccelerateCodeGenerator& concreteAccelerateCodeGenerator);
  private:
      std::string functionName;
      std::string returnType;
      taco::IndexExpr lhs;
      taco::IndexExpr rhs;
      std::vector<Argument> args;
      std::vector<Argument> callBefore;
      std::vector<Argument> callAfter;

};

std::ostream& operator<<(std::ostream&,  const ConcreteAccelerateCodeGenerator&);


class FunctionInterface : public util::IntrusivePtr<const AbstractFunctionInterface> {
  public: 
    FunctionInterface() : IntrusivePtr(nullptr) {}
    FunctionInterface(AbstractFunctionInterface * interface) : IntrusivePtr(interface) {}

    const AbstractFunctionInterface* getNode() const {
        return ptr;
    }
};


struct AbstractFunctionInterface :  public util::Manageable<AbstractFunctionInterface>{
    AbstractFunctionInterface() = default;
    virtual ~AbstractFunctionInterface() = default;

    virtual taco::AcceleratorStmt getStmt()    const  = 0; 
    virtual std::vector<Argument> getArguments() const= 0;
    virtual std::string getReturnType()   const = 0;
    virtual std::string getFunctionName() const = 0;
    // virtual std::vector<DeclVar> getDecelerations() const {return {};}

    /// call any functions before main function (useful to call "void"
    /// functions)
    virtual std::vector<Argument> callBefore() const {return {};}

    /// call any functions after main function (useful for error checking)
    virtual std::vector<Argument> callAfter() const {return {};}
    virtual bool checkerFunction(IndexStmt stmt) const {return true;}
    

};

}
#endif