#pragma once
#include<type_traits>
#include<string>
#include <memory>

class PolymorphicBase {
public:
	virtual ~PolymorphicBase() = 0;
	virtual std::string getTypeRealization() = 0;
};
PolymorphicBase::~PolymorphicBase() {}
std::string PolymorphicBase::getTypeRealization() {
	return "PolymorphicBase";
}






/*The base interface of all INN - Основа всех интерфейсов нейронных сетей*/
class IBaseNN : public virtual PolymorphicBase
{
protected:
	virtual void forward() = 0;
public:
	struct OutputValue : public BaseArgs {};
	struct InputValue : public BaseArgs {};
	virtual void inference() = 0;
	virtual void setInputStates(const InputValue* input_state) = 0;
	virtual OutputValue getOutputStates() = 0;
	
	
	virtual void setComputeBlock(IComputeBlock* compute_block) = 0;
};

/*The base interface of all INN with train - Основа всех интерфейсов нейронных сетей с обучением*/
class IBaseTrainableNN : public IBaseNN {};
class IBaseTrainableNN : public IBaseNN {
public:
	
	virtual void setRandomValues() = 0;
	virtual void setValuesRandomizer(IBaseRandomizer* randomaizer) = 0;
	virtual void optimize(ITrainableComputeBlock::IOptimizer::ValuesForOptimizer * values) = 0;
};


/*Feedforward - Полносвязные*/
class IBaseFFNN : public IBaseNN {};

/*Colvolutional - Сверточные*/
class IBaseCNN : public IBaseNN {};

/*Transformer - На основе трансформера*/
class IBaseTNN : public IBaseNN {};

/*Graph - Графы*/
class IBaseGNN : public IBaseNN {};

/*Reccurent - Последовательные(реккурентные)*/
class IBaseRNN : public IBaseNN {};

template<typename Base, typename = std::enable_if_t<std::is_base_of_v<IBaseFFNN, Base>>>
class IBaseTrainableFFNN : virtual public Base, virtual public IBaseTrainableNN {};

template<typename Base, typename = std::enable_if_t<std::is_base_of_v<IBaseCNN, Base>>>
class IBaseTrainableCNN : virtual public Base, virtual public IBaseTrainableNN {};

template<typename Base, typename = std::enable_if_t<std::is_base_of_v<IBaseTNN, Base>>>
class IBaseTrainableTNN : virtual public Base, virtual public IBaseTrainableNN {};

template<typename Base, typename = std::enable_if_t<std::is_base_of_v<IBaseGNN, Base>>>
class IBaseTrainableGNN : virtual public Base, virtual public IBaseTrainableNN {};

template<typename Base, typename = std::enable_if_t<std::is_base_of_v<IBaseRNN, Base>>>
class IBaseTrainableRNN : virtual public Base, virtual public IBaseTrainableNN {};
