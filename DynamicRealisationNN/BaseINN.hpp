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
  
/*The structure responsible for different arguments 
- Структура отвечающая за разные аргументы*/
struct BaseArgs : virtual public PolymorphicBase {};

/*The base class for all interfaces NN`s components 
- Базовый класс отвечающий за все компоненты нейронной сети*/
class BaseComponentsNN : virtual public PolymorphicBase
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct Args : public BaseArgs
	{
		struct ValuesForCalculation {};
		struct ArgsForValues {};
	};
	virtual void setArgs(Args* args) = 0;
	virtual void setValuesForClculation(Args::ValuesForCalculation* values) = 0;
	virtual void setArgsForValues(Args::ArgsForValues * args) = 0;
};

/*The base interface of all realisations savers 
- Базовый интефейс для всех реализаций классов хранителей*/
class IBaseSaver : public BaseComponentsNN
{
public:
	virtual void save() = 0;
};
/*The base interface of all realisations loaders 
- Базовый интефейс для всех реализаций классов всех загрузчиков*/
class IBaseLoader : public BaseComponentsNN
{
public:
	virtual void load() = 0;
};
/*The base interface of all realisations randomizers 
- Базовый интефейс для всех реализаций классов всех рандомайзеров*/
class IBaseRandomizer : public BaseComponentsNN
{
public:
	virtual void random() = 0;
};
/*The base interface of all realisations optimizers for gradients 
- Базовый интефейс для всех реализаций классов всех оптимизаторов для градиентов*/
class IOptimizer : public BaseComponentsNN
{
public:
	virtual void optimize() = 0;
};

/*The base interface of all realisations compute block - Базовый интефейс для всех реализаций классов всех блоков вычислений*/
class IComputeBlock : public BaseComponentsNN 
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ValuesForCompute : public BaseArgs 
	{ 
	struct Weights{};
	struct Bias{};
	};
	struct IOValues 
	{
		struct IValues {};
		struct OValues {};
	};
	virtual void setValuesForCompute(ValuesForCompute * values) = 0;
	virtual void setIOValues(IOValues * io_values) = 0;
	virtual void compute() = 0;

};
/*The base interface of all realisations compute block for train - Базовый интефейс для всех реализаций классов всех блоков вычислений для обучения*/
class ITrainableComputeBlock : public IComputeBlock 
{
protected:
	/*The struct for intermediate values from computing for future trining - Структура для промежуточных значений для будущего обучения*/
	struct IntermediateValues : public BaseArgs {};
public:
	/*The struct for gradients learning for optimizing - Структура для градиентов обучения для оптимизации*/
	struct Gradients : public ValuesForCompute {};
	virtual Gradients backward() = 0;
};


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
	virtual const OutputValue * getOutputStates() = 0;
	virtual void setComputeBlock(IComputeBlock* compute_block) = 0;
};

/*The base interface of all INN with train - Основа всех интерфейсов нейронных сетей с обучением*/
class IBaseTrainableNN : public IBaseNN {};