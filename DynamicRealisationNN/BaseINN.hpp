#pragma once
#include<type_traits>

/*The structure responsible for different arguments - Структура отвечающая за разные аргументы*/
struct BaseArgs 
{ 
	virtual ~BaseArgs() = 0; 
};
BaseArgs::~BaseArgs() {}

/*The base class for all interfaces NN`s components - Базовый класс отвечающий за все компоненты нейронной сети*/
class BaseComponentsNN 
{ 
public: 
	virtual ~BaseComponentsNN() = 0; 
};
BaseComponentsNN::~BaseComponentsNN() {}

/*The base interface of all realisations savers - Базовый интефейс для всех реализаций классов хранителей*/
class IBaseSaver : public BaseComponentsNN
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ArgsSaver : public BaseArgs {};
	virtual void save(ArgsSaver * args) = 0;
};
/*The base interface of all realisations loaders - Базовый интефейс для всех реализаций классов всех загрузчиков*/
class IBaseLoader : public BaseComponentsNN
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ArgsLoader : public BaseArgs {};
	virtual void load(ArgsLoader * args) = 0;
};


/*The base interface of all realisations compute block - Базовый интефейс для всех реализаций классов всех блоков вычислений*/
class IComputeBlock : public BaseComponentsNN {
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ValuesForCompute : public BaseArgs 
	{ 
	struct Weights{};
	struct Bias{};
	};
	virtual void setValuesForCompute(ValuesForCompute * values) = 0;
	virtual void compute() = 0;

};
/*The base interface of all realisations compute block for train - Базовый интефейс для всех реализаций классов всех блоков вычислений для обучения*/
class ITrainableComputeBlock : IComputeBlock {
protected:
	/*The struct for intermediate values from computing for future trining - Структура для промежуточных значений для будущего обучения*/
	struct IntermediateValues : public BaseArgs {};
public:
	/*The struct for gradients learning for optimizing - Структура для градиентов обучения для оптимизации*/
	struct Gradients : public ValuesForCompute {};
	/*The class that updating values for compute - Класс, который улучшает значения для вычислений*/
	class Optimizer : public BaseComponentsNN 
	{
	public:
		struct ValuesForOptimizer : public BaseArgs {};
		virtual void setValuesForOptimize(const Gradients* gradients, ValuesForCompute* values_for_compute) = 0;
		virtual void optimize(ValuesForOptimizer * values) = 0;
	};
	virtual Gradients backward() = 0;
	virtual void optimize(Gradients * gradients, Optimizer::ValuesForOptimizer * values) = 0;
	virtual void setOptimizer(Optimizer * optimizer) = 0;
};

/*The base interface of all realisations randomizers - Базовый интефейс для всех реализаций классов всех рандомайзеров*/
class IBaseRandomizer : public BaseComponentsNN
{
public:
	virtual void random(IComputeBlock::ValuesForCompute* args) = 0;
};

/*The base interface of all INN - Основа всех интерфейсов нейронных сетей*/
class IBaseNN
{
protected:
	virtual void forward() = 0;
public:
	struct IOutputValue : public BaseArgs {};
	struct IInputValue : public BaseArgs {};
	virtual void inference() = 0;
	virtual void setInputStates(const IInputValue* input_state) = 0;
	virtual IOutputValue getOutputStates() = 0;
	virtual void load() = 0;
	virtual void setLoader(IBaseLoader* loader) = 0;
	virtual void setComputeBlock(IComputeBlock* compute_block) = 0;
	virtual ~IBaseNN() = 0;
};
IBaseNN::~IBaseNN() {}
/*The base interface of all INN with train - Основа всех интерфейсов нейронных сетей с обучением*/
class IBaseTrainableNN : public IBaseNN {
public:
	virtual void save() = 0;
	virtual void setSaver(IBaseSaver* saver) = 0;
	virtual void setRandomValues() = 0;
	virtual void setValuesRandomizer(IBaseRandomizer* randomaizer) = 0;
	virtual void optimize(ITrainableComputeBlock::Optimizer::ValuesForOptimizer * values) = 0;
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
