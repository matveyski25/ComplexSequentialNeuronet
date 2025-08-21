#pragma once
#include<type_traits>

/*The structure responsible for different arguments - Структура отвечающая за разные аргументы*/
struct BaseArgs 
{ 
	virtual ~BaseArgs() = 0; 
};
BaseArgs::~BaseArgs() {}

class BaseComponentsNN 
{ 
public: virtual ~BaseComponentsNN() = 0; 
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
/*The base interface of all realisations randomizers - Базовый интефейс для всех реализаций классов всех рандомайзеров*/
class IBaseRandomizer : public BaseComponentsNN
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ArgsRandomizer : public BaseArgs {};
	virtual void random(ArgsRandomizer* args) = 0;
};

/**The base interface of all realisations compute block - Базовый интефейс для всех реализаций классов всех блоков вычислений*/
class IComputeBlock : public BaseComponentsNN {
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ValuesForCompute : public BaseArgs {};
	virtual void setValuesForCompute(ValuesForCompute * values) = 0;
	virtual void compute() = 0;

};

/*The base interface of all INN - Основа всех интерфейсов нейронных сетей*/
class IBaseNN
{
protected:
	virtual void allStateCalculation() = 0;
public:
	struct IOutputValue : public BaseArgs {};
	struct IInputValue : public BaseArgs {};
	virtual void inference() = 0;
	virtual void setInputStates(IInputValue* input_state) = 0;
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
};


/*Feedforward - Полносвязные*/
template<typename BaseT>
class IBaseFFNN : public BaseT { static_assert(std::is_base_of_v<IBaseNN, BaseT>, "BaseT must be the class NN - BaseT должен быть классом нейронной сети"); };

/*Colvolutional - Сверточные*/
template<typename BaseT>
class IBaseCNN : public BaseT { static_assert(std::is_base_of_v<IBaseNN, BaseT>, "BaseT must be the class NN - BaseT должен быть классом нейронной сети"); };

/*Transformer - На основе трансформера*/
template<typename BaseT>
class IBaseTNN : public BaseT { static_assert(std::is_base_of_v<IBaseNN, BaseT>, "BaseT must be the class NN - BaseT должен быть классом нейронной сети"); };

/*Graph - Графы*/
template<typename BaseT>
class IBaseGNN : public BaseT { static_assert(std::is_base_of_v<IBaseNN, BaseT>, "BaseT must be the class NN - BaseT должен быть классом нейронной сети"); };

/*Reccurent - Последовательные(реккурентные)*/
template<typename BaseT>
class IBaseRNN : public BaseT { static_assert(std::is_base_of_v<IBaseNN, BaseT>, "BaseT must be the class NN - BaseT должен быть классом нейронной сети"); };

using InterfaceFFNN = IBaseFFNN<IBaseNN>;
using InterfaceCNN = IBaseCNN<IBaseNN>;
using InterfaceTNN = IBaseTNN<IBaseNN>;
using InterfaceGNN = IBaseGNN<IBaseNN>;
using InterfaceRNN = IBaseRNN<IBaseNN>;

using TrainableInterfaceFFNN = IBaseFFNN<IBaseTrainableNN>;
using TrainableInterfaceCNN = IBaseCNN<IBaseTrainableNN>;
using TrainableInterfaceTNN = IBaseTNN<IBaseTrainableNN>;
using TrainableInterfaceGNN = IBaseGNN<IBaseTrainableNN>;
using TrainableInterfaceRNN = IBaseRNN<IBaseTrainableNN>;