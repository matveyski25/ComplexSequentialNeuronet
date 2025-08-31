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


class BaseSaveable 
{
protected:
	std::unique_ptr<IBaseSaver> saver = nullptr;
public:
	void setSaver(IBaseSaver * saver_) {
		this->saver.reset(saver_);
	}
	void setArgsSaver(IBaseSaver::Args * args) {
		this->saver->setArgs(args);
	}
	void setArgsForValuesSaver(IBaseSaver::Args::ArgsForValues * args) {
		this->saver->setArgsForValues(args);
	}
	void setValuesForCalculationSaver(IBaseSaver::Args::ValuesForCalculation* values) {
		this->saver->setValuesForClculation(values);
	}
	void save() {
		this->saver->save();
	}
};

class BaseLoadable
{
protected:
	std::unique_ptr<IBaseLoader> loader = nullptr;
public:
	void setLoader(IBaseLoader* loader_) {
		this->loader.reset(loader_);
	}
	void setArgsLoader(IBaseLoader::Args* args) {
		this->loader->setArgs(args);
	}
	void setArgsForValuesLoader(IBaseLoader::Args::ArgsForValues* args) {
		this->loader->setArgsForValues(args);
	}
	void setValuesForCalculationLoader(IBaseLoader::Args::ValuesForCalculation* values) {
		this->loader->setValuesForClculation(values);
	}
	void load() {
		this->loader->load();
	}
};

class BaseRandomizable
{
protected:
	std::unique_ptr<IBaseRandomizer> randomizer = nullptr;
public:
	void setRandomizer(IBaseRandomizer* randomizer_) {
		this->randomizer.reset(randomizer_);
	}
	void setArgsRandomizer(IBaseRandomizer::Args* args) {
		this->randomizer->setArgs(args);
	}
	void setArgsForValuesRandomizer(IBaseRandomizer::Args::ArgsForValues* args) {
		this->randomizer->setArgsForValues(args);
	}
	void setValuesForCalculationRandomizer(IBaseRandomizer::Args::ValuesForCalculation* values) {
		this->randomizer->setValuesForClculation(values);
	}
	void random() {
		this->randomizer->random();
	}
};

class BaseOptimazable
{
protected:
	std::unique_ptr<IOptimizer> optimizer = nullptr;
public:
	void setOptimizer(IOptimizer* optimizer_) {
		this->optimizer.reset(optimizer_);
	}
	void setArgsOptimizer(IOptimizer::Args* args) {
		this->optimizer->setArgs(args);
	}
	void setArgsForValuesOptimizer(IOptimizer::Args::ArgsForValues* args) {
		this->optimizer->setArgsForValues(args);
	}
	void setValuesForCalculationOptimizer(IOptimizer::Args::ValuesForCalculation* values) {
		this->optimizer->setValuesForClculation(values);
	}
	void optimize() {
		this->optimizer->optimize();
	}
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
