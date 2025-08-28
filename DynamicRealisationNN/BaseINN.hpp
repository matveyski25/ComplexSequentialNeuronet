#pragma once
#include<type_traits>
#include<string>

class PolymorphicBase {
public:
	virtual ~PolymorphicBase() = 0;
	virtual std::string getTypeRealization() = 0;
};
PolymorphicBase::~PolymorphicBase() {}
std::string PolymorphicBase::getTypeRealization() {
	return "PolymorphicBase";
}

struct IBaseArgsComponent {
	struct ArgsForComponent{};
	struct ArgsForCalculation {};
};

/*The base interface of all realisations savers - Базовый интефейс для всех реализаций классов хранителей*/
class ISaveable {
public:
	struct ArgsForSave {

	};
	struct ISaver {
		
	};
	void save();
	void setSaver(ISaver* saver);
	void setArgsForSave(ArgsForSave* args);
};
class ILoadable
{
public:
	/*The base interface of all realisations loaders - Базовый интефейс для всех реализаций классов всех загрузчиков*/
	class IBaseLoader
	{
	public:
		/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
		struct ArgsLoader {};
		virtual void load(IComputeBlock* compute_block, ArgsLoader* args) = 0;
	};

	virtual void load() = 0;
	virtual void setLoader(IBaseLoader* loader) = 0;
};
class ITrainable {
public:
	class IOptimizer
	{
	public:
		struct ValuesForOptimizer {};
		virtual void setValuesForOptimize(const Gradients* gradients, ValuesForCompute* values_for_compute) = 0;
		virtual void optimize(ValuesForOptimizer* values) = 0;
	};
	virtual void optimize() = 0;
	virtual void setOptimizer(IOptimizer* optimizer) = 0
};

/*The base interface of all realisations compute block - Базовый интефейс для всех реализаций классов всех блоков вычислений*/
class IComputeBlock {
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ValuesForCompute
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
class ITrainableComputeBlock : IComputeBlock {
protected:
	/*The struct for intermediate values from computing for future trining - Структура для промежуточных значений для будущего обучения*/
	struct IntermediateValues {};
public:
	
};

/*The base interface of all realisations randomizers - Базовый интефейс для всех реализаций классов всех рандомайзеров*/
class IBaseRandomizer : public BaseComponentsNN
{
public:
	virtual void random(IComputeBlock::ValuesForCompute* args) = 0;
};

/*The base interface of all INN - Основа всех интерфейсов нейронных сетей*/
class IBaseNN : public PolymorphicBase
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
