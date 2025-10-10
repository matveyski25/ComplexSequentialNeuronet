#pragma once


struct IBaseArgsComponent {
	struct ArgsForComponent {};
	struct ArgsForCalculation {};
	void setArgsForComponent();
};


class IOptimizer
{
public:
	struct ValuesForOptimizer {};
	virtual void setValuesForOptimize(const Gradients* gradients, ValuesForCompute* values_for_compute) = 0;
	virtual void optimize(ValuesForOptimizer* values) = 0;
};

/*The base interface of all realisations loaders - Базовый интефейс для всех реализаций классов всех загрузчиков*/
class IBaseLoader
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ArgsLoader {};
	virtual void load(IComputeBlock* compute_block, ArgsLoader* args) = 0;
};

/*The base interface of all realisations compute block - Базовый интефейс для всех реализаций классов всех блоков вычислений*/
class IComputeBlock {
public:
	/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
	struct ValuesForCompute
	{
		struct Weights {};
		struct Bias {};
	};
	struct IOValues
	{
		struct IValues {};
		struct OValues {};
	};
	virtual void setValuesForCompute(ValuesForCompute* values) = 0;
	virtual void setIOValues(IOValues* io_values) = 0;
	virtual void compute() = 0;

};
/*The base interface of all realisations compute block for train - Базовый интефейс для всех реализаций классов всех блоков вычислений для обучения*/
class ITrainableComputeBlock : public IComputeBlock
{
protected:
	/*The struct for intermediate values from computing for future trining - Структура для промежуточных значений для будущего обучения*/
	struct IntermediateValues {};
public:

};


//--------------------------------------------------------------------------------------------------------------------//


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
	

	virtual void load() = 0;
	virtual void setLoader(IBaseLoader* loader) = 0;
};
class ITrainable {
public:
	
	virtual void optimize() = 0;
	virtual void setOptimizer(IOptimizer* optimizer) = 0
};

