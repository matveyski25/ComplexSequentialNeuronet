#pragma once

/*The structure responsible for different arguments - Структруа отвечающая за разные аргументы*/
struct IBaseArgs {};

/*The base interface of all realisations savers - Базовый интефейс для всех реализаций классов хранителей*/
class IBaseSaver 
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структруа наследующаяся от общей IBaseArgs*/
	struct IArgsSaver : public IBaseArgs {};
	virtual void save(IArgsSaver * args) = 0;
};
/*The base interface of all realisations loaders - Базовый интефейс для всех реализаций классов всех загрузчиков*/
class IBaseLoader
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структруа наследующаяся от общей IBaseArgs*/
	struct IArgsLoader : public IBaseArgs {};
	virtual void load(IArgsLoader * args) = 0;
};
/*The base interface of all realisations randomizers - Базовый интефейс для всех реализаций классов всех рандомайзеров*/
class IBaseRandomizer
{
public:
	/*The structure inherited from the base interface IBaseArgs - Структруа наследующаяся от общей IBaseArgs*/
	struct IArgsRandomizer : public IBaseArgs {};
	virtual void random(IArgsRandomizer* args) = 0;
};

/**The base interface of all realisations compute block - Базовый интефейс для всех реализаций классов всех блоков вычислений*/
class IComputeBlock {
	/*The structure inherited from the base interface IBaseArgs - Структруа наследующаяся от общей IBaseArgs*/
	struct IValuesForCompute : public IBaseArgs {};
	virtual void setValuesForCompute(IValuesForCompute * values) = 0;
	virtual void compute() = 0;
};

/*The base interface of all INN - Основа всех интерфейсов нейронных сетей*/
class IBaseNN
{
protected:
	virtual void allStateCalculation() = 0;
public:
	virtual void inference() = 0;
	virtual void setInputStates() = 0;
	virtual void getOutputStates() = 0;
	virtual void save() = 0;
	virtual void load() = 0;
	virtual void setRandomValues() = 0;
	virtual void setSaver(IBaseSaver* saver) = 0;
	virtual void setLoader(IBaseLoader* loader) = 0;
	virtual void setValuesRandomizer(IBaseRandomizer* randomaizer) = 0;
	virtual void setComputeBlock(IComputeBlock* compute_block) = 0;
};

/*Feedforward - Полносвязные*/
class IBaseFFNN : public IBaseNN
{

};

/*Colvolutional - Сверточные*/
class IBaseCNN : public IBaseNN
{

};

/*Transformer - На основе трансформера*/
class IBaseTNN : public IBaseNN
{

};
/*Graph - Графы*/
class IBaseGNN : public IBaseNN
{

};
/*Reccurent - Последовательные(реккурентные)*/
class IBaseRNN : public IBaseNN
{

};

