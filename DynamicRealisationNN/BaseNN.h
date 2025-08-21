#pragma once

/*The structure responsible for different arguments - ��������� ���������� �� ������ ���������*/
struct IBaseArgs {};

/*The base interface of all realisations savers - ������� �������� ��� ���� ���������� ������� ����������*/
class IBaseSaver 
{
public:
	/*The structure inherited from the base interface IBaseArgs - ��������� ������������� �� ����� IBaseArgs*/
	struct IArgsSaver : public IBaseArgs {};
	virtual void save(IArgsSaver * args) = 0;
};
/*The base interface of all realisations loaders - ������� �������� ��� ���� ���������� ������� ���� �����������*/
class IBaseLoader
{
public:
	/*The structure inherited from the base interface IBaseArgs - ��������� ������������� �� ����� IBaseArgs*/
	struct IArgsLoader : public IBaseArgs {};
	virtual void load(IArgsLoader * args) = 0;
};
/*The base interface of all realisations randomizers - ������� �������� ��� ���� ���������� ������� ���� �������������*/
class IBaseRandomizer
{
public:
	/*The structure inherited from the base interface IBaseArgs - ��������� ������������� �� ����� IBaseArgs*/
	struct IArgsRandomizer : public IBaseArgs {};
	virtual void random(IArgsRandomizer* args) = 0;
};

/**The base interface of all realisations compute block - ������� �������� ��� ���� ���������� ������� ���� ������ ����������*/
class IComputeBlock {
	/*The structure inherited from the base interface IBaseArgs - ��������� ������������� �� ����� IBaseArgs*/
	struct IValuesForCompute : public IBaseArgs {};
	virtual void setValuesForCompute(IValuesForCompute * values) = 0;
	virtual void compute() = 0;
};

/*The base interface of all INN - ������ ���� ����������� ��������� �����*/
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

/*Feedforward - ������������*/
class IBaseFFNN : public IBaseNN
{

};

/*Colvolutional - ����������*/
class IBaseCNN : public IBaseNN
{

};

/*Transformer - �� ������ ������������*/
class IBaseTNN : public IBaseNN
{

};
/*Graph - �����*/
class IBaseGNN : public IBaseNN
{

};
/*Reccurent - ����������������(������������)*/
class IBaseRNN : public IBaseNN
{

};

