#pragma once

#include "BaseINN.hpp"


class BaseSaveable
{
protected:
	std::unique_ptr<IBaseSaver> saver = nullptr;
public:
	void setSaver(IBaseSaver* saver_) {
		this->saver.reset(saver_);
	}
	void setArgsSaver(IBaseSaver::Args* args) {
		this->saver->setArgs(args);
	}
	void setArgsForValuesSaver(IBaseSaver::Args::ArgsForValues* args) {
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


class BaseNN : public IBaseNN, public BaseLoadable
{
protected:
	std::unique_ptr<InputValue> input_value = nullptr;
	std::unique_ptr<OutputValue> output_value = nullptr;
public:
	BaseNN& operator=(const BaseNN& other) {
		*(this->loader) = *(other.loader);

		*(this->input_value) = *(other.input_value);
		*(this->output_value) = *(other.output_value);
	}
	std::string getTypeRealization() override {
		return "BaseNN";
	}
};

class BaseTrainableNN : 
	virtual public IBaseTrainableNN, 
	virtual public BaseNN, 
	public BaseSaveable,
	public BaseRandomizable, 
	public BaseOptimazable
{
public:
	BaseTrainableNN& operator=(const BaseTrainableNN& other) {
		this->BaseNN::operator=(other);
		*(this->saver) = *(other.saver);
	}
	std::string getTypeRealization() override {
		return "BaseTrainableNN";
	}
};

class BaseLoader : public IBaseLoader
{
protected:
	std::unique_ptr<Args> args_loader = nullptr;
public:
	BaseLoader& operator=(const BaseLoader& other) {
		*(this->args_loader) = *(other.args_loader);
	}
	std::string getTypeRealization() override {
		return "BaseLoader";
	}
};

class BaseSaver : IBaseSaver
{
protected:
	std::unique_ptr<Args> args_saver = nullptr;
public:
	BaseSaver& operator=(const BaseSaver& other) {
		*(this->args_saver) = *(other.args_saver);
	}
	std::string getTypeRealization() override {
		return "BaseSaver";
	}
};

class BaseComputeBlock : public IComputeBlock 
{
protected:
	std::uint64_t input_size;
	std::uint64_t output_size;
	std::unique_ptr<ValuesForCompute> values_for_compute = nullptr;
	std::unique_ptr<IOValues> io_values = nullptr;
public:
	BaseComputeBlock& operator=(const BaseComputeBlock& other) {
		this->input_size = other.input_size;
		this->output_size = other.output_size;
		*(this->values_for_compute) = *(other.values_for_compute);
		*(this->io_values) = *(other.io_values);
	}
	std::string getTypeRealization() override {
		return "BaseComputeBlock";
	}
};

class BaseTrainableComputeBlock : virtual public BaseComputeBlock, virtual public ITrainableComputeBlock 
{
protected:
	std::unique_ptr<IOptimizer> opimizer = nullptr;

	std::unique_ptr<Gradients> gradients = nullptr;
	std::unique_ptr<IntermediateValues> intermediate_values = nullptr;
public:
	BaseTrainableComputeBlock& operator=(const BaseTrainableComputeBlock& other) {
		this->BaseComputeBlock::operator=(other);

		*(this->opimizer) = *(other.opimizer);
		*(this->gradients) = *(other.gradients);
		*(this->intermediate_values) = *(other.intermediate_values);
	}
	std::string getTypeRealization() override {
		return "BaseTrainableComputeBlock";
	}
};

