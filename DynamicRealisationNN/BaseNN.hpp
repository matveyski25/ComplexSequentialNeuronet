#pragma once

#include "BaseINN.hpp"
#include <memory>

class BaseNN : public IBaseNN
{
protected:
	std::unique_ptr<IBaseLoader> loader;
	std::unique_ptr<InputValue> input_value;
	std::unique_ptr<OutputValue> output_value;
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

class BaseTrainableNN : virtual public IBaseTrainableNN, virtual public BaseNN 
{
protected:
	std::unique_ptr<IBaseSaver> saver;
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
	std::unique_ptr<ArgsLoader> args_loader;
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
	std::unique_ptr<ArgsSaver> args_saver;
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
	std::unique_ptr<ValuesForCompute> values_for_compute;
	std::unique_ptr<IOValues> io_values;
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
	std::unique_ptr<IOptimizer> opimizer;

	std::unique_ptr<Gradients> gradients;
	std::unique_ptr<IntermediateValues> intermediate_values;
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

