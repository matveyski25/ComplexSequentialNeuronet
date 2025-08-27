#pragma once
#include "BaseNN.hpp"

class BaseRNN : virtual public IBaseRNN,  virtual public BaseNN
{
protected:
	std::uint64_t hidden_size;
public:
	BaseRNN& operator=(const BaseRNN& other) {
		this->BaseNN::operator=(other);
		this->hidden_size = other.hidden_size;
	}
	std::string getTypeRealization() override {
		return "BaseRNN";
	}
};

class BaseTrainableRNN : virtual public BaseRNN, virtual public BaseTrainableNN
{
public:
	std::string getTypeRealization() override {
		return "BaseTrainableRNN";
	}
};