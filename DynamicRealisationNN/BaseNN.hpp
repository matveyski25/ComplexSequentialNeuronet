#pragma once

#include "BaseINN.hpp"

class BaseNN : public IBaseNN{
protected:
	std::uint64_t input_size;
	std::uint64_t output_size;
};

class BaseTrainableNN : virtual public IBaseTrainableNN, virtual public BaseNN {};

class BaseTrainableComputeBlock : public IComputeBlock {
	
};

class BaseRNN : public BaseNN {
protected:
	std::uint64_t hidden_size;

};