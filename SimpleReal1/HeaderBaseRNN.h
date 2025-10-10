#pragma once
#include "HeaderBaseNN.h"

namespace MyNN {
	TEMPLATE_ARITH(T)
	class IComputeBlockRNN : public IComputeBlockNN<T> {
	protected:
		/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T>> in basecompblockrnn*/
		inline virtual LinearAlgebra::BaseRowVector<T> nStepCalculation(std::uint64_t n_step, LinearAlgebra::BaseRowVector<T> n_step_input) = 0; 
		virtual void allStepsCalculation() = 0;
	};
	TEMPLATE_ARITH(T)
	class ITrainableComputeBlockRNN : public ITrainableComputeBlockNN<T>, public IComputeBlockRNN<T> {};

	TEMPLATE_ARITH(T)
	class ComputeBlockRNN : public ComputeBlockNN<T>, public IComputeBlockRNN<T> {
	protected:
		std::uint64_t hidden_size;
	public:
		ComputeBlockRNN& operator=(const ComputeBlockRNN& other) {
			this->hidden_size = other.hidden_size;
			ComputeBlockNN<T>::operator=(other);
		}
		ComputeBlockRNN& operator=(ComputeBlockRNN&& other) {
			this->hidden_size = std::move(other.hidden_size);
			ComputeBlockNN<T>::operator=(std::move(other));
		}
	};
	TEMPLATE_ARITH(T)
	class TrainableComputeBlockRNN : 
		virtual public ComputeBlockRNN<T>, 
		virtual public TrainableComputeBlockNN<T>, 
		public ITrainableComputeBlockRNN<T>
	{
		
	};
}