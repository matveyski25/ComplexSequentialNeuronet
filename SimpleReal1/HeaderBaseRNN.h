#pragma once
#include "HeaderBaseNN.h"

namespace MyNN {
	namespace RNN{
		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		class IComputeBlockRNN : public IComputeBlockNN<T> {
			protected:
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T>> in basecompblockrnn*/
				inline virtual LinearAlgebra::BaseRowVector<T> nStepCalculation(std::uint64_t n_step, LinearAlgebra::BaseRowVector<T> n_step_input) = 0;
				virtual void allStepsCalculation() = 0;
		};
		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		class ITrainableComputeBlockRNN : public ITrainableComputeBlockNN<T>, public IComputeBlockRNN<T> {};

		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
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
		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		class TrainableComputeBlockRNN :
			virtual public ComputeBlockRNN<T>,
			virtual public TrainableComputeBlockNN<T>,
			public ITrainableComputeBlockRNN<T>
		{
		};

		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		class BaseRNN : public BaseNN<T> {};
		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		class TrainableBaseRNN : virtual public BaseTrainableNN<T>, virtual public BaseRNN<T> {};
	}
}