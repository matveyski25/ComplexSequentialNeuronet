#pragma once
#include "HeaderBaseNN.h"

namespace MyNN {
	namespace RNN{
		template<typename T>
		class IComputeBlockRNN : public IComputeBlockNN<T> {
			protected:
				struct NState {};
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T>> in basecompblockrnn*/
				//inline virtual void nStepCalculation(const typename IComputeBlockNN<T>::ValuesForCompute * values_for_compute, const NState * n_state, std::uint64_t number_n) = 0;
				virtual void allStepsCalculation() = 0;
		};
		template<typename T>
		class ITrainableComputeBlockRNN : public ITrainableComputeBlockNN<T>, public IComputeBlockRNN<T> {};

		template<typename T>
		class ComputeBlockRNN : public ComputeBlockNN<T>, public IComputeBlockRNN<T> {
		protected:
			std::unique_ptr<typename IComputeBlockRNN<T>::NState> n_state_;
			std::uint64_t hidden_size_;
			std::uint64_t max_steps_;
		public:
			ComputeBlockRNN& operator=(const ComputeBlockRNN&);
			ComputeBlockRNN& operator=(ComputeBlockRNN&&) noexcept;

			std::uint64_t getMaxSteps() {
				return this->max_steps_;
			}
			void setMaxSteps(std::uint64_t max_steps_) {
				this->max_steps_ = max_steps_;
			}
			std::uint64_t getHiddenSize() {
				return this->hidden_size_;
			}
		};
		template<typename T>
		class TrainableComputeBlockRNN :
			virtual public ComputeBlockRNN<T>,
			virtual public TrainableComputeBlockNN<T>,
			public ITrainableComputeBlockRNN<T>
		{
		};

		template<typename T>
		class BaseRNN : public BaseNN<T> {};
		template<typename T>
		class BaseTrainableRNN : virtual public BaseTrainableNN<T>, virtual public BaseRNN<T> {};
	}
}