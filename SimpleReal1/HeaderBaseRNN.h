#pragma once
#include "HeaderBaseNN.h"

namespace MyNN {
	namespace RNN{
		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class IComputeBlockRNN : public IComputeBlockNN<T, Enable> {
			protected:
				struct NState {};
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T, Enable>> in basecompblockrnn*/
				//inline virtual void nStepCalculation(const typename IComputeBlockNN<T, Enable>::ValuesForCompute * values_for_compute, const NState * n_state, std::uint64_t number_n) = 0;
				virtual void allStepsCalculation() = 0;
		};
		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class ITrainableComputeBlockRNN : public ITrainableComputeBlockNN<T, Enable>, public IComputeBlockRNN<T, Enable> {};

		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class ComputeBlockRNN : public ComputeBlockNN<T, Enable>, public IComputeBlockRNN<T, Enable> {
		protected:
			std::unique_ptr<typename IComputeBlockRNN<T, Enable>::NState> n_state_;
			std::uint64_t hidden_size_;
			std::uint64_t max_steps_;
		public:
			ComputeBlockRNN& operator=(const ComputeBlockRNN&);
			ComputeBlockRNN& operator=(ComputeBlockRNN&&) noexcept;
		};
		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class TrainableComputeBlockRNN :
			virtual public ComputeBlockRNN<T, Enable>,
			virtual public TrainableComputeBlockNN<T, Enable>,
			public ITrainableComputeBlockRNN<T, Enable>
		{
		};

		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class BaseRNN : public BaseNN<T, Enable> {};
		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class TrainableBaseRNN : virtual public BaseTrainableNN<T, Enable>, virtual public BaseRNN<T, Enable> {};
	}
}