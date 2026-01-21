#pragma once
#include "BaseNN.h"


	namespace MyNN::RNN{
		using std::uint64_t;
		template<typename T, typename Context, typename Derived>
		struct FeatureComputeBlockRNN : Base::FeatureComputeBlockNN<T, Context> {
			protected:
				//struct NState {};
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T>> in basecompblockrnn*/
				//inline virtual void nStepCalculation(const typename IComputeBlockNN<T>::ValuesForCompute * values_for_compute, const NState * n_state, std::uint64_t number_n) = 0;
				virtual void allStepsCalculation() = 0;
				void nStepsCalculation(uint64_t step) {
					static_cast<Derived*>(this)->nStepsCalculationImpl(step);
				}
		};
		template<typename T, typename Context, typename Derived>
		struct FeatureTrainableComputeBlockRNN :
		Base::FeatureTrainableComputeBlockNN<T, Context>, FeatureComputeBlockRNN<T, Context, Derived> {};

		template<typename T>
		struct ContextComputeBlockRNN : Base::ContextComputeBlockNN<T> {
			std::uint64_t hidden_size_;
			std::uint64_t max_steps_;
		};

		template<typename T, typename Context>
		class ContextTrainableComputeBlockRNN :
			virtual ContextComputeBlockRNN<T>,
			virtual Base::ContextTrainableComputeBlockNN<T, Context>
		{};

		template<typename T, typename Derived, typename FromType, typename ToType>
		struct ContextBaseRNN : Base::ContextBaseNN<T, Derived, FromType, ToType> {};
		template<typename T, typename Derived, typename FromType, typename ToType>
		struct ContextBaseTrainableRNN :
		virtual Base::ContextBaseTrainableNN<T, Derived, FromType, ToType>,
		virtual ContextBaseRNN<T, Derived, FromType, ToType> {};
	}
