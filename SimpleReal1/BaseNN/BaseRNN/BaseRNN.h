#pragma once
#include "BaseNN.h"


	namespace MyNN::RNN{
		template<typename T, typename Derived>
		struct FeatureComputeBlockRNN : Base::FeatureComputeBlockNN<T> {
			protected:
				//struct NState {};
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T>> in basecompblockrnn*/
				//inline virtual void nStepCalculation(const typename IComputeBlockNN<T>::ValuesForCompute * values_for_compute, const NState * n_state, std::uint64_t number_n) = 0;
				virtual void allStepsCalculation() = 0;
				void nStepsCalculation() {
					static_cast<Derived*>(this)->nStepsCalculationImpl();
				}
		};
		template<typename T, typename Derived>
		struct FeatureTrainableComputeBlockRNN : Base::FeatureTrainableComputeBlockNN<T>, FeatureComputeBlockRNN<T, Derived> {};

		template<typename T>
		struct ContextComputeBlockRNN : Base::ContextComputeBlockNN<T> {
			std::uint64_t hidden_size_;
			std::uint64_t max_steps_;
		};

		template<typename T>
		class ContextTrainableComputeBlockRNN :
			virtual ContextComputeBlockRNN<T>,
			virtual Base::ContextTrainableComputeBlockNN<T>
		{};

		template<typename T, typename FromType, typename ToType>
		struct ContextBaseRNN : Base::ContextBaseNN<T, FromType, ToType> {};
		template<typename T, typename FromType, typename ToType>
		struct ContextBaseTrainableRNN :
		virtual Base::ContextBaseTrainableNN<T, FromType, ToType>,
		virtual ContextBaseRNN<T, FromType, ToType> {};
	}
