#include "LSTM.h"

namespace MyNN {
	namespace RNN {
		template<typename T, typename Enable>
		LSTM<T, Enable>::LSTM()
		{
			this->compute_block_ = std::make_unique<typename LSTM<T, Enable>::DefaultComputeBlockOneH>();
		}

		template<typename T, typename Enable>
		__forceinline void LSTM<T, Enable>::DefaultComputeBlockOneH::nStepCalculation(
			const typename LSTM<T, Enable>::DefaultComputeBlockOneH::ValuesForCompute* __restrict values_for_compute,
			typename LSTM<T, Enable>::DefaultComputeBlockOneH::NState* __restrict n_state,
			const LinearAlgebra::BaseRowVector<T, Enable>& x_n)
		{
			const std::uint64_t& H = this->hidden_size_;

			const LinearAlgebra::BaseRowVector<T, Enable>& c_n_l = n_state->n_cell_state;
			const LinearAlgebra::BaseRowVector<T, Enable>& h_n_l = n_state->n_hidden_state;

			const LinearAlgebra::BaseMatrix<T, Enable>& W = values_for_compute->W;
			const LinearAlgebra::BaseMatrix<T, Enable>& U = values_for_compute->U;
			const LinearAlgebra::BaseRowVector<T, Enable>& B = values_for_compute->B;

			n_state->tmp_Z = (x_n * W + h_n_l * U).noalias();
			n_state->tmp_Z += B;

			n_state->tmp_f = FunctionsActivate::baseSigmoid(n_state->tmp_Z.leftCols(H));
			n_state->tmp_i = FunctionsActivate::baseSigmoid(n_state->tmp_Z.middleCols(H, H));
			n_state->tmp_c_bar = FunctionsActivate::baseTanh(n_state->tmp_Z.middleCols(2 * H, H));
			n_state->tmp_o = FunctionsActivate::baseSigmoid(n_state->tmp_Z.rightCols(H));

			LinearAlgebra::BaseRowVector<T, Enable> new_c_n = n_state->tmp_f.array() * c_n_l.array() + n_state->tmp_i.array() * n_state->tmp_c_bar.array();
			LinearAlgebra::BaseRowVector<T, Enable> new_h_n = n_state->tmp_o.array() * FunctionsActivate::baseTanh(new_c_n).array();


			n_state->n_cell_state = new_c_n;
			n_state->n_hidden_state = new_h_n;
		}
		template<typename T, typename Enable>
		__forceinline void LSTM<T, Enable>::DefaultComputeBlockOneH::allStepsCalculation()
		{
			const std::uint64_t& H = this->hidden_size_;
			std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

			NState* n_state = static_cast<NState*>(this->n_state_.get());
			const ValuesForCompute* values_for_compute = static_cast<const ValuesForCompute*>(this->values_for_compute.get());

			n_state->setZero(H);

			for (std::uint64_t n = 0; n < number_steps; n++) {
				const LinearAlgebra::BaseRowVector<T, Enable>& x_n = this->input_state_.row(n);
				this->nStepCalculation(values_for_compute, n_state, x_n);
			}
		}
		template<typename T, typename Enable>
		LinearAlgebra::BaseMatrix<T, Enable> LSTM<T, Enable>::DefaultComputeBlockOneH::getOutput()
		{
			return LinearAlgebra::BaseMatrix<T, Enable>(
				static_cast<NState*>(this->n_state_.get())->n_hidden_state
			);
		}
		template<typename T, typename Enable>
		__forceinline void LSTM<T, Enable>::DefaultComputeBlockOneH::compute()
		{
			this->allStepsCalculation();
		}

		template<typename T, typename Enable>
		const typename LSTM<T, Enable>::DefaultComputeBlockOneH::ValuesForCompute* LSTM<T, Enable>::DefaultComputeBlockOneH::getValuesForCompute()
		{
			return this->values_for_compute.get();
		}

		template<typename T, typename Enable>
		void LSTM<T, Enable>::DefaultComputeBlockOneH::setValuesForCompute(const ValuesForCompute* values_for_compute_)
		{
			if (values_for_compute_) {
				if(this->values_for_compute_){
					*(this->values_for_compute_) = *(values_for_compute_);
				}
				else {
					this->values_for_compute_ = std::make_unique<ValuesForCompute>(values_for_compute_);
				}
			}
			else {
				this->values_for_compute_ = nullptr;
			}
		}

		template<typename T, typename Enable>
		void LSTM<T, Enable>::DefaultComputeBlockAllH::allStepsCalculation()
		{
			const std::uint64_t& H = this->hidden_size_;
			std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

			using NSate_ = typename DefaultComputeBlockOneH::NState;
			using ValuesForCompute_ = typename DefaultComputeBlockOneH::ValuesForCompute;

			NSate_* n_state = static_cast<NSate_*>(this->n_state_.get());
			const ValuesForCompute_* values_for_compute = static_cast<const ValuesForCompute_*>(this->values_for_compute.get());

			this->hidden_states_ = LinearAlgebra::BaseMatrix<T, Enable>::Zero(number_steps, H);

			n_state->setZero(H);

			for (std::uint64_t n = 0; n < number_steps; n++) {
				const LinearAlgebra::BaseRowVector<T, Enable>& x_n = this->input_state_.row(n);
				this->nStepCalculation(values_for_compute, n_state, x_n);
				this->hidden_states_.row(n) = n_state->n_hidden_state;
			}
		}
		template<typename T, typename Enable>
		LinearAlgebra::BaseMatrix<T, Enable> LSTM<T, Enable>::DefaultComputeBlockAllH::getOutput()
		{
			return this->hidden_states_;
		}

		template<typename T, typename Enable>
		TrainableLSTM<T, Enable>::TrainableLSTM()
		{
			this->compute_block_ = std::make_unique<typename TrainableLSTM<T, Enable>::DefaultComputeBlockOneH>();
		}

		template<typename T, typename Enable>
		void TrainableLSTM<T, Enable>::DefaultComputeBlockOneH::allStepsCalculation() {
			const std::uint64_t& H = this->hidden_size_;
			std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

			using NState_ = typename TrainableLSTM<T, Enable>::DefaultComputeBlockOneH::NState;
			using ValuesForCompute_ = typename TrainableLSTM<T, Enable>::DefaultComputeBlockOneH::ValuesForCompute;

			IntermediateValues* intermediate_values = static_cast<IntermediateValues*>(this->intermediate_values_);
			NState_* n_state = static_cast<NState_*>(this->n_state_.get());
			const ValuesForCompute_* values_for_compute = static_cast<const ValuesForCompute_*>(this->values_for_compute.get());

			n_state->setZero(H);
			intermediate_values->states_->resize(number_steps);

			for (std::uint64_t n = 0; n < number_steps; n++) {
				const LinearAlgebra::BaseRowVector<T, Enable>& x_n = this->input_state_.row(n);
				this->nStepCalculation(values_for_compute, n_state, x_n);
				intermediate_values->states_[n] = n_state;
			}
		}

		template<typename T, typename Enable>
		LinearAlgebra::BaseMatrix<T, Enable> TrainableLSTM<T, Enable>::DefaultComputeBlockAllH::getOutput() {
			using IntermediateValues_ = typename TrainableLSTM<T, Enable>::DefaultComputeBlockAllH::IntermediateValues;
			IntermediateValues_* intermediate_values = static_cast<IntermediateValues_*>(this->intermediate_values_);
			LinearAlgebra::BaseMatrix<T, Enable> out(intermediate_values->states_.size() - 1, this->hidden_size_);
			for (std::uint64_t i = 0; i < intermediate_values->states_.size() - 1; i++) {
				out.row(i) = intermediate_values->states_[i]->n_hidden_state;
			}
			return out;
		}
	}
}