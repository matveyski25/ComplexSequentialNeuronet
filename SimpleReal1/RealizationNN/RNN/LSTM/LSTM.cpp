#include "LSTM.hpp"
#include "FunctionsActivate.hpp"

namespace MyNN::RNN::LSTM
{
	using std::uint64_t;
	template<typename T>
	void DefaultFeatureComputeBlockOneH<T>::setInput(const BaseMatrix<T>& input) {
		auto & self = static_cast<DefaultFeatureComputeBlockOneH<T>::Context_*>(this);
		self.input_size_ = input.cols();
		self->input_state_ = input;
		assert(self->input_size_ == input.cols());
	}

	template<typename T>
	BaseMatrix<T> DefaultFeatureComputeBlockOneH<T>::getOutput() {
		auto & self = static_cast<DefaultFeatureComputeBlockOneH<T>::Context_*>(this);
		return self->output_state_;
	}

	template<typename T>
	void DefaultFeatureComputeBlockOneH<T>::compute() {
		auto & self = static_cast<DefaultFeatureComputeBlockOneH<T>::Context_*>(this);
		this->allStepsCalculation();
		self->output_state_ = self->n_hidden_state;
	}

	template<typename T>
	void DefaultFeatureComputeBlockOneH<T>::nStepsCalculationImpl(uint64_t step) {
		using FunctionsActivate::baseSigmoid, FunctionsActivate::baseTanh, LinearAlgebra::BaseRowVector;

		auto & self = static_cast<DefaultFeatureComputeBlockOneH<T>::Context_*>(this);

		const std::uint64_t& H = self->hidden_size_;

		const BaseRowVector<T>& c_n_l = self->n_cell_state;
		const BaseRowVector<T>& h_n_l = self->n_hidden_state;

		const BaseMatrix<T>& W = self->W;
		const BaseMatrix<T>& U = self->U;
		const BaseRowVector<T>& B = self->B;

		const BaseRowVector<T>& x_n = self.input_state_.row(step);

		self->tmp_Z = ((x_n * W) + (h_n_l * U)).noalias();
		self->tmp_Z += B;

		self->tmp_f = baseSigmoid(self->tmp_Z.leftCols(H));
		self->tmp_i = baseSigmoid(self->tmp_Z.middleCols(H, H));
		self->tmp_c_bar = baseTanh(self->tmp_Z.middleCols(2 * H, H));
		self->tmp_o = baseSigmoid(self->tmp_Z.rightCols(H));

		BaseRowVector<T> new_c_n = (self->tmp_f.array() * c_n_l.array()) + (self->tmp_i.array() *
			self->tmp_c_bar.array());
		BaseRowVector<T> new_h_n = self->tmp_o.array() * baseTanh(new_c_n).array();


		self->n_cell_state = new_c_n;
		self->n_hidden_state = new_h_n;
	}

	template<typename T>
	void DefaultFeatureComputeBlockOneH<T>::allStepsCalculation() {
		auto & self = static_cast<DefaultFeatureComputeBlockOneH<T>::Context_*>(this);

		const uint64_t& H = self->hidden_size_;
		uint64_t number_steps = std::min(self->input_state_.rows(), self->max_steps_);

		for (uint64_t n = 0; n < number_steps; n++)
		{
			this->nStepsCalculation(n);
		}
	}


	template <typename T>
	void LSTM<T>::DefaultComputeBlockAllH::allStepsCalculation()
	{
		const std::uint64_t& H = this->hidden_size_;
		std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

		using NSate_ = typename DefaultComputeBlockOneH::NState;
		using ValuesForCompute_ = typename DefaultComputeBlockOneH::ValuesForCompute;

		NSate_* n_state = static_cast<NSate_*>(this->n_state_.get());
		const ValuesForCompute_* values_for_compute = static_cast<const ValuesForCompute_*>(this->values_for_compute.
			get());

		this->hidden_states_ = LinearAlgebra::BaseMatrix<T>::Zero(number_steps, H);

		n_state->setZero(H);

		for (std::uint64_t n = 0; n < number_steps; n++)
		{
			const LinearAlgebra::BaseRowVector<T>& x_n = this->input_state_.row(n);
			this->nStepCalculation(values_for_compute, n_state, x_n);
			this->hidden_states_.row(n) = n_state->n_hidden_state;
		}
	}

	template <typename T>
	LinearAlgebra::BaseMatrix<T> LSTM<T>::DefaultComputeBlockAllH::getOutput()
	{
		return this->hidden_states_;
	}

	template <typename T>
	TrainableLSTM<T>::TrainableLSTM()
	{
		this->compute_block_ = std::make_unique<typename TrainableLSTM<T>::DefaultComputeBlockOneH>();
	}

	template <typename T>
	void TrainableLSTM<T>::DefaultComputeBlockOneH::allStepsCalculation()
	{
		const std::uint64_t& H = this->hidden_size_;
		std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

		using NState_ = typename TrainableLSTM<T>::DefaultComputeBlockOneH::NState;
		using ValuesForCompute_ = typename TrainableLSTM<T>::DefaultComputeBlockOneH::ValuesForCompute;

		IntermediateValues* intermediate_values = static_cast<IntermediateValues*>(this->intermediate_values_);
		NState_* n_state = static_cast<NState_*>(this->n_state_.get());
		const ValuesForCompute_* values_for_compute = static_cast<const ValuesForCompute_*>(this->values_for_compute.
			get());

		n_state->setZero(H);
		intermediate_values->states_->resize(number_steps);

		for (std::uint64_t n = 0; n < number_steps; n++)
		{
			const LinearAlgebra::BaseRowVector<T>& x_n = this->input_state_.row(n);
			this->nStepCalculation(values_for_compute, n_state, x_n);
			intermediate_values->states_[n] = n_state;
		}
	}

	template <typename T>
	LinearAlgebra::BaseMatrix<T> TrainableLSTM<T>::DefaultComputeBlockAllH::getOutput()
	{
		using IntermediateValues_ = typename TrainableLSTM<T>::DefaultComputeBlockAllH::IntermediateValues;
		IntermediateValues_* intermediate_values = static_cast<IntermediateValues_*>(this->intermediate_values_);
		LinearAlgebra::BaseMatrix<T> out(intermediate_values->states_.size() - 1, this->hidden_size_);
		for (std::uint64_t i = 0; i < intermediate_values->states_.size() - 1; i++)
		{
			out.row(i) = intermediate_values->states_[i]->n_hidden_state;
		}
		return out;
	}
} // namespace MyNN::RNN
