#pragma once
#include "HeaderBaseRNN.h"
#include "FunctionsActivate.hpp"

#include <vector>

namespace MyNN {
	namespace RNN {
		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class LSTM : public BaseRNN<T, Enable> {
		protected:

		public:
			LSTM();
			~LSTM() override;
		public:
			template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
			class DefaultComputeBlockOneH : public ComputeBlockRNN<T, Enable> {
			protected:
				struct ValuesForCompute : IComputeBlockRNN<T, Enable>::ValuesForCompute {

					LinearAlgebra::BaseMatrix<T, Enable> U; //[H x 4H]
					LinearAlgebra::BaseMatrix<T, Enable> W; //[I x 4H]
					LinearAlgebra::BaseRowVector<T, Enable> B;  //[1 x 4H]

					//LinearAlgebra::BaseMatrix<T, Enable> W_Out;
					//LinearAlgebra::BaseRowVector<T, Enable> B_Out;
				};
				struct NState : ComputeBlockRNN<T, Enable>::NState {
					//input_state_n - [1 x I]
					LinearAlgebra::BaseRowVector<T, Enable> n_cell_state; // [1 x H]
					LinearAlgebra::BaseRowVector<T, Enable> tmp_f, tmp_i, tmp_c_bar, tmp_o; // [1 x H]
					LinearAlgebra::BaseRowVector<T, Enable> tmp_Z; // [1 x 4H]
				};
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T, Enable>> in basecompblockrnn*/
				__forceinline void nStepCalculation(
					const typename ValuesForCompute* __restrict values_for_compute,
					typename NState* __restrict n_state,
					const LinearAlgebra::BaseRowVector<T, Enable>& x_n
				) //noexcept
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

				void allStepsCalculation() override {
					const std::uint64_t& H = this->hidden_size_;
					std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

					NState* n_state = static_cast<NState*>(this->n_state_.get());
					const ValuesForCompute* values_for_compute = static_cast<const ValuesForCompute*>(this->values_for_compute.get());

					const LinearAlgebra::BaseRowVector<T, Enable>& x_n = this->input_state_.row(number_n);

					n_state->n_cell_state = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->n_hidden_state = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);

					n_state->tmp_f = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_i = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_c_bar = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_o = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_Z = LinearAlgebra::BaseRowVector<T, Enable>::Zero(4 * H);

					for (std::uint64_t n = 0; n < number_steps; n++) {
						this->nStepCalculation(values_for_compute, n_state, n, x_n);
					}
				}
			public:

				LinearAlgebra::BaseMatrix<T, Enable> getOutput() override {
					return LinearAlgebra::BaseMatrix<T, Enable>(
						static_cast<NState*>(this->n_state_.get())->n_hidden_state
					);
				}
				__forceinline void compute() override {
					this->allStepsCalculation();
				}
			};
			template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
			class DefaultComputeBlockAllH : public DefaultComputeBlockOneH<T, Enable> {
			protected:
				LinearAlgebra::BaseMatrix<T, Enable> hidden_states_;
				//std::vector<DefaultComputeBlockAllH::DefaultComputeBlockOneH<T, Enable>::NState> 

				void allStepsCalculation() override {
					const std::uint64_t& H = this->hidden_size_;
					std::uint64_t number_steps = std::min(this->input_state_.rows(), this->max_steps_);

					NState* n_state = static_cast<NState*>(this->n_state_.get());
					const ValuesForCompute* values_for_compute = static_cast<const ValuesForCompute*>(this->values_for_compute.get());

					this->hidden_states_ = LinearAlgebra::BaseMatrix<T, Enable>::Zero(number_steps, H);

					n_state->n_cell_state = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->n_hidden_state = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);

					n_state->tmp_f = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_i = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_c_bar = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_o = LinearAlgebra::BaseRowVector<T, Enable>::Zero(H);
					n_state->tmp_Z = LinearAlgebra::BaseRowVector<T, Enable>::Zero(4 * H);

					for (std::uint64_t n = 0; n < number_steps; n++) {
						this->nStepCalculation(values_for_compute, n_state, n);
						this->hidden_states_.row(n) = n_state->n_hidden_state;
					}
				}
			public:
				LinearAlgebra::BaseMatrix<T, Enable> getOutput() override {
					return this->hidden_states_;
				}
			};

		};
	}
}