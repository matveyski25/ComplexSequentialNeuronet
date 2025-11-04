#pragma once
#include "HeaderBaseRNN.h"
#include "FunctionsActivate.hpp"
#include <fstream>
#include <vector>

namespace MyNN {
	namespace RNN {
		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class LSTM : public BaseRNN<T, Enable> {
		public:
			LSTM();
			~LSTM() override = default;
		public:
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
					LinearAlgebra::BaseRowVector<T, Enable> n_cell_state, n_hidden_state; // [1 x H]
					LinearAlgebra::BaseRowVector<T, Enable> tmp_f, tmp_i, tmp_c_bar, tmp_o; // [1 x H]
					LinearAlgebra::BaseRowVector<T, Enable> tmp_Z; // [1 x 4H]

					void setZero(const std::uint64_t & hidden_size) {
						this->n_cell_state = LinearAlgebra::BaseRowVector<T, Enable>::Zero(hidden_size);
						this->n_hidden_state = LinearAlgebra::BaseRowVector<T, Enable>::Zero(hidden_size);

						this->tmp_f = LinearAlgebra::BaseRowVector<T, Enable>::Zero(hidden_size);
						this->tmp_i = LinearAlgebra::BaseRowVector<T, Enable>::Zero(hidden_size);
						this->tmp_c_bar = LinearAlgebra::BaseRowVector<T, Enable>::Zero(hidden_size);
						this->tmp_o = LinearAlgebra::BaseRowVector<T, Enable>::Zero(hidden_size);

						this->tmp_Z = LinearAlgebra::BaseRowVector<T, Enable>::Zero(4 * hidden_size);
					}
				};
				/*this function return cell_state, hidden_state saving in vector<LinearAlgebra::BaseRowVector<T, Enable>> in basecompblockrnn*/
				 void nStepCalculation(
					const typename ValuesForCompute* __restrict values_for_compute,
					typename NState* __restrict n_state,
					const LinearAlgebra::BaseRowVector<T, Enable>& x_n
				); //noexcept
				
				void allStepsCalculation() override;
			public:
				LinearAlgebra::BaseMatrix<T, Enable> getOutput() override;

				void compute() override;

				const ValuesForCompute* getValuesForCompute() override;
				void setValuesForCompute(const ValuesForCompute*) override;

				class DefaultSaver : public ComputeBlockRNN<T, Enable>::Saver {
					void saveMatrix(const LinearAlgebra::BaseMatrix<T, Enable>& m, std::ofstream& file) {
						file << 'm' << m.rows() << ' ' << m.cols() << ' ';
						for (std::uint64_t i = 0; i < m.rows(); ++i) {
							for (std::uint64_t j = 0; j < m.cols(); ++j) {
								file << m(i, j) << ' ';
							}
						}
						file << "\n";
					}
					void saveVector(const LinearAlgebra::BaseVector<T, Enable>& v, std::ofstream& file) {
						file << 'c' << v.rows() << ' ';
						for (std::uint64_t i = 0; i < v.rows(); ++i) {
							file << v(i) << ' ';
						}
						file << "\n";
					}
					void saveVector(const LinearAlgebra::BaseRowVector<T, Enable>& v, std::ofstream& file) {
						file << 'r' << v.cols() << ' ';
						for (std::uint64_t i = 0; i < v.cols(); ++i) {
							file << v(i) << ' ';
						}
						file << "\n";
					}
				public:
					void save(IComputeBlockNN<T, Enable> * compute_block_) override {
						std::ofstream file(filename, std::ios::trunc); // Используйте trunc для перезаписи
						if (!file) throw std::runtime_error("Cannot open file for writing");

						DefaultComputeBlockOneH* compute_block = static_cast<DefaultComputeBlockOneH *>(compute_block_);
						DefaultComputeBlockOneH::ValuesForCompute* values_for_compute = static_cast<DefaultComputeBlockOneH::ValuesForCompute*>(compute_block->getValuesForCompute());
						this->saveMatrix(values_for_compute->W, file);
						this->saveMatrix(values_for_compute->U, file);
						this->saveVector(values_for_compute->B, file);

						file << compute_block->getInputSize() << ' ' << compute_block->getHiddenSize() << ' ' << compute_block->getOutputSize() << ' ' << compute_block->getMaxSteps() << '\n';
					}
				};
				class DefaultLoader : public ComputeBlockRNN<T, Enable>::Loader {
					void loadMatrix(const LinearAlgebra::BaseMatrix<T, Enable>& m, std::ofstream& file) {
						char temp;
						file >> temp;
						if (temp != 'm') throw std::runtime_error("Attemp load matrix in not matrix");
						file << 'm' << m.rows() << ' ' << m.cols() << ' ';
						for (std::uint64_t i = 0; i < m.rows(); ++i) {
							for (std::uint64_t j = 0; j < m.cols(); ++j) {
								file << m(i, j) << ' ';
							}
						}
						file << "\n";
					}
					void loadVector(const LinearAlgebra::BaseVector<T, Enable>& v, std::ofstream& file) {
						file << 'c' << v.rows() << ' ';
						for (std::uint64_t i = 0; i < v.rows(); ++i) {
							file << v(i) << ' ';
						}
						file << "\n";
					}
					void loadVector(const LinearAlgebra::BaseRowVector<T, Enable>& v, std::ofstream& file) {
						file << 'r' << v.cols() << ' ';
						for (std::uint64_t i = 0; i < v.cols(); ++i) {
							file << v(i) << ' ';
						}
						file << "\n";
					}
				public:
					void load() override {
						std::string path_and_name;
					}
				};
			};

			class DefaultComputeBlockAllH : public DefaultComputeBlockOneH {
			protected:
				LinearAlgebra::BaseMatrix<T, Enable> hidden_states_;

				void allStepsCalculation() override;
			public:
				LinearAlgebra::BaseMatrix<T, Enable> getOutput() override;
			};
		};

		template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
		class TrainableLSTM : public virtual LSTM<T, Enable>, public virtual BaseTrainableRNN<T, Enable>{
		public:
			TrainableLSTM();
			~TrainableLSTM() override = default;
			class DefaultComputeBlockOneH : public LSTM<T, Enable>::DefaultComputeBlockOneH, public ITrainableComputeBlockRNN<T, Enable> {
			protected:
				struct IntermediateValues : ITrainableComputeBlockRNN<T, Enable>::IntermediateValues{
					std::vector<typename LSTM<T, Enable>::DefaultComputeBlockAllH::DefaultComputeBlockOneH::NState> states_;
				};
				void allStepsCalculation() override;
			};
			class DefaultComputeBlockAllH : public DefaultComputeBlockOneH {
			protected:
				LinearAlgebra::BaseMatrix<T, Enable> getOutput() override;
			};
		};
	}
}