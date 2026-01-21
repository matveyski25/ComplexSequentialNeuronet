#pragma once
#include "BaseRNN.h"
#include <cassert>

#include "FunctionsActivate.hpp"

namespace MyNN::RNN::LSTM {
	using LinearAlgebra::BaseMatrix, std::uint32_t;

	template<typename T>
	struct DefaultContextComputeBlockOneH;

	template<typename T, typename Derived, typename FromType, typename ToType>
	struct DefaultContextLSTM : ContextBaseRNN<T, Derived, FromType, ToType> {};

	template<typename T>
	struct DefaultFeatureComputeBlockOneH :
	FeatureComputeBlockRNN<T, DefaultContextComputeBlockOneH<T>, DefaultFeatureComputeBlockOneH<T>>
	{
		void setInput(const BaseMatrix<T>&) override;
		LinearAlgebra::BaseMatrix<T> getOutput() override;
		void compute() override;

		void nStepsCalculationImpl(uint64_t);
		void allStepsCalculation() override;
	};


	template<typename T>
	struct DefaultContextComputeBlockOneH : ContextComputeBlockRNN<T>{
		LinearAlgebra::BaseMatrix<T> U; //[H x 4H]
		LinearAlgebra::BaseMatrix<T> W; //[I x 4H]
		LinearAlgebra::BaseRowVector<T> B;  //[1 x 4H]

		//LinearAlgebra::BaseMatrix<T> W_Out;
		//LinearAlgebra::BaseRowVector<T> B_Out;

		//input_state_n - [1 x I]
		LinearAlgebra::BaseRowVector<T> n_cell_state, n_hidden_state; // [1 x H]
		LinearAlgebra::BaseRowVector<T> tmp_f, tmp_i, tmp_c_bar, tmp_o; // [1 x H]
		LinearAlgebra::BaseRowVector<T> tmp_Z; // [1 x 4H]

		void setVoidNState(const std::uint64_t & hidden_size) {
			this->n_cell_state = LinearAlgebra::BaseRowVector<T>(hidden_size);
			this->n_hidden_state = LinearAlgebra::BaseRowVector<T>(hidden_size);

			this->tmp_f = LinearAlgebra::BaseRowVector<T>(hidden_size);
			this->tmp_i = LinearAlgebra::BaseRowVector<T>(hidden_size);
			this->tmp_c_bar = LinearAlgebra::BaseRowVector<T>(hidden_size);
			this->tmp_o = LinearAlgebra::BaseRowVector<T>(hidden_size);

			this->tmp_Z = LinearAlgebra::BaseRowVector<T>(4 * hidden_size);
		}
	};

	/*class DefaultSaver : public ComputeBlockRNN<T>::Saver {
			protected:
			void saveMatrix(const LinearAlgebra::BaseMatrix<T>& matx, std::ofstream& file) {file << 'm' << matx.rows() << ' ' << matx.cols() << ' ';for (std::uint64_t i = 0; i < matx.rows(); ++i) {for (std::uint64_t j = 0; j < matx.cols(); ++j) {file << matx(i, j) << ' ';
			}
			}file << "\n";
			}
			void saveVector(const LinearAlgebra::BaseVector<T>& vec, std::ofstream& file) {file << 'c' << vec.rows() << ' ';for (std::uint64_t i = 0; i < vec.rows(); ++i) {file << vec(i) << ' ';
			}file << "\n";
			}
			void saveVector(const LinearAlgebra::BaseRowVector<T>& vec, std::ofstream& file) {file << 'r' << vec.cols() << ' ';for (std::uint64_t i = 0; i < vec.cols(); ++i) {file << vec(i) << ' ';
			}file << "\n";
			}
			public:
			void save(IComputeBlockNN<T> * compute_block_) override {std::ofstream file(static_cast<Args>(this->args_.get()).path_and_file, std::ios::trunc); // Используйте trunc для перезаписиif (!file) throw std::runtime_error("Cannot open file for writing");DefaultComputeBlockOneH* compute_block = static_cast<DefaultComputeBlockOneH *>(compute_block_);DefaultComputeBlockOneH::ValuesForCompute* values_for_compute = static_cast<DefaultComputeBlockOneH::ValuesForCompute*>(compute_block->getValuesForCompute());this->saveMatrix(values_for_compute->W, file);this->saveMatrix(values_for_compute->U, file);this->saveVector(values_for_compute->B, file);file << compute_block->getInputSize() << ' ' << compute_block->getHiddenSize() << ' ' << compute_block->getOutputSize() << ' ' << compute_block->getMaxSteps() << '\n';
			}
			struct Args : public IComputeBlockNN<T>::ILoader::ArgsLoader
{
	std::string path_and_file;// /Абсолютный/относительный путь до файла включая его имя
};
			void setArgsForLoad(const typename IComputeBlockNN<T>::ILoader::ArgsLoader* args_) override {this->args->path_and_file = static_cast<const Args*>(args_)->path_and_file;
			}
		};
	class DefaultLoader : public ComputeBlockRNN<T>::Loader {
			protected:
			void loadMatrix(LinearAlgebra::BaseMatrix<T>& matx, std::ifstream& file) {char temp = ' ';file >> temp;if (temp != 'm') throw std::runtime_error("Attemp load matrix in not matrix");std::uint64_t rows = 0;std::uint64_t cols = 0;file >> rows >> cols;matx = LinearAlgebra::BaseMatrix<T>(rows, cols);for (std::uint64_t i = 0; i < rows; ++i) {for (std::uint64_t j = 0; j < cols; ++j) {file >> matx(i, j);
			}
			}
			}
			void loadVector(LinearAlgebra::BaseVector<T>& vec, std::ifstream& file) {char temp = ' ';file >> temp;if (temp != 'c') throw std::runtime_error("Attemp load vector in not vector");std::uint64_t rows = 0;file >> rows;vec = LinearAlgebra::BaseVector<T>(rows);for (std::uint64_t i = 0; i < rows; ++i) {file >> vec(i);
			}
			}
			void loadVector(LinearAlgebra::BaseRowVector<T>& vec, std::ifstream& file) {char temp = ' ';file >> temp;if (temp != 'r') throw std::runtime_error("Attemp load row-vector in not row-vector");std::uint64_t cols = 0;file >> cols;vec = LinearAlgebra::BaseRowVector<T>(cols);for (std::uint64_t i = 0; i < cols; ++i) {file >> vec(i);
			}
			}
			public:
			void load(IComputeBlockNN<T>* compute_block_) override {std::ifstream file(static_cast<Args>(this->args_.get()).path_and_file);if (!file) throw std::runtime_error("Cannot open file for reading");DefaultComputeBlockOneH* compute_block = static_cast<DefaultComputeBlockOneH*>(compute_block_);DefaultComputeBlockOneH::ValuesForCompute* values_for_compute = static_cast<DefaultComputeBlockOneH::ValuesForCompute*>(compute_block->getValuesForCompute());this->saveMatrix(values_for_compute->W, file);this->saveMatrix(values_for_compute->U, file);this->saveVector(values_for_compute->B, file);file >> compute_block->input_size_ >> compute_block->hidden_size_ >> compute_block->output_size_ >> compute_block->max_steps_;
			}
			struct Args : public IComputeBlockNN<T>::ILoader::ArgsLoader
{
	std::string path_and_file;// /Абсолютный/относительный путь до файла включая его имя
};
			void setArgsForLoad(const typename IComputeBlockNN<T>::ILoader::ArgsLoader* args_) override {this->args->path_and_file = static_cast<const Args*>(args_)->path_and_file;
			}
		};
	class DefaultOptimizer {
		};
	class DefaultRandomizer
				{

				};

	class DefaultComputeBlockAllH : public DefaultComputeBlockOneH {
			protected:
				LinearAlgebra::BaseMatrix<T> hidden_states_;

				void allStepsCalculation() override;
			public:
				LinearAlgebra::BaseMatrix<T> getOutput() override;
			};

	template<typename T>
	class TrainableLSTM : public virtual LSTM<T>, public virtual BaseTrainableRNN<T>{
		public:
			TrainableLSTM();
			~TrainableLSTM() = default;
			class DefaultComputeBlockOneH : public LSTM<T>::DefaultComputeBlockOneH, public ITrainableComputeBlockRNN<T> {
			protected:
				struct IntermediateValues : ITrainableComputeBlockRNN<T>::IntermediateValues{
					std::vector<typename LSTM<T>::DefaultComputeBlockAllH::DefaultComputeBlockOneH::NState> states_;
				};
				void allStepsCalculation() override;
			};
			class DefaultComputeBlockAllH : public DefaultComputeBlockOneH {
			protected:
				LinearAlgebra::BaseMatrix<T> getOutput() override;
			};
		};*/
}
