#pragma once

#include "HeaderLSTM_and_BiLSTM.h"

#include <math.h>
#include <algorithm>
#include <random>
#include <chrono>

/*void save_vector(std::ofstream& file, const std::vector<MatrixXld>& vec) const {
	file << vec.size() << "\n";
	for (const auto& m : vec) {
		save_matrix(file, m);
	}
}

void load_vector(std::ifstream& file, std::vector<MatrixXld>& vec) {
	size_t size;
	file >> size;
	vec.resize(size);
	for (auto& m : vec) {
		load_matrix(file, m);
	}
	this->dW_out.conservativeResize(other.dW_out.rows(), other.dW_out.cols());
			this->dB_out.conservativeResize(other.dB_out.rows(), other.dB_out.cols());

			this->dW_gamma_layernorm.conservativeResize(other.dW_gamma_layernorm.rows(), other.dW_gamma_layernorm.cols());
			this->dB_beta_layernorm.conservativeResize(other.dB_beta_layernorm.rows(), other.dB_beta_layernorm.cols());

			this->dV_a_attention.conservativeResize(other.dV_a_attention.rows(), other.dV_a_attention.cols());
			this->dW_e_attention.conservativeResize(other.dW_e_attention.rows(), other.dW_e_attention.cols());
			this->dW_d_attention.conservativeResize(other.dW_d_attention.rows(), other.dW_d_attention.cols());

			this->dW_f_dec.conservativeResize(other.dW_f_dec.rows(), other.dW_f_dec.cols());
			this->dU_f_dec.conservativeResize(other.dU_f_dec.rows(), other.dU_f_dec.cols());
			this->dB_f_dec.conservativeResize(other.dB_f_dec.rows(), other.dB_f_dec.cols());
			this->dW_i_dec.conservativeResize(other.dW_i_dec.rows(), other.dW_i_dec.cols());
			this->dU_i_dec.conservativeResize(other.dU_i_dec.rows(), other.dU_i_dec.cols());
			this->dB_i_dec.conservativeResize(other.dB_i_dec.rows(), other.dB_i_dec.cols());
			this->dW_ccond_dec.conservativeResize(other.dW_ccond_dec.rows(), other.dW_ccond_dec.cols());
			this->dU_ccond_dec.conservativeResize(other.dU_ccond_dec.rows(), other.dU_ccond_dec.cols());
			this->dB_ccond_dec.conservativeResize(other.dB_ccond_dec.rows(), other.dB_ccond_dec.cols());
			this->dW_o_dec.conservativeResize(other.dW_o_dec.rows(), other.dW_o_dec.cols());
			this->dU_o_dec.conservativeResize(other.dU_o_dec.rows(), other.dU_o_dec.cols());
			this->dB_o_dec.conservativeResize(other.dB_o_dec.rows(), other.dB_o_dec.cols());

			this->dW_f_forw_enc.conservativeResize(other.dW_f_forw_enc.rows(), other.dW_f_forw_enc.cols());
			this->dU_f_forw_enc.conservativeResize(other.dU_f_forw_enc.rows(), other.dU_f_forw_enc.cols());
			this->dB_f_forw_enc.conservativeResize(other.dB_f_forw_enc.rows(), other.dB_f_forw_enc.cols());
			this->dW_i_forw_enc.conservativeResize(other.dW_i_forw_enc.rows(), other.dW_i_forw_enc.cols());
			this->dU_i_forw_enc.conservativeResize(other.dU_i_forw_enc.rows(), other.dU_i_forw_enc.cols());
			this->dB_i_forw_enc.conservativeResize(other.dB_i_forw_enc.rows(), other.dB_i_forw_enc.cols());
			this->dW_ccond_forw_enc.conservativeResize(other.dW_ccond_forw_enc.rows(), other.dW_ccond_forw_enc.cols());
			this->dU_ccond_forw_enc.conservativeResize(other.dU_ccond_forw_enc.rows(), other.dU_ccond_forw_enc.cols());
			this->dB_ccond_forw_enc.conservativeResize(other.dB_ccond_forw_enc.rows(), other.dB_ccond_forw_enc.cols());
			this->dW_o_forw_enc.conservativeResize(other.dW_o_forw_enc.rows(), other.dW_o_forw_enc.cols());
			this->dU_o_forw_enc.conservativeResize(other.dU_o_forw_enc.rows(), other.dU_o_forw_enc.cols());
			this->dB_o_forw_enc.conservativeResize(other.dB_o_forw_enc.rows(), other.dB_o_forw_enc.cols());

			this->dW_f_back_enc.conservativeResize(other.dW_f_back_enc.rows(), other.dW_f_back_enc.cols());
			this->dU_f_back_enc.conservativeResize(other.dU_f_back_enc.rows(), other.dU_f_back_enc.cols());
			this->dB_f_back_enc.conservativeResize(other.dB_f_back_enc.rows(), other.dB_f_back_enc.cols());
			this->dW_i_back_enc.conservativeResize(other.dW_i_back_enc.rows(), other.dW_i_back_enc.cols());
			this->dU_i_back_enc.conservativeResize(other.dU_i_back_enc.rows(), other.dU_i_back_enc.cols());
			this->dB_i_back_enc.conservativeResize(other.dB_i_back_enc.rows(), other.dB_i_back_enc.cols());
			this->dW_ccond_back_enc.conservativeResize(other.dW_ccond_back_enc.rows(), other.dW_ccond_back_enc.cols());
			this->dU_ccond_back_enc.conservativeResize(other.dU_ccond_back_enc.rows(), other.dU_ccond_back_enc.cols());
			this->dB_ccond_back_enc.conservativeResize(other.dB_ccond_back_enc.rows(), other.dB_ccond_back_enc.cols());
			this->dW_o_back_enc.conservativeResize(other.dW_o_back_enc.rows(), other.dW_o_back_enc.cols());
			this->dU_o_back_enc.conservativeResize(other.dU_o_back_enc.rows(), other.dU_o_back_enc.cols());
			this->dB_o_back_enc.conservativeResize(other.dB_o_back_enc.rows(), other.dB_o_back_enc.cols());
}*/


class Seq2SeqWithAttention {
protected:
	class Encoder : public BiLSTM {
	public:
		using BiLSTM::BiLSTM;
		Encoder() : BiLSTM() {};
		void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
			this->SetInput_states(input_sequence_batch);
			this->All_state_Сalculation();
		}

		const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
			return this->Common_Hidden_states;
		}

	};
	class Decoder : public SimpleLSTM {
	public:
		void All_state_Сalculation() override {
			if (this->encoder_outputs.empty()) return;

			auto apply_layernorm = [this](const RowVectorXld& x) -> RowVectorXld {
				double epsilon = 1e-5L;
				double mean = x.mean();
				double variance = (x.array() - mean).square().mean();
				return ((x.array() - mean) / std::sqrt(variance + epsilon)).matrix().array() * layernorm_gamma.array()
					+ layernorm_beta.array();
				};

			auto l2_normalize = [](const RowVectorXld& x) -> RowVectorXld {
				double norm = std::sqrt(x.squaredNorm() + 1e-8L);
				return x / norm;
				};

			auto normalize_scale = [](RowVectorXld& vec) {
				double maxval = vec.cwiseAbs().maxCoeff();
				if (maxval > 0.0L) vec /= maxval;
				};

			// Очистка
			Output_state.clear();

			size_t batch_size = encoder_outputs.size();
			Output_state.resize(batch_size);

			MatrixXld W_x(Input_size, 4 * Hidden_size);
			W_x << W_F, W_I, W_C, W_O;

			MatrixXld W_h(Hidden_size, 4 * Hidden_size);
			W_h << U_F, U_I, U_C, U_O;

			RowVectorXld b(4 * Hidden_size);
			b << B_F, B_I, B_C, B_O;

			for (size_t n = 0; n < batch_size; ++n) {
				const auto& enc_out = encoder_outputs[n];
				std::vector<RowVectorXld> y_sequence;

				RowVectorXld y_prev = start_token;
				RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
				RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

				for (size_t t = 0; t < max_steps; ++t) {
					// Новый вызов ComputeContext
					BahdanauAttention::AttnOutput ao = attention_->ComputeContext(enc_out, h_prev);
					RowVectorXld& context = ao.context;

					RowVectorXld Decoderinput(Input_size);
					Decoderinput << y_prev, context;
					Decoderinput = l2_normalize(Decoderinput);

					RowVectorXld Z = Decoderinput * W_x + h_prev * W_h + b;
					RowVectorXld f_t = ActivationFunctions::Sigmoid(Z.leftCols(Hidden_size));
					RowVectorXld i_t = ActivationFunctions::Sigmoid(Z.middleCols(Hidden_size, Hidden_size));
					RowVectorXld c_bar = ActivationFunctions::Tanh(Z.middleCols(2 * Hidden_size, Hidden_size));
					RowVectorXld o_t = ActivationFunctions::Sigmoid(Z.rightCols(Hidden_size));

					RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_bar.array();
					RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();

					RowVectorXld proj_input_(Hidden_size + context.size());
					proj_input_ << h_t, context;
					auto proj_input = apply_layernorm(proj_input_);

					RowVectorXld y_t = proj_input * W_output.transpose() + B_output;
					y_sequence.push_back(y_t);

					if (IsEndToken(y_t)) {
						size_t end_len = static_cast<size_t>(end_token.rows());
						if (y_sequence.size() >= end_len - 1) {
							y_sequence.resize(y_sequence.size() - (end_len - 1));
							//context_sequence.resize(context_sequence.size() - (end_len - 1));
							//u_sequence.resize(u_sequence.size() - (end_len - 1));
						}
						break;
					}

					y_prev = y_t;
					h_prev = h_t;
					c_prev = c_t;
				}

				Eigen::Index T_out = static_cast<Eigen::Index>(y_sequence.size());
				Eigen::Index D = static_cast<Eigen::Index>(y_sequence[0].cols());
				Output_state[n] = MatrixXld(T_out, D);
				for (Eigen::Index t = 0; t < T_out; ++t) {
					Output_state[n].row(t) = y_sequence[t];
				}
			}
		}

		Decoder(std::unique_ptr<BahdanauAttention> attention_module,
			Eigen::Index hidden_size_encoder, Eigen::Index Hidden_size_, Eigen::Index embedding_dim_,
			RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_)
			: SimpleLSTM(embedding_dim_ + 2 * hidden_size_encoder/*= H_emb + 2H_enc*/, Hidden_size_), attention_(std::move(attention_module))
		{
			this->output_size = embedding_dim_;
			//размер контекста = 2 * Hidden_size_encoder = Number_states - embedding_dim
			size_t context_size = 2 * hidden_size_encoder;
			W_output = ActivationFunctions::matrix_random(output_size, Hidden_size_ + context_size, 0.0, 1.0);
			B_output = RowVectorXld::Zero(output_size);

			this->layernorm_gamma = RowVectorXld::Ones(Hidden_size_ + context_size);
			this->layernorm_beta = RowVectorXld::Zero(Hidden_size_ + context_size);
			// теперь SimpleLSTM::Input_size = Number_states, Hidden_size = Hidden_size_

			this->start_token = start_token_;   // эмбеддинг стартового токена (1 символ)
			this->end_token = end_token_;     // матрица эмбеддингов финишного токена (несколько символов)
			this->max_steps = max_steps_;    // ограничение на число шагов генерации
		}
		Decoder() = default;
		void SetEncoderOutputs(const std::vector<MatrixXld>& encoder_outputs) {
			this->encoder_outputs = encoder_outputs;
		}

		void Decode(const std::vector<MatrixXld>& encoder_outputs) {
			this->SetEncoderOutputs(encoder_outputs);
			this->All_state_Сalculation();
		}

		const std::vector<MatrixXld> & GetOutputStates() const { return Output_state; }
	protected:
		bool IsEndToken(const RowVectorXld& vec) const {
			for (int i = 0; i < end_token.rows(); ++i) {
				if ((vec - end_token.row(i)).norm() < 1e-6L) return true;
			}
			return false;
		}

		RowVectorXld start_token;   // эмбеддинг стартового токена (1 символ)
		MatrixXld end_token;     // матрица эмбеддингов финишного токена (несколько символов)
		size_t max_steps;    // ограничение на число шагов генерации

		std::unique_ptr<BahdanauAttention> attention_;

		std::vector<MatrixXld> encoder_outputs;
		//std::vector<MatrixXld> context_vectors;

		//std::vector<MatrixXld> U_state;

		std::vector<MatrixXld> Output_state;
		// --- Обновляемый выходной слой ---
		MatrixXld W_output;      // [output_size x (hidden_size + context_size)]
		RowVectorXld B_output;   // [1 x output_size]

		size_t output_size;
		//size_t embedding_dim;

		RowVectorXld layernorm_gamma; // [1 x Input_size]
		RowVectorXld layernorm_beta;  // [1 x Input_size]

	};
public:
	Seq2SeqWithAttention(
		std::unique_ptr<Encoder> encoder = std::make_unique<Encoder>(),
		std::unique_ptr<Decoder> decoder = std::make_unique<Decoder>());

	Seq2SeqWithAttention(
		Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
		Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
		std::unique_ptr<BahdanauAttention> attention_ = std::make_unique<BahdanauAttention>());

	//Seq2SeqWithAttention() = default;

	void SetInput_states(const std::vector<MatrixXld>& _inputs);

	void Inference();

	void Inference(const std::vector<MatrixXld>& input_sequence_batch);

	const std::vector<MatrixXld> & GetOutputs() const;

	void Save(std::string packname) {
		std::filesystem::create_directories(packname);
		encoder_->Save(packname + "/" + "Encoder");
		decoder_->save(packname + "/" + "Decoder");
	}

	void Load(std::string packname) {
		encoder_->Load(packname + "/" + "Encoder");
		decoder_->load(packname + "/" + "Decoder");
	}
protected:
	std::vector<MatrixXld> Input_States;
private:
	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
	
};

class Seq2SeqWithAttention_ForTrain : public Seq2SeqWithAttention {
protected:
	class Encoder : public BiLSTM_ForTrain {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		using BiLSTM_ForTrain::BiLSTM_ForTrain;
		Encoder() : BiLSTM_ForTrain() {};
		void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
			this->SetInput_states(input_sequence_batch);
			this->Batch_All_state_Сalculation();
		}

		const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
			return this->Common_Hidden_states;
		}
		
	};
	class Decoder : public Seq2SeqWithAttention::Decoder {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		using Seq2SeqWithAttention::Decoder::Decoder;
		Decoder() : Seq2SeqWithAttention::Decoder() {};

		void All_state_Сalculation() override {
			if (this->encoder_outputs.empty()) return;

			auto apply_layernorm = [this](const RowVectorXld& x, size_t n, size_t t) -> RowVectorXld {
				double epsilon = 1e-5L;
				double mean = x.mean();
				double variance = (x.array() - mean).square().mean();
				RowVectorXld x_lnorm = ((x.array() - mean) / std::sqrt(variance + epsilon));
				this->StatesForgrads.p_[n].row(t) = x_lnorm;
				return  x_lnorm.array() * layernorm_gamma.array()
					+ layernorm_beta.array();
				};

			auto l2_normalize = [](const RowVectorXld& x) -> RowVectorXld {
				double norm = std::sqrt(x.squaredNorm() + 1e-8L);
				return x / norm;
				};

			auto normalize_scale = [](RowVectorXld& vec) {
				double maxval = vec.cwiseAbs().maxCoeff();
				if (maxval > 0.0L) vec /= maxval;
				};

			// Очистка
			Output_state.clear();
			this->StatesForgrads.context.clear();
			this->StatesForgrads.x.clear();
			//this->StatesForgrads.p.clear();
			this->StatesForgrads.p_.clear();
			this->StatesForgrads.p__.clear();
			//this->StatesForgrads.z.clear();
			this->StatesForgrads.f.clear();
			this->StatesForgrads.i.clear();
			this->StatesForgrads.o.clear();
			this->StatesForgrads.ccond.clear();
			this->StatesForgrads.c.clear();
			this->StatesForgrads.h.clear();
			this->StatesForgrads.all_u.clear();
			this->StatesForgrads.all_alpha.clear();

			size_t batch_size = encoder_outputs.size();
			Output_state.resize(batch_size);
			this->StatesForgrads.context.resize(batch_size);
			this->StatesForgrads.x.resize(batch_size);
			//this->StatesForgrads.p.resize(batch_size);
			this->StatesForgrads.p_.resize(batch_size);
			this->StatesForgrads.p__.resize(batch_size);
			//this->StatesForgrads.z.resize(batch_size);
			this->StatesForgrads.f.resize(batch_size);
			this->StatesForgrads.i.resize(batch_size);
			this->StatesForgrads.o.resize(batch_size);
			this->StatesForgrads.ccond.resize(batch_size);
			this->StatesForgrads.c.resize(batch_size);
			this->StatesForgrads.h.resize(batch_size);
			this->StatesForgrads.all_u.resize(batch_size);
			this->StatesForgrads.all_alpha.resize(batch_size);

			// Резерв под timesteps (необязательно, но желательно)
			for (size_t n = 0; n < batch_size; ++n) {
				this->StatesForgrads.all_u[n].reserve(max_steps);
				this->StatesForgrads.all_alpha[n].reserve(max_steps);
			}

			MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
			W_x << W_F, W_I, W_C, W_O;

			MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
			W_h << U_F, U_I, U_C, U_O;

			RowVectorXld b(4 * this->Hidden_size);
			b << B_F, B_I, B_C, B_O;

			for (size_t n = 0; n < batch_size; ++n) {
				const auto& enc_out = encoder_outputs[n];
				std::vector<RowVectorXld> y_sequence;

				RowVectorXld y_prev = start_token;
				RowVectorXld h_prev = RowVectorXld::Zero(this->Hidden_size);
				RowVectorXld c_prev = RowVectorXld::Zero(this->Hidden_size);

				this->StatesForgrads.f[n] = MatrixXld::Zero(max_steps, this->Hidden_size);
				this->StatesForgrads.i[n] = MatrixXld::Zero(max_steps, this->Hidden_size);
				this->StatesForgrads.ccond[n] = MatrixXld::Zero(max_steps, this->Hidden_size);
				this->StatesForgrads.o[n] = MatrixXld::Zero(max_steps, this->Hidden_size);
				this->StatesForgrads.c[n] = MatrixXld::Zero(max_steps, this->Hidden_size);
				this->StatesForgrads.h[n] = MatrixXld::Zero(max_steps, this->Hidden_size);

				this->StatesForgrads.context[n] = MatrixXld::Zero(max_steps, this->attention_->duo_encoder_hidden_size_);
				this->StatesForgrads.x[n] = MatrixXld::Zero(max_steps, this->Input_size);
				//this->StatesForgrads.p[n] = MatrixXld::Zero(max_steps, this->Hidden_size + this->attention_->duo_encoder_hidden_size_);
				this->StatesForgrads.p_[n] = MatrixXld::Zero(max_steps, this->Hidden_size + this->attention_->duo_encoder_hidden_size_);
				this->StatesForgrads.p__[n] = MatrixXld::Zero(max_steps, this->Hidden_size + this->attention_->duo_encoder_hidden_size_);
				//this->StatesForgrads.z[n] = MatrixXld::Zero(max_steps, 4 * this->Hidden_size);

				for (size_t t = 0; t < max_steps; ++t) {
					// Новый вызов ComputeContext
					BahdanauAttention::AttnOutput ao = attention_->ComputeContext(enc_out, h_prev);
					RowVectorXld& context = ao.context;
					std::vector<RowVectorXld>& u_t = ao.u_t;
					VectorXld& alpha = ao.alpha;

					this->StatesForgrads.all_u[n].push_back(std::move(u_t));
					this->StatesForgrads.all_alpha[n].push_back(std::move(alpha));

					RowVectorXld Decoderinput(this->Input_size);
					Decoderinput << y_prev, context;
					Decoderinput = l2_normalize(Decoderinput);

					RowVectorXld Z = Decoderinput * W_x + h_prev * W_h + b;
					RowVectorXld f_t = ActivationFunctions::Sigmoid(Z.leftCols(this->Hidden_size));
					RowVectorXld i_t = ActivationFunctions::Sigmoid(Z.middleCols(this->Hidden_size, this->Hidden_size));
					RowVectorXld c_bar = ActivationFunctions::Tanh(Z.middleCols(2 * this->Hidden_size, this->Hidden_size));
					RowVectorXld o_t = ActivationFunctions::Sigmoid(Z.rightCols(this->Hidden_size));

					RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_bar.array();
					RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();

					RowVectorXld proj_input_(this->Hidden_size + context.size());
					proj_input_ << h_t, context;
					auto proj_input = apply_layernorm(proj_input_, n, t);


					RowVectorXld y_t = proj_input * W_output.transpose() + B_output;
					y_sequence.push_back(y_t);


					this->StatesForgrads.f[n].row(t) = f_t;
					this->StatesForgrads.i[n].row(t) = i_t;
					this->StatesForgrads.ccond[n].row(t) = c_bar;
					this->StatesForgrads.o[n].row(t) = o_t;
					this->StatesForgrads.c[n].row(t) = c_t;
					this->StatesForgrads.h[n].row(t) = h_t;

					this->StatesForgrads.context[n].row(t) = context;
					this->StatesForgrads.x[n].row(t) = Decoderinput;
					//this->StatesForgrads.p[n].row(t) = proj_input_;
					this->StatesForgrads.p__[n].row(t) = proj_input;
					//this->StatesForgrads.z[n].row(t) = Z;

					if (IsEndToken(y_t)) break;

					y_prev = y_t;
					h_prev = h_t;
					c_prev = c_t;
				}

				Eigen::Index T_out = static_cast<Eigen::Index>(y_sequence.size());
				Eigen::Index D = static_cast<Eigen::Index>(y_sequence[0].cols());
				Output_state[n] = MatrixXld(T_out, D);
				for (Eigen::Index t = 0; t < T_out; ++t) {
					Output_state[n].row(t) = y_sequence[t];
				}
			}
		}


	protected:
		struct states_forgrads {
			std::vector<MatrixXld> f, i, o, ccond, c, h, context, /*z,*/ x, /*p,*/ p_, p__;
			std::vector<std::vector<std::vector<RowVectorXld>>>  all_u;    // batch × time_steps × A
			std::vector<std::vector<VectorXld>> all_alpha; // batch × time_steps
		};
		states_forgrads StatesForgrads;
	};

	struct grads_Seq2SeqWithAttention {
		MatrixXld dW_out; RowVectorXld dB_out;

		RowVectorXld dW_gamma_layernorm; RowVectorXld dB_beta_layernorm;

		MatrixXld dW_e_attention, dW_d_attention; VectorXld dV_a_attention;

		MatrixXld dW_f_dec, dU_f_dec; RowVectorXld dB_f_dec;
		MatrixXld dW_i_dec, dU_i_dec; RowVectorXld dB_i_dec;
		MatrixXld dW_ccond_dec, dU_ccond_dec; RowVectorXld dB_ccond_dec;
		MatrixXld dW_o_dec, dU_o_dec; RowVectorXld dB_o_dec;

		MatrixXld dW_f_forw_enc, dU_f_forw_enc; RowVectorXld dB_f_forw_enc;
		MatrixXld dW_i_forw_enc, dU_i_forw_enc; RowVectorXld dB_i_forw_enc;
		MatrixXld dW_ccond_forw_enc, dU_ccond_forw_enc; RowVectorXld dB_ccond_forw_enc;
		MatrixXld dW_o_forw_enc, dU_o_forw_enc; RowVectorXld dB_o_forw_enc;

		MatrixXld dW_f_back_enc, dU_f_back_enc; RowVectorXld dB_f_back_enc;
		MatrixXld dW_i_back_enc, dU_i_back_enc; RowVectorXld dB_i_back_enc;
		MatrixXld dW_ccond_back_enc, dU_ccond_back_enc; RowVectorXld dB_ccond_back_enc;
		MatrixXld dW_o_back_enc, dU_o_back_enc; RowVectorXld dB_o_back_enc;

		/*void operator +=(const grads_Seq2SeqWithAttention& other) {
			this->dW_out += other.dW_out;
			this->dB_out += other.dB_out;

			this->dW_gamma_layernorm += other.dW_gamma_layernorm;
			this->dB_beta_layernorm += other.dB_beta_layernorm;

			this->dV_a_attention += other.dV_a_attention;
			this->dW_e_attention += other.dW_e_attention;
			this->dW_d_attention += other.dW_d_attention;

			this->dW_f_dec += other.dW_f_dec; this->dU_f_dec += other.dU_f_dec; this->dB_f_dec += other.dB_f_dec;
			this->dW_i_dec += other.dW_i_dec; this->dU_i_dec += other.dU_i_dec; this->dB_i_dec += other.dB_i_dec;
			this->dW_ccond_dec += other.dW_ccond_dec; this->dU_ccond_dec += other.dU_ccond_dec; this->dB_ccond_dec += other.dB_ccond_dec;
			this->dW_o_dec += other.dW_o_dec; this->dU_o_dec += other.dU_o_dec; this->dB_o_dec += other.dB_o_dec;

			this->dW_f_forw_enc += other.dW_f_forw_enc; this->dU_f_forw_enc += other.dU_f_forw_enc; this->dB_f_forw_enc += other.dB_f_forw_enc;
			this->dW_i_forw_enc += other.dW_i_forw_enc; this->dU_i_forw_enc += other.dU_i_forw_enc; this->dB_i_forw_enc += other.dB_i_forw_enc;
			this->dW_ccond_forw_enc += other.dW_ccond_forw_enc; this->dU_ccond_forw_enc += other.dU_ccond_forw_enc; this->dB_ccond_forw_enc += other.dB_ccond_forw_enc;
			this->dW_o_forw_enc += other.dW_o_forw_enc; this->dU_o_forw_enc += other.dU_o_forw_enc; this->dB_o_forw_enc += other.dB_o_forw_enc;

			this->dW_f_back_enc += other.dW_f_back_enc; this->dU_f_back_enc += other.dU_f_back_enc; this->dB_f_back_enc += other.dB_f_back_enc;
			this->dW_i_back_enc += other.dW_i_back_enc; this->dU_i_back_enc += other.dU_i_back_enc; this->dB_i_back_enc += other.dB_i_back_enc;
			this->dW_ccond_back_enc += other.dW_ccond_back_enc; this->dU_ccond_back_enc += other.dU_ccond_back_enc; this->dB_ccond_back_enc += other.dB_ccond_back_enc;
			this->dW_o_back_enc += other.dW_o_back_enc; this->dU_o_back_enc += other.dU_o_back_enc; this->dB_o_back_enc += other.dB_o_back_enc;
		}
		void operator /=(const grads_Seq2SeqWithAttention& other) {
			this->dW_out.array() /= other.dW_out.array();
			this->dB_out.array() /= other.dB_out.array();

			this->dW_gamma_layernorm.array() /= other.dW_gamma_layernorm.array();
			this->dB_beta_layernorm.array() /= other.dB_beta_layernorm.array();

			this->dV_a_attention.array() /= other.dV_a_attention.array();
			this->dW_e_attention.array() /= other.dW_e_attention.array();
			this->dW_d_attention.array() /= other.dW_d_attention.array();

			this->dW_f_dec.array() /= other.dW_f_dec.array(); this->dU_f_dec.array() /= other.dU_f_dec.array(); this->dB_f_dec.array() /= other.dB_f_dec.array();
			this->dW_i_dec.array() /= other.dW_i_dec.array(); this->dU_i_dec.array() /= other.dU_i_dec.array(); this->dB_i_dec.array() /= other.dB_i_dec.array();
			this->dW_ccond_dec.array() /= other.dW_ccond_dec.array(); this->dU_ccond_dec.array() /= other.dU_ccond_dec.array(); this->dB_ccond_dec.array() /= other.dB_ccond_dec.array();
			this->dW_o_dec.array() /= other.dW_o_dec.array(); this->dU_o_dec.array() /= other.dU_o_dec.array(); this->dB_o_dec.array() /= other.dB_o_dec.array();

			this->dW_f_forw_enc.array() /= other.dW_f_forw_enc.array(); this->dU_f_forw_enc.array() /= other.dU_f_forw_enc.array(); this->dB_f_forw_enc.array() /= other.dB_f_forw_enc.array();
			this->dW_i_forw_enc.array() /= other.dW_i_forw_enc.array(); this->dU_i_forw_enc.array() /= other.dU_i_forw_enc.array(); this->dB_i_forw_enc.array() /= other.dB_i_forw_enc.array();
			this->dW_ccond_forw_enc.array() /= other.dW_ccond_forw_enc.array(); this->dU_ccond_forw_enc.array() /= other.dU_ccond_forw_enc.array(); this->dB_ccond_forw_enc.array() /= other.dB_ccond_forw_enc.array();
			this->dW_o_forw_enc.array() /= other.dW_o_forw_enc.array(); this->dU_o_forw_enc.array() /= other.dU_o_forw_enc.array(); this->dB_o_forw_enc.array() /= other.dB_o_forw_enc.array();

			this->dW_f_back_enc.array() /= other.dW_f_back_enc.array(); this->dU_f_back_enc.array() /= other.dU_f_back_enc.array(); this->dB_f_back_enc.array() /= other.dB_f_back_enc.array();
			this->dW_i_back_enc.array() /= other.dW_i_back_enc.array(); this->dU_i_back_enc.array() /= other.dU_i_back_enc.array(); this->dB_i_back_enc.array() /= other.dB_i_back_enc.array();
			this->dW_ccond_back_enc.array() /= other.dW_ccond_back_enc.array(); this->dU_ccond_back_enc.array() /= other.dU_ccond_back_enc.array(); this->dB_ccond_back_enc.array() /= other.dB_ccond_back_enc.array();
			this->dW_o_back_enc.array() /= other.dW_o_back_enc.array(); this->dU_o_back_enc.array() /= other.dU_o_back_enc.array(); this->dB_o_back_enc.array() /= other.dB_o_back_enc.array();
		}
		void operator /=(const double& val) {
			this->dW_out /= val;
			this->dB_out /= val;

			this->dW_gamma_layernorm /= val;
			this->dB_beta_layernorm /= val;

			this->dV_a_attention /= val;
			this->dW_e_attention /= val;
			this->dW_d_attention /= val;

			this->dW_f_dec /= val; this->dU_f_dec /= val; this->dB_f_dec /= val;
			this->dW_i_dec /= val; this->dU_i_dec /= val; this->dB_i_dec /= val;
			this->dW_ccond_dec /= val; this->dU_ccond_dec /= val; this->dB_ccond_dec /= val;
			this->dW_o_dec /= val; this->dU_o_dec /= val; this->dB_o_dec /= val;

			this->dW_f_forw_enc /= val; this->dU_f_forw_enc /= val; this->dB_f_forw_enc /= val;
			this->dW_i_forw_enc /= val; this->dU_i_forw_enc /= val; this->dB_i_forw_enc /= val;
			this->dW_ccond_forw_enc /= val; this->dU_ccond_forw_enc /= val; this->dB_ccond_forw_enc /= val;
			this->dW_o_forw_enc /= val; this->dU_o_forw_enc /= val; this->dB_o_forw_enc /= val;

			this->dW_f_back_enc /= val; this->dU_f_back_enc /= val; this->dB_f_back_enc /= val;
			this->dW_i_back_enc /= val; this->dU_i_back_enc /= val; this->dB_i_back_enc /= val;
			this->dW_ccond_back_enc /= val; this->dU_ccond_back_enc /= val; this->dB_ccond_back_enc /= val;
			this->dW_o_back_enc /= val; this->dU_o_back_enc /= val; this->dB_o_back_enc /= val;
		}*/

		static void check_nan(const std::string& name, const MatrixXld& mat) {
			if (!mat.allFinite()) {
				std::cerr << "[GRAD WARNING] NaN or Inf in: " << name << std::endl;
				throw std::invalid_argument(name);
			}
		}
		void operator +=(const grads_Seq2SeqWithAttention& other) {
#define CHECK_ADD(name)check_nan(#name " (before+=)", this->name); check_nan(#name " (other.name)", other.name);  this->name.noalias() += other.name; check_nan(#name " (after+=)", this->name)

			CHECK_ADD(dW_out); CHECK_ADD(dB_out);
			CHECK_ADD(dW_gamma_layernorm); CHECK_ADD(dB_beta_layernorm);
			CHECK_ADD(dV_a_attention); CHECK_ADD(dW_e_attention); CHECK_ADD(dW_d_attention);

			CHECK_ADD(dW_f_dec); CHECK_ADD(dU_f_dec); CHECK_ADD(dB_f_dec);
			CHECK_ADD(dW_i_dec); CHECK_ADD(dU_i_dec); CHECK_ADD(dB_i_dec);
			CHECK_ADD(dW_ccond_dec); CHECK_ADD(dU_ccond_dec); CHECK_ADD(dB_ccond_dec);
			CHECK_ADD(dW_o_dec); CHECK_ADD(dU_o_dec); CHECK_ADD(dB_o_dec);

			CHECK_ADD(dW_f_forw_enc); CHECK_ADD(dU_f_forw_enc); CHECK_ADD(dB_f_forw_enc);
			CHECK_ADD(dW_i_forw_enc); CHECK_ADD(dU_i_forw_enc); CHECK_ADD(dB_i_forw_enc);
			CHECK_ADD(dW_ccond_forw_enc); CHECK_ADD(dU_ccond_forw_enc); CHECK_ADD(dB_ccond_forw_enc);
			CHECK_ADD(dW_o_forw_enc); CHECK_ADD(dU_o_forw_enc); CHECK_ADD(dB_o_forw_enc);

			CHECK_ADD(dW_f_back_enc); CHECK_ADD(dU_f_back_enc); CHECK_ADD(dB_f_back_enc);
			CHECK_ADD(dW_i_back_enc); CHECK_ADD(dU_i_back_enc); CHECK_ADD(dB_i_back_enc);
			CHECK_ADD(dW_ccond_back_enc); CHECK_ADD(dU_ccond_back_enc); CHECK_ADD(dB_ccond_back_enc);
			CHECK_ADD(dW_o_back_enc); CHECK_ADD(dU_o_back_enc); CHECK_ADD(dB_o_back_enc);

#undef CHECK_ADD
		}
		void operator /=(const grads_Seq2SeqWithAttention& other) {
#define CHECK_DIV(name) check_nan(#name " (before/=)", this->name); this->name.array() /= other.name.array(); check_nan(#name " (after/=)", this->name)

			CHECK_DIV(dW_out); CHECK_DIV(dB_out);
			CHECK_DIV(dW_gamma_layernorm); CHECK_DIV(dB_beta_layernorm);
			CHECK_DIV(dV_a_attention); CHECK_DIV(dW_e_attention); CHECK_DIV(dW_d_attention);

			CHECK_DIV(dW_f_dec); CHECK_DIV(dU_f_dec); CHECK_DIV(dB_f_dec);
			CHECK_DIV(dW_i_dec); CHECK_DIV(dU_i_dec); CHECK_DIV(dB_i_dec);
			CHECK_DIV(dW_ccond_dec); CHECK_DIV(dU_ccond_dec); CHECK_DIV(dB_ccond_dec);
			CHECK_DIV(dW_o_dec); CHECK_DIV(dU_o_dec); CHECK_DIV(dB_o_dec);

			CHECK_DIV(dW_f_forw_enc); CHECK_DIV(dU_f_forw_enc); CHECK_DIV(dB_f_forw_enc);
			CHECK_DIV(dW_i_forw_enc); CHECK_DIV(dU_i_forw_enc); CHECK_DIV(dB_i_forw_enc);
			CHECK_DIV(dW_ccond_forw_enc); CHECK_DIV(dU_ccond_forw_enc); CHECK_DIV(dB_ccond_forw_enc);
			CHECK_DIV(dW_o_forw_enc); CHECK_DIV(dU_o_forw_enc); CHECK_DIV(dB_o_forw_enc);

			CHECK_DIV(dW_f_back_enc); CHECK_DIV(dU_f_back_enc); CHECK_DIV(dB_f_back_enc);
			CHECK_DIV(dW_i_back_enc); CHECK_DIV(dU_i_back_enc); CHECK_DIV(dB_i_back_enc);
			CHECK_DIV(dW_ccond_back_enc); CHECK_DIV(dU_ccond_back_enc); CHECK_DIV(dB_ccond_back_enc);
			CHECK_DIV(dW_o_back_enc); CHECK_DIV(dU_o_back_enc); CHECK_DIV(dB_o_back_enc);

#undef CHECK_DIV
		}
		void operator /=(const double& val) {
#define CHECK_DIV_SCALAR(name) check_nan(#name " (before/=ld)", this->name); this->name /= val; check_nan(#name " (after/=ld)", this->name)

			CHECK_DIV_SCALAR(dW_out); CHECK_DIV_SCALAR(dB_out);
			CHECK_DIV_SCALAR(dW_gamma_layernorm); CHECK_DIV_SCALAR(dB_beta_layernorm);
			CHECK_DIV_SCALAR(dV_a_attention); CHECK_DIV_SCALAR(dW_e_attention); CHECK_DIV_SCALAR(dW_d_attention);

			CHECK_DIV_SCALAR(dW_f_dec); CHECK_DIV_SCALAR(dU_f_dec); CHECK_DIV_SCALAR(dB_f_dec);
			CHECK_DIV_SCALAR(dW_i_dec); CHECK_DIV_SCALAR(dU_i_dec); CHECK_DIV_SCALAR(dB_i_dec);
			CHECK_DIV_SCALAR(dW_ccond_dec); CHECK_DIV_SCALAR(dU_ccond_dec); CHECK_DIV_SCALAR(dB_ccond_dec);
			CHECK_DIV_SCALAR(dW_o_dec); CHECK_DIV_SCALAR(dU_o_dec); CHECK_DIV_SCALAR(dB_o_dec);

			CHECK_DIV_SCALAR(dW_f_forw_enc); CHECK_DIV_SCALAR(dU_f_forw_enc); CHECK_DIV_SCALAR(dB_f_forw_enc);
			CHECK_DIV_SCALAR(dW_i_forw_enc); CHECK_DIV_SCALAR(dU_i_forw_enc); CHECK_DIV_SCALAR(dB_i_forw_enc);
			CHECK_DIV_SCALAR(dW_ccond_forw_enc); CHECK_DIV_SCALAR(dU_ccond_forw_enc); CHECK_DIV_SCALAR(dB_ccond_forw_enc);
			CHECK_DIV_SCALAR(dW_o_forw_enc); CHECK_DIV_SCALAR(dU_o_forw_enc); CHECK_DIV_SCALAR(dB_o_forw_enc);

			CHECK_DIV_SCALAR(dW_f_back_enc); CHECK_DIV_SCALAR(dU_f_back_enc); CHECK_DIV_SCALAR(dB_f_back_enc);
			CHECK_DIV_SCALAR(dW_i_back_enc); CHECK_DIV_SCALAR(dU_i_back_enc); CHECK_DIV_SCALAR(dB_i_back_enc);
			CHECK_DIV_SCALAR(dW_ccond_back_enc); CHECK_DIV_SCALAR(dU_ccond_back_enc); CHECK_DIV_SCALAR(dB_ccond_back_enc);
			CHECK_DIV_SCALAR(dW_o_back_enc); CHECK_DIV_SCALAR(dU_o_back_enc); CHECK_DIV_SCALAR(dB_o_back_enc);

#undef CHECK_DIV_SCALAR
		}

		void SetZero(const Seq2SeqWithAttention_ForTrain * seq2seq) {
			Eigen::Index E = seq2seq->decoder_->output_size;
			Eigen::Index H = seq2seq->decoder_->Hidden_size;
			Eigen::Index X = seq2seq->decoder_->Input_size;
			Eigen::Index C = X - E;
			Eigen::Index D = H + C;
			Eigen::Index A = seq2seq->decoder_->attention_->attention_size_;
			Eigen::Index HE = seq2seq->encoder_->Common_Hidden_size;
			Eigen::Index EE = seq2seq->encoder_->Common_Input_size;

			this->dW_out = MatrixXld::Zero(E, D), this->dB_out = RowVectorXld::Zero(E);

			this->dW_gamma_layernorm = RowVectorXld::Zero(D), this->dB_beta_layernorm = RowVectorXld::Zero(D);

			this->dV_a_attention = VectorXld::Zero(A), this->dW_e_attention = MatrixXld::Zero(A, C), this->dW_d_attention = MatrixXld::Zero(A, H);

			this->dW_f_dec = MatrixXld::Zero(X, H), this->dU_f_dec = MatrixXld::Zero(H, H), this->dB_f_dec = RowVectorXld::Zero(H),
			this->dW_i_dec = MatrixXld::Zero(X, H), this->dU_i_dec = MatrixXld::Zero(H, H), this->dB_i_dec = RowVectorXld::Zero(H),
			this->dW_ccond_dec = MatrixXld::Zero(X, H), this->dU_ccond_dec = MatrixXld::Zero(H, H), this->dB_ccond_dec = RowVectorXld::Zero(H),
			this->dW_o_dec = MatrixXld::Zero(X, H), this->dU_o_dec = MatrixXld::Zero(H, H), this->dB_o_dec = RowVectorXld::Zero(H);

			this->dW_f_forw_enc = MatrixXld::Zero(EE, HE), this->dU_f_forw_enc = MatrixXld::Zero(HE, HE), this->dB_f_forw_enc = RowVectorXld::Zero(HE),
			this->dW_i_forw_enc = MatrixXld::Zero(EE, HE), this->dU_i_forw_enc = MatrixXld::Zero(HE, HE), this->dB_i_forw_enc = RowVectorXld::Zero(HE),
			this->dW_ccond_forw_enc = MatrixXld::Zero(EE, HE), this->dU_ccond_forw_enc = MatrixXld::Zero(HE, HE), this->dB_ccond_forw_enc = RowVectorXld::Zero(HE),
			this->dW_o_forw_enc = MatrixXld::Zero(EE, HE), this->dU_o_forw_enc = MatrixXld::Zero(HE, HE), this->dB_o_forw_enc = RowVectorXld::Zero(HE);

			this->dW_f_back_enc = MatrixXld::Zero(EE, HE), this->dU_f_back_enc = MatrixXld::Zero(HE, HE), this->dB_f_back_enc = RowVectorXld::Zero(HE),
			this->dW_i_back_enc = MatrixXld::Zero(EE, HE), this->dU_i_back_enc = MatrixXld::Zero(HE, HE), this->dB_i_back_enc = RowVectorXld::Zero(HE),
			this->dW_ccond_back_enc = MatrixXld::Zero(EE, HE), this->dU_ccond_back_enc = MatrixXld::Zero(HE, HE), this->dB_ccond_back_enc = RowVectorXld::Zero(HE),
			this->dW_o_back_enc = MatrixXld::Zero(EE, HE), this->dU_o_back_enc = MatrixXld::Zero(HE, HE), this->dB_o_back_enc = RowVectorXld::Zero(HE);
			//std::cout << "[TRACE] SetZero() called" << std::endl;//at:  << this 
			//check_nan("dW_ccond_dec right after SetZero()", dW_ccond_dec );

		}
	};

	grads_Seq2SeqWithAttention Backward(size_t Number_InputState, MatrixXld Y_True);

	grads_Seq2SeqWithAttention BackwardWithLogging(size_t Number_InputState, MatrixXld Y_True);
public:
	Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train = std::make_unique<Encoder>(), std::unique_ptr<Decoder> decoder_train = std::make_unique<Decoder>());

	Seq2SeqWithAttention_ForTrain(
		Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
		Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
		std::unique_ptr<BahdanauAttention> attention_, size_t batch_size);

	Seq2SeqWithAttention_ForTrain(
		Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_,
		Eigen::Index Decoder_Hidden_size_, Eigen::Index Attention_size_,
		Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_, size_t batch_size) {
		std::unique_ptr<Seq2SeqWithAttention_ForTrain::Encoder> encoder__ = std::make_unique<Seq2SeqWithAttention_ForTrain::Encoder>(batch_size, Input_size_, Encoder_Hidden_size_);
		std::unique_ptr<Seq2SeqWithAttention_ForTrain::Decoder> decoder__ = std::make_unique<Seq2SeqWithAttention_ForTrain::Decoder>(std::make_unique<BahdanauAttention>(Encoder_Hidden_size_, Decoder_Hidden_size_, Attention_size_),
			Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_);
		this->encoder_ = std::move(encoder__);
		this->decoder_ = std::move(decoder__);
	}

	void UpdateAdamOpt
	(
		const std::vector<std::vector<MatrixXld>>& Target_input_output, /*std::vector<MatrixXld> Target_output,*/
		size_t epochs, size_t optima_steps, size_t batch_size,
		double learning_rate = 0.001L, double epsilon = 1e-8L,
		double beta1 = 0.9L, double beta2 = 0.999L
	);

	void UpdateAdamOptWithLogging
	(
		const std::vector<std::vector<MatrixXld>>& Target_input_output, /*std::vector<MatrixXld> Target_output,*/
		size_t epochs, size_t optima_steps, size_t batch_size,
		double learning_rate = 0.001L, double epsilon = 1e-8L,
		double beta1 = 0.9L, double beta2 = 0.999L
	);

	void Save(std::string packname) {
		std::filesystem::create_directories(packname);
		encoder_->Save(packname + "/" + "Encoder");
		decoder_->save(packname + "/" + "Decoder");
	}

	void Load(std::string packname) {
		encoder_->Load(packname + "/" + "Encoder");
		decoder_->load(packname + "/" + "Decoder");
	}

	void Inference();

	void Inference(const std::vector<MatrixXld>& input_sequence_batch);

	const std::vector<MatrixXld>& GetOutputs() const;

protected:

	//std::vector<MatrixXld> Target_outputs;  // [B][T_dec x Output_dim]

	std::unique_ptr<Seq2SeqWithAttention_ForTrain::Encoder> encoder_;
	std::unique_ptr<Seq2SeqWithAttention_ForTrain::Decoder> decoder_;
};
