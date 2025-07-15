#include "RealizationsNN_for_decoder.h"

class SimpleLSTM_ForTrain_ForDecoder : public Decoder_<SimpleLSTM_ForTrain> {
public:
	struct states_forgrads {
		std::vector<MatrixXld> f, i, o, ccond, c, h, context, z, x, p, p_;
	};
	states_forgrads StatesForgrads;

	friend class Seq2SeqWithAttention_ForTrain;
	/*void Batch_All_state_Сalculation(
		const std::vector<MatrixXld>& encoder_outputs,
		const std::vector<MatrixXld>& teacher_inputs,        // [B][T_dec x emb_dim]
		const std::vector<std::vector<bool>>& loss_mask,     // [B][T_dec]
		long double teacher_forcing_ratio)
	{
	}*/

	void All_state_Calculation() {
		if (this->encoder_outputs.empty()) return;

		auto apply_layernorm = [this](const RowVectorXld& x) -> RowVectorXld {
			long double mean = x.mean();
			long double variance = (x.array() - mean).square().mean();
			return ((x.array() - mean) / std::sqrt(variance + 1e-5L)).matrix().array() * layernorm_gamma.array() + layernorm_beta.array();
			};

		auto l2_normalize = [](const RowVectorXld& x) -> RowVectorXld {
			long double norm = std::sqrt(x.squaredNorm() + 1e-8L);
			return x / norm;
			};

		// --- Lambda: Масштабирование по max(abs)
		auto normalize_scale = [](RowVectorXld& vec) {
			long double maxval = vec.cwiseAbs().maxCoeff();
			if (maxval > 0.0L) vec /= maxval;
			};

		// Очистка
		Output_state.clear();
		this->StatesForgrads.context.clear();
		this->StatesForgrads.x.clear();
		this->StatesForgrads.p.clear();
		this->StatesForgrads.p_.clear();
		this->StatesForgrads.z.clear();
		this->StatesForgrads.f.clear();
		this->StatesForgrads.i.clear();
		this->StatesForgrads.o.clear();
		this->StatesForgrads.ccond.clear();
		this->StatesForgrads.c.clear();
		this->StatesForgrads.h.clear();
		attention_->ClearCache();

		size_t batch_size = encoder_outputs.size();
		Output_state.resize(batch_size);
		this->StatesForgrads.context.resize(batch_size);
		this->StatesForgrads.x.resize(batch_size);
		this->StatesForgrads.p.resize(batch_size);
		this->StatesForgrads.p_.resize(batch_size);
		this->StatesForgrads.z.resize(batch_size);
		this->StatesForgrads.f.resize(batch_size);
		this->StatesForgrads.i.resize(batch_size);
		this->StatesForgrads.o.resize(batch_size);
		this->StatesForgrads.ccond.resize(batch_size);
		this->StatesForgrads.c.resize(batch_size);
		this->StatesForgrads.h.resize(batch_size);

		// Общие веса
		MatrixXld W_x(Input_size, 4 * Hidden_size);
		W_x << W_F_I, W_I_I, W_C_I, W_O_I;

		MatrixXld W_h(Hidden_size, 4 * Hidden_size);
		W_h << W_F_H, W_I_H, W_C_H, W_O_H;

		RowVectorXld b(4 * Hidden_size);
		b << B_F, B_I, B_C, B_O;

		for (size_t n = 0; n < batch_size; ++n) {
			const auto& enc_out = encoder_outputs[n];
			std::vector<RowVectorXld> y_sequence;

			RowVectorXld y_prev = start_token;
			RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
			RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

			for (size_t t = 0; t < max_steps; ++t) {
				RowVectorXld context = attention_->ComputeContext(enc_out, h_prev);

				RowVectorXld decoder_input(Input_size);
				decoder_input << y_prev, context;
				decoder_input = l2_normalize(decoder_input);

				RowVectorXld Z = decoder_input * W_x + h_prev * W_h + b;

				RowVectorXld f_t = ActivationFunctions::Sigmoid(Z.leftCols(Hidden_size));
				RowVectorXld i_t = ActivationFunctions::Sigmoid(Z.middleCols(Hidden_size, Hidden_size));
				RowVectorXld c_bar = ActivationFunctions::Tanh(Z.middleCols(2 * Hidden_size, Hidden_size));
				RowVectorXld o_t = ActivationFunctions::Sigmoid(Z.rightCols(Hidden_size));

				RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_bar.array();
				RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();

				RowVectorXld proj_input_(Hidden_size + context.size());
				proj_input_ << h_t, context;
				auto proj_input = apply_layernorm(proj_input_);

				RowVectorXld y_t = proj_input * W_output.transpose() + b_output;
				y_sequence.push_back(y_t);

				this->StatesForgrads.f[n].row(t) = f_t;
				this->StatesForgrads.i[n].row(t) = i_t;
				this->StatesForgrads.ccond[n].row(t) = c_bar;
				this->StatesForgrads.o[n].row(t) = o_t;
				this->StatesForgrads.c[n].row(t) = c_t;
				this->StatesForgrads.h[n].row(t) = h_t;

				this->StatesForgrads.context[n].row(t) = context;
				this->StatesForgrads.x[n].row(t) = decoder_input;
				this->StatesForgrads.p[n].row(t) = proj_input_;
				this->StatesForgrads.p_[n].row(t) = proj_input;
				this->StatesForgrads.z[n].row(t) = Z;

				if (IsEndToken(y_t)) {
					/*size_t end_len = static_cast<size_t>(end_token.rows());
					if (y_sequence.size() >= end_len - 1) {
						y_sequence.resize(y_sequence.size() - (end_len - 1));
						context_sequence.resize(context_sequence.size() - (end_len - 1));
						u_sequence.resize(u_sequence.size() - (end_len - 1));
					}*/
					break;
				}

				y_prev = y_t;
				h_prev = h_t;
				c_prev = c_t;
			}

			// Преобразуем в матрицы
			Eigen::Index T = static_cast<Eigen::Index>(y_sequence.size());
			Eigen::Index D = static_cast<Eigen::Index>(y_sequence[0].cols());

			Output_state[n] = MatrixXld(T, D);

			for (Eigen::Index t = 0; t < T; ++t) {
				Output_state[n].row(t) = y_sequence[t];
			}
		}
	}
};

class SimpleLSTM_ForDecoder : public Decoder_<SimpleLSTM> {
public:

	/*void Batch_All_state_Сalculation(
		const std::vector<MatrixXld>& encoder_outputs,
		const std::vector<MatrixXld>& teacher_inputs,        // [B][T_dec x emb_dim]
		const std::vector<std::vector<bool>>& loss_mask,     // [B][T_dec]
		long double teacher_forcing_ratio)
	{
	}*/

	void All_state_Calculation() {
		if (this->encoder_outputs.empty()) return;

		auto apply_layernorm = [this](const RowVectorXld& x) -> RowVectorXld {
			long double mean = x.mean();
			long double variance = (x.array() - mean).square().mean();
			return ((x.array() - mean) / std::sqrt(variance + 1e-5L)).matrix().array() * layernorm_gamma.array() + layernorm_beta.array();
			};

		auto l2_normalize = [](const RowVectorXld& x) -> RowVectorXld {
			long double norm = std::sqrt(x.squaredNorm() + 1e-8L);
			return x / norm;
			};

		// --- Lambda: Масштабирование по max(abs)
		auto normalize_scale = [](RowVectorXld& vec) {
			long double maxval = vec.cwiseAbs().maxCoeff();
			if (maxval > 0.0L) vec /= maxval;
			};

		// Очистка
		Output_state.clear();
		//context_vectors.clear();
		//U_state.clear();
		attention_->ClearCache();

		size_t batch_size = encoder_outputs.size();
		Output_state.resize(batch_size);
		//context_vectors.resize(batch_size);
		//U_state.resize(batch_size);

		// Общие веса
		MatrixXld W_x(Input_size, 4 * Hidden_size);
		W_x << W_F_I, W_I_I, W_C_I, W_O_I;

		MatrixXld W_h(Hidden_size, 4 * Hidden_size);
		W_h << W_F_H, W_I_H, W_C_H, W_O_H;

		RowVectorXld b(4 * Hidden_size);
		b << B_F, B_I, B_C, B_O;

		for (size_t n = 0; n < batch_size; ++n) {
			const auto& enc_out = encoder_outputs[n];
			std::vector<RowVectorXld> y_sequence;
			//std::vector<RowVectorXld> context_sequence;
			//std::vector<RowVectorXld> u_sequence;

			RowVectorXld y_prev = start_token;
			RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
			RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

			for (size_t t = 0; t < max_steps; ++t) {
				RowVectorXld context = attention_->ComputeContext(enc_out, h_prev);
				//context_sequence.push_back(context);

				RowVectorXld decoder_input(Input_size);
				decoder_input << y_prev, context;
				decoder_input = l2_normalize(decoder_input);

				RowVectorXld Z = decoder_input * W_x + h_prev * W_h + b;

				RowVectorXld f_t = ActivationFunctions::Sigmoid(Z.leftCols(Hidden_size));
				RowVectorXld i_t = ActivationFunctions::Sigmoid(Z.middleCols(Hidden_size, Hidden_size));
				RowVectorXld c_bar = ActivationFunctions::Tanh(Z.middleCols(2 * Hidden_size, Hidden_size));
				RowVectorXld o_t = ActivationFunctions::Sigmoid(Z.rightCols(Hidden_size));

				RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_bar.array();
				RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();

				RowVectorXld proj_input(Hidden_size + context.size());
				proj_input << h_t, context;
				proj_input = apply_layernorm(proj_input);

				RowVectorXld y_t = proj_input * W_output.transpose() + b_output;
				//u_sequence.push_back(proj_input);
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

			// Преобразуем в матрицы
			Eigen::Index T = static_cast<Eigen::Index>(y_sequence.size());
			Eigen::Index D = static_cast<Eigen::Index>(y_sequence[0].cols());

			Output_state[n] = MatrixXld(T, D);
			//U_state[n] = MatrixXld(T, u_sequence[0].cols());
			//context_vectors[n] = MatrixXld(T, context_sequence[0].cols());

			for (Eigen::Index t = 0; t < T; ++t) {
				Output_state[n].row(t) = y_sequence[t];
				//this->states[n].row(t) = u_sequence[t];
				//context_vectors[n].row(t) = context_sequence[t];
			}
		}
	}
};