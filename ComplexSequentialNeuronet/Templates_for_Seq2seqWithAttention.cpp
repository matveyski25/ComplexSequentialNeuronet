#include "HeaderTemplates_for_Seq2seqWithAttention.h"

class Attention {
public:
	virtual ~Attention() = default;

	// Абстрактный метод: вычисляет контекст по шагу
	virtual RowVectorXld ComputeContext(const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden) = 0;

	// Очистка накопленных значений
	virtual void ClearCache() {
		all_attention_weights_.clear();
		all_scores_.clear();
		all_tanh_outputs_.clear();

	}

	// Получение attention-весов по всем временным шагам
	const std::vector<VectorXld>& GetAllAttentionWeights() const { return all_attention_weights_; }

	// Получение сырых score-векторов (до softmax)
	const std::vector<VectorXld>& GetAllScores() const { return all_scores_; }

protected:
	// Вспомогательные буферы для накопления истории attention по всем шагам
	std::vector<VectorXld> all_attention_weights_;  // α_t для всех t
	std::vector<VectorXld> all_scores_;             // e_{t,i} для всех t
	std::vector<std::vector<RowVectorXld>> all_tanh_outputs_;  // u_{ti} для всех t, i

};

template<class Base_of_encoder>
class Encoder : public Base_of_encoder {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		Encoder(Eigen::Index input_size, Eigen::Index hidden_size)
			: Base_of_encoder(input_size, hidden_size) {
		}
		Encoder() : Base_of_encoder() {};

		void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
			SetInput_states(input_sequence_batch);
			All_state_Сalculation();
		}

		const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
			return this->Common_Hidden_states;
		}
		virtual ~Encoder() = default;
	};

template<class Base_of_decoder>
class Decoder : public Base_of_decoder {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		Decoder(std::unique_ptr<Attention> attention_module,
			Eigen::Index hidden_size_encoder, Eigen::Index Hidden_size_, Eigen::Index embedding_dim_,
			RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_)
			: Base_of_decoder(embedding_dim_ + 2 * hidden_size_encoder/*= H_emb + 2H_enc*/, Hidden_size_), attention_(std::move(attention_module))
		{
			this->output_size = embedding_dim_;
			//размер контекста = 2 * Hidden_size_encoder = Number_states - embedding_dim
			size_t context_size = 2 * hidden_size_encoder;
			W_output = ActivationFunctions::matrix_random(output_size, Hidden_size_ + context_size);
			b_output = RowVectorXld::Zero(output_size);

			this->layernorm_gamma = RowVectorXld::Ones(Input_size);
			this->layernorm_beta = RowVectorXld::Zero(Input_size);
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
			this->All_state_Calculation();
		}

		const std::vector<MatrixXld>& GetOutputStates() const { return Output_state; }

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

		std::unique_ptr<Attention> attention_;

		std::vector<MatrixXld> encoder_outputs;
		//std::vector<MatrixXld> context_vectors;

		//std::vector<MatrixXld> U_state;

		std::vector<MatrixXld> Output_state;
		// --- Обновляемый выходной слой ---
		MatrixXld W_output;      // [output_size x (hidden_size + context_size)]
		RowVectorXld b_output;   // [1 x output_size]

		size_t output_size;
		//size_t embedding_dim;

		RowVectorXld layernorm_gamma; // [1 x Input_size]
		RowVectorXld layernorm_beta;  // [1 x Input_size]

	private:
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