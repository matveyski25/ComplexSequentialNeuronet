#pragma once

#include "HeaderLSTM_and_BiLSTM.h"

#include <filesystem>

class Attention {
public:
	virtual ~Attention();

	// Абстрактный метод: вычисляет контекст по шагу
	virtual RowVectorXld ComputeContext(const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden);

	// Очистка накопленных значений
	virtual void ClearCache() {}

	// Получение attention-весов по всем временным шагам
	const std::vector<VectorXld>& GetAllAttentionWeights() const {}

	// Получение сырых score-векторов (до softmax)
	const std::vector<VectorXld>& GetAllScores() const {}

protected:
	// Вспомогательные буферы для накопления истории attention по всем шагам
	std::vector<VectorXld> all_attention_weights_;  // α_t для всех t
	std::vector<VectorXld> all_scores_;             // e_{t,i} для всех t
	std::vector<std::vector<RowVectorXld>> all_tanh_outputs_;  // u_{ti} для всех t, i

};

template<class Base_of_encoder>
class Encoder_ : public Base_of_encoder {
	public:
		friend class Seq2SeqWithAttention_ForTrain;/////////////////
		using Base_of_encoder::Base_of_encoder;
		Encoder_() : Base_of_encoder() {};

		void Encode(const std::vector<MatrixXld>& input_sequence_batch) {}

		const std::vector<MatrixXld>& GetEncodedHiddenStates() const {}
	};

template<class Base_of_decoder>
class Decoder_ : public Base_of_decoder {
	public:
		friend class Seq2SeqWithAttention_ForTrain; /////////////
		Decoder_(std::unique_ptr<Attention> attention_module,
			Eigen::Index hidden_size_encoder, Eigen::Index Hidden_size_, Eigen::Index embedding_dim_,
			RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_)
			: Base_of_decoder(embedding_dim_ + 2 * hidden_size_encoder/*= H_emb + 2H_enc*/, Hidden_size_), attention_(std::move(attention_module))
		{}

		Decoder_();

		void SetEncoderOutputs(const std::vector<MatrixXld>& encoder_outputs) {}

		void Decode(const std::vector<MatrixXld>& encoder_outputs) {}

		const std::vector<MatrixXld>& GetOutputStates() const {}

	protected:
		bool IsEndToken(const RowVectorXld& vec) const {}

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
		void All_state_Calculation() {}
	};

