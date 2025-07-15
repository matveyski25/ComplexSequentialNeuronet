#pragma once

#include "RealizationsNN_for_decoder.h"

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
}*/
class BahdanauAttention : public Attention {
public:
	friend class Seq2SeqWithAttention_ForTrain;////////////
	BahdanauAttention(Eigen::Index encoder_hidden_size, Eigen::Index decoder_hidden_size, Eigen::Index attention_size)
		: encoder_hidden_size_(encoder_hidden_size),
		decoder_hidden_size_(decoder_hidden_size),
		attention_size_(attention_size)
	{}
	// Вычисляет контекстный вектор и сохраняет внутренние веса
	RowVectorXld ComputeContext(const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden) override
	{}


protected:
	Eigen::Index encoder_hidden_size_;    // 2H
	Eigen::Index decoder_hidden_size_;    // H_dec
	Eigen::Index attention_size_;         // A

	MatrixXld W_encoder_;       // [A x 2H]
	MatrixXld W_decoder_;       // [A x H_dec]
	MatrixXld attention_vector_; // [A x 1]

};

class Seq2SeqWithAttention {
private:
	using Encoder = Encoder_<BiLSTM>;
	using Decoder = Decoder_<SimpleLSTM_ForDecoder, BahdanauAttention>;
public:
	template<typename EncoderT, typename DecoderT>
	Seq2SeqWithAttention(
		std::unique_ptr<EncoderT> encoder = std::make_unique<Encoder>,
		std::unique_ptr<DecoderT> decoder = std::make_unique<Decoder>) {}

	Seq2SeqWithAttention(){}

	void SetInput_states(const std::vector<MatrixXld>& _inputs) {}

	void Inference(){}

	void Inference(const std::vector<MatrixXld>& input_sequence_batch){}

	const std::vector<MatrixXld>& GetDecoderOutputs() const {}

	void Save(std::string packname) {}

	void Load(std::string packname) {}
protected:
	std::vector<MatrixXld> Input_States;

	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
};

class Seq2SeqWithAttention_ForTrain : public Seq2SeqWithAttention {
private:
	using Encoder = Encoder_<BiLSTM_ForTrain>;
	using Decoder = Decoder_<SimpleLSTM_ForTrain_ForDecoder, BahdanauAttention>;
public:
	Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train = std::make_unique<Encoder>(), std::unique_ptr<Decoder> decoder_train = std::make_unique<Decoder>())
		: Seq2SeqWithAttention(std::move(encoder_train), std::move(decoder_train)) {}

	struct grads_Seq2SeqWithAttention {
		MatrixXld dW_out; MatrixXld dB_out;

		MatrixXld dW_gamma_layernorm; MatrixXld dB_beta_layernorm;

		MatrixXld dV_a_attention, dW_e_attention, dW_d_attention;

		MatrixXld dW_f_dec, dU_f_dec; MatrixXld dB_f_dec;
		MatrixXld dW_i_dec, dU_i_dec; MatrixXld dB_i_dec;
		MatrixXld dW_c_dec, dU_c_dec; MatrixXld dB_c_dec;
		MatrixXld dW_o_dec, dU_o_dec; MatrixXld dB_o_dec;

		MatrixXld dW_f_forw_enc, dU_f_forw_enc; MatrixXld dB_f_forw_enc;
		MatrixXld dW_i_forw_enc, dU_i_forw_enc; MatrixXld dB_i_forw_enc;
		MatrixXld dW_c_forw_enc, dU_c_forw_enc; MatrixXld dB_c_forw_enc;
		MatrixXld dW_o_forw_enc, dU_o_forw_enc; MatrixXld dB_o_forw_enc;

		MatrixXld dW_f_back_enc, dU_f_back_enc; MatrixXld dB_f_back_enc;
		MatrixXld dW_i_back_enc, dU_i_back_enc; MatrixXld dB_i_back_enc;
		MatrixXld dW_c_back_enc, dU_c_back_enc; MatrixXld dB_c_back_enc;
		MatrixXld dW_o_back_enc, dU_o_back_enc; MatrixXld dB_o_back_enc;
	};

	struct states_forgrads {
		MatrixXld f_enc_forw, i_enc_forw, ccond_enc_forw, o_enc_forw, c_enc_forw;
		MatrixXld f_enc_back, i_enc_back, ccond_enc_back, o_enc_back, c_enc_back;
	};

	grads_Seq2SeqWithAttention Backward(size_t Number_InputState, MatrixXld Y_True) {}

	std::vector<MatrixXld> Target_outputs;  // [B][T_dec x Output_dim]

	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
};
