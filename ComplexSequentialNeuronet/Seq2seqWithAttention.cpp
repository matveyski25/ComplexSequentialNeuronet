#include "HeaderSeq2seqWithAttention.h"
 

template<typename EncoderT, typename DecoderT>
Seq2SeqWithAttention::Seq2SeqWithAttention(
	std::unique_ptr<EncoderT> encoder,
	std::unique_ptr<DecoderT> decoder)
		: encoder_(std::move(encoder)), decoder_(std::move(decoder)) {
	}

Seq2SeqWithAttention::Seq2SeqWithAttention(
	Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
	Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
	std::unique_ptr<BahdanauAttention> attention_)
		:
		encoder_(std::make_unique<Encoder>(Input_size_, Encoder_Hidden_size_)),
		decoder_(std::make_unique<Decoder>(attention_, Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {
	}

void Seq2SeqWithAttention::SetInput_states(const std::vector<MatrixXld>& _inputs) {
		this->Input_States = _inputs;
	}

void Seq2SeqWithAttention::Inference()
	{
		if (this->Input_States.empty()) { throw std::invalid_argument("Вход пустой"); }
		encoder_->Encode(this->Input_States);
		decoder_->Decode(encoder_->GetEncodedHiddenStates());
	}

void Seq2SeqWithAttention::Inference(const std::vector<MatrixXld>& input_sequence_batch)
	{
		SetInput_states(input_sequence_batch);
		encoder_->Encode(this->Input_States);
		decoder_->Decode(encoder_->GetEncodedHiddenStates());
	}

const std::vector<MatrixXld>& Seq2SeqWithAttention::GetDecoderOutputs() const {
		return decoder_->GetOutputStates();
	}

void Seq2SeqWithAttention::Save(std::string packname) {
		std::filesystem::create_directories(packname);
		encoder_->Save(packname + "/" + "Encoder");
		decoder_->save(packname + "/" + "Decoder");
	}

void Seq2SeqWithAttention::Load(std::string packname) {
		encoder_->Load(packname + "/" + "Encoder");
		decoder_->load(packname + "/" + "Decoder");
	}



Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train, std::unique_ptr<Decoder> decoder_train)
	: Seq2SeqWithAttention(std::move(encoder_train), std::move(decoder_train)) {
}

Seq2SeqWithAttention_ForTrain::grads_Seq2SeqWithAttention Seq2SeqWithAttention_ForTrain::Backward(size_t Number_InputState, MatrixXld Y_True) {
		grads_Seq2SeqWithAttention grads;
		{
			Eigen::Index E = this->decoder_->output_size;
			Eigen::Index H = this->decoder_->Hidden_size;
			Eigen::Index C = this->decoder_->Input_size - E;
			Eigen::Index D = H + C;
			Eigen::Index A = this->decoder_->attention_->attention_size_;
			Eigen::Index X = E + C;
			Eigen::Index HE = this->encoder_->Common_Hidden_size;
			Eigen::Index EE = this->encoder_->Common_Input_size;
			grads.dW_out.conservativeResize(E, D), grads.dB_out.conservativeResize(1, E);

			grads.dW_gamma_layernorm.conservativeResize(1, D), grads.dB_beta_layernorm.conservativeResize(1, D);

			grads.dV_a_attention.conservativeResize(A, 1), grads.dW_e_attention.conservativeResize(A, C), grads.dW_d_attention.conservativeResize(A, H);

			grads.dW_f_dec.conservativeResize(H, X), grads.dU_f_dec.conservativeResize(H, H), grads.dB_f_dec.conservativeResize(1, H),
				grads.dW_i_dec.conservativeResize(H, X), grads.dU_i_dec.conservativeResize(H, H), grads.dB_i_dec.conservativeResize(1, H),
				grads.dW_c_dec.conservativeResize(H, X), grads.dU_c_dec.conservativeResize(H, H), grads.dB_c_dec.conservativeResize(1, H),
				grads.dW_o_dec.conservativeResize(H, X), grads.dU_o_dec.conservativeResize(H, H), grads.dB_o_dec.conservativeResize(1, H);

			grads.dW_f_forw_enc.conservativeResize(HE, EE), grads.dU_f_forw_enc.conservativeResize(HE, HE), grads.dB_f_forw_enc.conservativeResize(1, HE),
				grads.dW_i_forw_enc.conservativeResize(HE, EE), grads.dU_i_forw_enc.conservativeResize(HE, HE), grads.dB_i_forw_enc.conservativeResize(1, HE),
				grads.dW_c_forw_enc.conservativeResize(HE, EE), grads.dU_c_forw_enc.conservativeResize(HE, HE), grads.dB_c_forw_enc.conservativeResize(1, HE),
				grads.dW_o_forw_enc.conservativeResize(HE, EE), grads.dU_o_forw_enc.conservativeResize(HE, HE), grads.dB_o_forw_enc.conservativeResize(1, HE);

			grads.dW_f_back_enc.conservativeResize(HE, EE), grads.dU_f_back_enc.conservativeResize(HE, HE), grads.dB_f_back_enc.conservativeResize(1, HE),
				grads.dW_i_back_enc.conservativeResize(HE, EE), grads.dU_i_back_enc.conservativeResize(HE, HE), grads.dB_i_back_enc.conservativeResize(1, HE),
				grads.dW_c_back_enc.conservativeResize(HE, EE), grads.dU_c_back_enc.conservativeResize(HE, HE), grads.dB_c_back_enc.conservativeResize(1, HE),
				grads.dW_o_back_enc.conservativeResize(HE, EE), grads.dU_o_back_enc.conservativeResize(HE, HE), grads.dB_o_back_enc.conservativeResize(1, HE);
		}

		Eigen::Index T = std::min(this->GetDecoderOutputs()[Number_InputState].rows(), Y_True.rows());
		Eigen::Index N = this->encoder_->Common_Input_states[Number_InputState].rows();

		RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
		for (Eigen::Index t = T; t >= 0; t--) {
			RowVectorXld dY_t = Y_True.row(t) - this->GetDecoderOutputs()[Number_InputState].row(t); //Y_true_t - Y_t
			MatrixXld DW_out_t = dY_t * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).transpose();
			RowVectorXld dp__t = this->decoder_->W_output.transpose() * dY_t;
			RowVectorXld DB_out_t = std::move(dY_t);

			RowVectorXld dS_t = dp__t.leftCols(this->decoder_->Hidden_size);
			RowVectorXld dContext_t = dp__t.middleCols(this->decoder_->Hidden_size, dp__t.cols() - this->decoder_->Hidden_size);

			RowVectorXld DGamma_t = dp__t.array() * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).array();
			RowVectorXld DBeta_t = dp__t;

			RowVectorXld F_t = this->decoder_->StatesForgrads.f[Number_InputState].row(t);
			RowVectorXld I_t = this->decoder_->StatesForgrads.i[Number_InputState].row(t);
			RowVectorXld Ccond_t = this->decoder_->StatesForgrads.ccond[Number_InputState].row(t);
			RowVectorXld O_t = this->decoder_->StatesForgrads.o[Number_InputState].row(t);
			RowVectorXld C_t = this->decoder_->StatesForgrads.c[Number_InputState].row(t);
			RowVectorXld C_t_l;
			if (t == 0) {
				C_t_l = RowVectorXld::Zero(this->decoder_->StatesForgrads.c[Number_InputState].row(t).cols());
			}
			else {
				C_t_l = this->decoder_->StatesForgrads.c[Number_InputState].row(t - 1);
			}

			RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (MatrixXld::Constant((O_t).size(), 1) - O_t).array();
			RowVectorXld dC_t = dS_t.array() * O_t.array() *
				(MatrixXld::Constant((C_t * C_t).size(), 1) - (ActivationFunctions::Tanh(C_t) * ActivationFunctions::Tanh(C_t))).array() +
				_dC_t.array() * F_t.array();
			RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (MatrixXld::Constant((Ccond_t * Ccond_t).size(), 1) - Ccond_t * Ccond_t).array();
			RowVectorXld dI_t = dC_t.array() * I_t.array() * Ccond_t.array() * (MatrixXld::Constant((I_t).size(), 1) - I_t).array();
			RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (MatrixXld::Constant((F_t).size(), 1) - F_t).array();

			RowVectorXld dGates_t(4 * this->decoder_->Hidden_size);
			dGates_t << dF_t, dI_t, dCcond_t, dO_t;

			MatrixXld DW_dec_t = this->decoder_->StatesForgrads.x[Number_InputState].row(t).transpose() * dGates_t;
			MatrixXld DU_dec_t;
			if (t == 0) {
				DU_dec_t = MatrixXld::Zero(this->decoder_->StatesForgrads.h[Number_InputState].row(t).cols(), 4 * this->decoder_->Hidden_size);
			}
			else {
				DU_dec_t = this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1).transpose() * dGates_t;
			}
			VectorXld DB_dec_t = dGates_t;

			std::vector<MatrixXld> _dH_back;
			RowVectorXld Enc_Forw__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
			for (Eigen::Index j = N - 1; j >= 0; --j) {
				RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
				RowVectorXld s_t_1;
				if (t > 0) {
					s_t_1 = this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1);
				}
				else{
					s_t_1 = RowVectorXld::Zero(this->decoder_->Hidden_size);
				}


				long double alpha_j = this->decoder_->attention_->all_attention_weights_[t](j);
				long double dAlpha_j = dContext_t.dot(h_j);

				long double dE_tj = 0.0;
				for (int k = 0; k < N; ++k) {
					long double alpha_k = this->decoder_->attention_->all_attention_weights_[t](k);
					RowVectorXld h_k = this->encoder_->Common_Hidden_states[Number_InputState].row(k);
					long double dAlpha_k = dContext_t.dot(h_k);

					dE_tj += dAlpha_k * alpha_k * ((j == k) - alpha_j);  // ∂α_k / ∂e_j
				}

				RowVectorXld u_tj = this->decoder_->attention_->all_tanh_outputs_[t][j];
				RowVectorXld dU_tj = dE_tj * this->decoder_->attention_->attention_vector_.transpose();  // [1 x A]
				RowVectorXld dPreact_tj = dU_tj.array() * (1.0 - u_tj.array().square());

				MatrixXld DW_att_enc_tj = dPreact_tj.transpose() * h_j;   // [A x 1] * [1 x H_enc]
				MatrixXld DW_att_dec_tj = dPreact_tj.transpose() * s_t_1; // [A x 1] * [1 x H_dec]
				MatrixXld DV_att_tj = u_tj.transpose() * dE_tj;       // [A x 1]

				MatrixXld dH_j = dContext_t * alpha_j + this->decoder_->attention_->W_decoder_.transpose() * dU_tj;
				RowVectorXld dH_forw_j = dH_j.leftCols(this->encoder_->Common_Hidden_size);
				RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);
				
				_dH_back.push_back(dH_back_j);


				RowVectorXld Enc_Forw_F_j = this->encoder_->Forward.statesForgrads.f[Number_InputState].row(j);
				RowVectorXld Enc_Forw_I_j = this->encoder_->Forward.statesForgrads.i[Number_InputState].row(j);
				RowVectorXld Enc_Forw_Ccond_j = this->encoder_->Forward.statesForgrads.ccond[Number_InputState].row(j);
				RowVectorXld Enc_Forw_O_j = this->encoder_->Forward.statesForgrads.o[Number_InputState].row(j);
				RowVectorXld Enc_Forw_C_j = this->encoder_->Forward.statesForgrads.c[Number_InputState].row(j);
				RowVectorXld Enc_Forw_C_j_l;
				if (j == 0) {
					Enc_Forw_C_j_l = RowVectorXld::Zero(this->encoder_->Forward.statesForgrads.c[Number_InputState].row(j).cols());
				}
				else {
					Enc_Forw_C_j_l = this->encoder_->Forward.statesForgrads.c[Number_InputState].row(j - 1);
				}

				RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (MatrixXld::Constant((Enc_Forw_O_j).size(), 1) - Enc_Forw_O_j).array();
				RowVectorXld dEnc_Forw_C_j = dH_forw_j.array() * Enc_Forw_O_j.array() *
					(MatrixXld::Constant((Enc_Forw_C_j * Enc_Forw_C_j).size(), 1) - ActivationFunctions::Tanh(Enc_Forw_C_j) * ActivationFunctions::Tanh(Enc_Forw_C_j)).array() +
					Enc_Forw__dC_j.array() * Enc_Forw_F_j.array();
				RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (MatrixXld::Constant((Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).size(), 1) - Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).array();
				RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * Enc_Forw_Ccond_j.array() * (MatrixXld::Constant((Enc_Forw_I_j).size(), 1) - Enc_Forw_I_j).array();
				RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (MatrixXld::Constant((Enc_Forw_F_j).size(), 1) - Enc_Forw_F_j).array();

				RowVectorXld dEnc_Forw_Gates_j(4 * this->encoder_->Common_Hidden_size);
				dEnc_Forw_Gates_j << dEnc_Forw_F_j, dEnc_Forw_I_j, dEnc_Forw_Ccond_j, dEnc_Forw_O_j;

				MatrixXld DW_Enc_Forw_j = this->encoder_->Common_Input_states[Number_InputState].row(t).transpose() * dEnc_Forw_Gates_j;
				MatrixXld DU_Enc_Forw_j;
				if (t == 0) {
					DU_Enc_Forw_j = MatrixXld::Zero(this->encoder_->Forward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
				}
				else {
					DU_Enc_Forw_j = this->encoder_->Forward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Forw_Gates_j;
				}
				VectorXld DB_Enc_Forw_j = dEnc_Forw_Gates_j;
			}

		}
	}

/*class Seq2SeqWithAttention {
private:
	using Encoder = Encoder_<BiLSTM>;
	using Decoder = Decoder_<SimpleLSTM_ForDecoder, BahdanauAttention>;
public:
	template<typename EncoderT, typename DecoderT>
	Seq2SeqWithAttention(
		std::unique_ptr<EncoderT> encoder = std::make_unique<Encoder>,
		std::unique_ptr<DecoderT> decoder = std::make_unique<Decoder>)
		: encoder_(std::move(encoder)), decoder_(std::move(decoder)) {
	}

	Seq2SeqWithAttention(
		Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
		Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
		std::unique_ptr<BahdanauAttention> attention_ = std::make_unique<BahdanauAttention>())
		:
		encoder_(std::make_unique<Encoder>(Input_size_, Encoder_Hidden_size_)),
		decoder_(std::make_unique<Decoder>(attention_, Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {
	}

	void SetInput_states(const std::vector<MatrixXld>& _inputs) {
		this->Input_States = _inputs;
	}

	void Inference()
	{
		if (this->Input_States.empty()) { throw std::invalid_argument("Вход пустой"); }
		encoder_->Encode(this->Input_States);
		decoder_->Decode(encoder_->GetEncodedHiddenStates());
	}

	void Inference(const std::vector<MatrixXld>& input_sequence_batch)
	{
		SetInput_states(input_sequence_batch);
		encoder_->Encode(this->Input_States);
		decoder_->Decode(encoder_->GetEncodedHiddenStates());
	}

	const std::vector<MatrixXld>& GetDecoderOutputs() const {
		return decoder_->GetOutputStates();
	}

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

	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
};

class Seq2SeqWithAttention_ForTrain : public Seq2SeqWithAttention {
private:
	using Encoder = Encoder_<BiLSTM_ForTrain>;
	using Decoder = Decoder_<SimpleLSTM_ForTrain_ForDecoder, BahdanauAttention>;
public:
	Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train = std::make_unique<Encoder>(), std::unique_ptr<Decoder> decoder_train = std::make_unique<Decoder>())
		: Seq2SeqWithAttention(std::move(encoder_train), std::move(decoder_train)) {
	}

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

	grads_Seq2SeqWithAttention Backward(size_t Number_InputState, MatrixXld Y_True) {
		grads_Seq2SeqWithAttention grads;
		{
			Eigen::Index E = this->decoder_->output_size;
			Eigen::Index H = this->decoder_->Hidden_size;
			Eigen::Index C = this->decoder_->Input_size - E;
			Eigen::Index D = H + C;
			Eigen::Index A = this->decoder_->attention_->attention_size_;
			Eigen::Index X = E + C;
			Eigen::Index HE = this->encoder_->Common_Hidden_size;
			Eigen::Index EE = this->encoder_->Common_Input_size;
			grads.dW_out.conservativeResize(E, D), grads.dB_out.conservativeResize(1, E);

			grads.dW_gamma_layernorm.conservativeResize(1, D), grads.dB_beta_layernorm.conservativeResize(1, D);

			grads.dV_a_attention.conservativeResize(A, 1), grads.dW_e_attention.conservativeResize(A, C), grads.dW_d_attention.conservativeResize(A, H);

			grads.dW_f_dec.conservativeResize(H, X), grads.dU_f_dec.conservativeResize(H, H), grads.dB_f_dec.conservativeResize(1, H),
				grads.dW_i_dec.conservativeResize(H, X), grads.dU_i_dec.conservativeResize(H, H), grads.dB_i_dec.conservativeResize(1, H),
				grads.dW_c_dec.conservativeResize(H, X), grads.dU_c_dec.conservativeResize(H, H), grads.dB_c_dec.conservativeResize(1, H),
				grads.dW_o_dec.conservativeResize(H, X), grads.dU_o_dec.conservativeResize(H, H), grads.dB_o_dec.conservativeResize(1, H);

			grads.dW_f_forw_enc.conservativeResize(HE, EE), grads.dU_f_forw_enc.conservativeResize(HE, HE), grads.dB_f_forw_enc.conservativeResize(1, HE),
				grads.dW_i_forw_enc.conservativeResize(HE, EE), grads.dU_i_forw_enc.conservativeResize(HE, HE), grads.dB_i_forw_enc.conservativeResize(1, HE),
				grads.dW_c_forw_enc.conservativeResize(HE, EE), grads.dU_c_forw_enc.conservativeResize(HE, HE), grads.dB_c_forw_enc.conservativeResize(1, HE),
				grads.dW_o_forw_enc.conservativeResize(HE, EE), grads.dU_o_forw_enc.conservativeResize(HE, HE), grads.dB_o_forw_enc.conservativeResize(1, HE);

			grads.dW_f_back_enc.conservativeResize(HE, EE), grads.dU_f_back_enc.conservativeResize(HE, HE), grads.dB_f_back_enc.conservativeResize(1, HE),
				grads.dW_i_back_enc.conservativeResize(HE, EE), grads.dU_i_back_enc.conservativeResize(HE, HE), grads.dB_i_back_enc.conservativeResize(1, HE),
				grads.dW_c_back_enc.conservativeResize(HE, EE), grads.dU_c_back_enc.conservativeResize(HE, HE), grads.dB_c_back_enc.conservativeResize(1, HE),
				grads.dW_o_back_enc.conservativeResize(HE, EE), grads.dU_o_back_enc.conservativeResize(HE, HE), grads.dB_o_back_enc.conservativeResize(1, HE);
		}

		Eigen::Index T = std::min(this->GetDecoderOutputs()[Number_InputState].rows(), Y_True.rows());
		Eigen::Index N = this->encoder_->Common_Input_states[Number_InputState].rows();

		RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
		for (Eigen::Index t = T; t >= 0; t--) {
			RowVectorXld dY_t = Y_True.row(t) - this->GetDecoderOutputs()[Number_InputState].row(t); //Y_true_t - Y_t
			MatrixXld DW_out_t = dY_t * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).transpose();
			RowVectorXld dp__t = this->decoder_->W_output.transpose() * dY_t;
			RowVectorXld DB_out_t = std::move(dY_t);

			RowVectorXld dS_t = dp__t.leftCols(this->decoder_->Hidden_size);
			RowVectorXld dContext_t = dp__t.middleCols(this->decoder_->Hidden_size, dp__t.cols() - this->decoder_->Hidden_size);

			RowVectorXld DGamma_t = dp__t.array() * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).array();
			RowVectorXld DBeta_t = dp__t;

			RowVectorXld F_t = this->decoder_->StatesForgrads.f[Number_InputState].row(t);
			RowVectorXld I_t = this->decoder_->StatesForgrads.i[Number_InputState].row(t);
			RowVectorXld Ccond_t = this->decoder_->StatesForgrads.ccond[Number_InputState].row(t);
			RowVectorXld O_t = this->decoder_->StatesForgrads.o[Number_InputState].row(t);
			RowVectorXld C_t = this->decoder_->StatesForgrads.c[Number_InputState].row(t);
			RowVectorXld C_t_l;
			if (t == 0) {
				C_t_l = RowVectorXld::Zero(this->decoder_->StatesForgrads.c[Number_InputState].row(t).cols());
			}
			else {
				C_t_l = this->decoder_->StatesForgrads.c[Number_InputState].row(t - 1);
			}

			RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (MatrixXld::Constant((O_t).size(), 1) - O_t).array();
			RowVectorXld dC_t = dS_t.array() * O_t.array() *
				(MatrixXld::Constant((C_t * C_t).size(), 1) - ActivationFunctions::Tanh(C_t) * ActivationFunctions::Tanh(C_t)).array() +
				_dC_t.array() * F_t.array();
			RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (MatrixXld::Constant((Ccond_t * Ccond_t).size(), 1) - Ccond_t * Ccond_t).array();
			RowVectorXld dI_t = dC_t.array() * I_t.array() * Ccond_t.array() * (MatrixXld::Constant((I_t).size(), 1) - I_t).array();
			RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (MatrixXld::Constant((F_t).size(), 1) - F_t).array();

			RowVectorXld dGates_t(4 * this->decoder_->Hidden_size);
			dGates_t << dF_t, dI_t, dCcond_t, dO_t;

			MatrixXld DW_dec_t = this->decoder_->StatesForgrads.x[Number_InputState].row(t).transpose() * dGates_t;
			MatrixXld DU_dec_t;
			if (t == 0) {
				DU_dec_t = MatrixXld::Zero(this->decoder_->StatesForgrads.h[Number_InputState].row(t).cols(), 4 * this->decoder_->Hidden_size);
			}
			else {
				DU_dec_t = this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1).transpose() * dGates_t;
			}
			VectorXld DB_dec_t = dGates_t;

			std::vector<MatrixXld> _dH_back;
			RowVectorXld Enc_Forw__dC_t = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
			for (Eigen::Index j = N - 1; j >= 0; --j) {
				RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
				RowVectorXld s_t_1 = (t > 0) ? this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1)
					: RowVectorXld::Zero(this->decoder_->Hidden_size);

				long double alpha_j = this->decoder_->attention_->all_attention_weights_[t](j);
				long double dAlpha_j = dContext_t.dot(h_j);

				long double dE_tj = 0.0;
				for (int k = 0; k < N; ++k) {
					long double alpha_k = this->decoder_->attention_->all_attention_weights_[t](k);
					RowVectorXld h_k = this->encoder_->Common_Hidden_states[Number_InputState].row(k);
					long double dAlpha_k = dContext_t.dot(h_k);

					dE_tj += dAlpha_k * alpha_k * ((j == k) - alpha_j);  // ∂α_k / ∂e_j
				}

				RowVectorXld u_tj = this->decoder_->attention_->all_tanh_outputs_[t][j];
				RowVectorXld dU_tj = dE_tj * this->decoder_->attention_->attention_vector_.transpose();  // [1 x A]
				RowVectorXld dPreact_tj = dU_tj.array() * (1.0 - u_tj.array().square());

				MatrixXld DW_att_enc_tj = dPreact_tj.transpose() * h_j;   // [A x 1] * [1 x H_enc]
				MatrixXld DW_att_dec_tj = dPreact_tj.transpose() * s_t_1; // [A x 1] * [1 x H_dec]
				MatrixXld DV_att_tj = u_tj.transpose() * dE_tj;       // [A x 1]

				MatrixXld dH_j = dContext_t * alpha_j + this->decoder_->attention_->W_decoder_.transpose() * dU_tj;
				RowVectorXld dH_forw_j = dH_j.leftCols(this->encoder_->Common_Hidden_size);
				RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);

				_dH_back.push_back(dH_back_j);


				RowVectorXld Enc_Forw_F_j = this->encoder_->Forward->StatesForgrads.f[Number_InputState].row(j);
				RowVectorXld Enc_Forw_I_j = this->encoder_->Forward->StatesForgrads.i[Number_InputState].row(j);
				RowVectorXld Enc_Forw_Ccond_j = this->encoder_->Forward->StatesForgrads.ccond[Number_InputState].row(j);
				RowVectorXld Enc_Forw_O_j = this->encoder_->Forward->StatesForgrads.o[Number_InputState].row(j);
				RowVectorXld Enc_Forw_C_j = this->encoder_->Forward->StatesForgrads.c[Number_InputState].row(j);
				RowVectorXld Enc_Forw_C_j_l;
				if (j == 0) {
					Enc_Forw_C_j_l = RowVectorXld::Zero(this->encoder_->Forward->StatesForgrads.c[Number_InputState].row(j).cols());
				}
				else {
					Enc_Forw_C_j_l = this->encoder_->Forward->StatesForgrads.c[Number_InputState].row(j - 1);
				}

				RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (MatrixXld::Constant((Enc_Forw_O_j).size(), 1) - Enc_Forw_O_j).array();
				RowVectorXld dEnc_Forw_C_j = dH_forw_j.array() * Enc_Forw_O_j.array() *
					(MatrixXld::Constant((Enc_Forw_C_j * Enc_Forw_C_j).size(), 1) - ActivationFunctions::Tanh(Enc_Forw_C_j) * ActivationFunctions::Tanh(Enc_Forw_C_j)).array() +
					Enc_Forw__dC_j.array() * Enc_Forw_F_j.array();
				RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (MatrixXld::Constant((Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).size(), 1) - Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).array();
				RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * Enc_Forw_Ccond_j.array() * (MatrixXld::Constant((Enc_Forw_I_j).size(), 1) - Enc_Forw_I_j).array();
				RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (MatrixXld::Constant((Enc_Forw_F_j).size(), 1) - Enc_Forw_F_j).array();

				RowVectorXld dEnc_Forw_Gates_j(4 * this->encoder->Common_Hidden_size);
				dEnc_Forw_Gates_j << dEnc_Forw_F_j, dEnc_Forw_I_j, dEnc_Forw_Ccond_j, dEnc_Forw_O_j;

				MatrixXld DW_Enc_Forw_j = this->encoder_->Common_Input_states[Number_InputState].row(t).transpose() * dEnc_Forw_Gates_j;
				MatrixXld DU_Enc_Forw_j;
				if (t == 0) {
					DU_Enc_Forw_j = MatrixXld::Zero(this->encoder_->Forward->StatesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder->Common_Hidden_size);
				}
				else {
					DU_Enc_Forw_j = this->encoder_->Forward->StatesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Forw_Gates_j;
				}
				VectorXld DB_Enc_Forw_j = dEnc_Forw_Gates_j;
			}

		}
	}

	std::vector<MatrixXld> Target_outputs;  // [B][T_dec x Output_dim]

	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
};*/


int main() {
	setlocale(LC_ALL, "Russian");

	return 0;
}