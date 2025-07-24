#include "HeaderSeq2seqWithAttention.h"

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train, std::unique_ptr<Decoder> decoder_train)
	: encoder_(std::move(encoder_train)), decoder_(std::move(decoder_train)) {
}

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(
	Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
	Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
	std::unique_ptr<BahdanauAttention> attention_, size_t batch_size)
	:
	encoder_(std::make_unique<Encoder>(batch_size, Input_size_, Encoder_Hidden_size_)),
	decoder_(std::make_unique<Decoder>(std::move(attention_), Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {
}

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(
	Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_,
	Eigen::Index Decoder_Hidden_size_, Eigen::Index Attention_size_,
	Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_, size_t batch_size) : 
	encoder_(std::make_unique<Encoder>(batch_size, Input_size_, Encoder_Hidden_size_)),
	decoder_(std::make_unique<Decoder>(std::make_unique<BahdanauAttention>(Encoder_Hidden_size_, Decoder_Hidden_size_, Attention_size_), Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {

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
			grads.dW_ccond_dec.conservativeResize(H, X), grads.dU_ccond_dec.conservativeResize(H, H), grads.dB_ccond_dec.conservativeResize(1, H),
			grads.dW_o_dec.conservativeResize(H, X), grads.dU_o_dec.conservativeResize(H, H), grads.dB_o_dec.conservativeResize(1, H);

		grads.dW_f_forw_enc.conservativeResize(HE, EE), grads.dU_f_forw_enc.conservativeResize(HE, HE), grads.dB_f_forw_enc.conservativeResize(1, HE),
			grads.dW_i_forw_enc.conservativeResize(HE, EE), grads.dU_i_forw_enc.conservativeResize(HE, HE), grads.dB_i_forw_enc.conservativeResize(1, HE),
			grads.dW_ccond_forw_enc.conservativeResize(HE, EE), grads.dU_ccond_forw_enc.conservativeResize(HE, HE), grads.dB_ccond_forw_enc.conservativeResize(1, HE),
			grads.dW_o_forw_enc.conservativeResize(HE, EE), grads.dU_o_forw_enc.conservativeResize(HE, HE), grads.dB_o_forw_enc.conservativeResize(1, HE);

		grads.dW_f_back_enc.conservativeResize(HE, EE), grads.dU_f_back_enc.conservativeResize(HE, HE), grads.dB_f_back_enc.conservativeResize(1, HE),
			grads.dW_i_back_enc.conservativeResize(HE, EE), grads.dU_i_back_enc.conservativeResize(HE, HE), grads.dB_i_back_enc.conservativeResize(1, HE),
			grads.dW_ccond_back_enc.conservativeResize(HE, EE), grads.dU_ccond_back_enc.conservativeResize(HE, HE), grads.dB_ccond_back_enc.conservativeResize(1, HE),
			grads.dW_o_back_enc.conservativeResize(HE, EE), grads.dU_o_back_enc.conservativeResize(HE, HE), grads.dB_o_back_enc.conservativeResize(1, HE);
	}

	Eigen::Index T = std::min(this->GetOutputs()[Number_InputState].rows(), Y_True.rows());
	Eigen::Index N = this->encoder_->Common_Input_states[Number_InputState].rows();

	RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	RowVectorXld _dS_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	for (Eigen::Index t = T; t >= 0; t--) {
		RowVectorXld dY_t = Y_True.row(t) - this->GetOutputs()[Number_InputState].row(t); //Y_true_t - Y_t
		MatrixXld DW_out_t = dY_t * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).transpose();
		RowVectorXld dp__t = this->decoder_->W_output.transpose() * dY_t;
		RowVectorXld DB_out_t = std::move(dY_t);

		RowVectorXld dS_t = dp__t.leftCols(this->decoder_->Hidden_size) + _dS_t;
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

		RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (MatrixXld::Constant(O_t.rows(), O_t.cols(), 1) - O_t).array();
		RowVectorXld dC_t = dS_t.array() * O_t.array() *
			(MatrixXld::Constant((C_t).rows(), (C_t).cols(), 1).array() - (ActivationFunctions::Tanh(C_t).array() * ActivationFunctions::Tanh(C_t).array())) +
			_dC_t.array();
		RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (MatrixXld::Constant(Ccond_t.rows(), Ccond_t.cols(), 1).array() - (Ccond_t.array() * Ccond_t.array()));
		RowVectorXld dI_t = dC_t.array() * I_t.array() * Ccond_t.array() * (MatrixXld::Constant(I_t.rows(), I_t.cols(), 1) - I_t).array();
		RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (MatrixXld::Constant(F_t.rows(), F_t.cols(), 1) - F_t).array();

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

		_dC_t = dC_t.array() * F_t.array();
		MatrixXld U(this->decoder_->Hidden_size, 4 * this->decoder_->Hidden_size);
		U << this->decoder_->U_F, this->decoder_->U_I, this->decoder_->U_C, this->decoder_->U_O;
		_dS_t = U.transpose() * dGates_t;

		std::vector<MatrixXld> _dH_Back;
		RowVectorXld Enc_Forw__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
		RowVectorXld Enc_Forw__dH_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);

		for (Eigen::Index j = N - 1; j >= 0; j--) {
			RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
			RowVectorXld s_t_1;
			if (t > 0) {
				s_t_1 = this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1);
			}
			else {
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
			dH_forw_j += Enc_Forw__dH_j;
			RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);

			_dH_Back.push_back(dH_back_j);


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

			RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (MatrixXld::Constant(Enc_Forw_O_j.rows(), Enc_Forw_O_j.cols(), 1) - Enc_Forw_O_j).array();
			RowVectorXld dEnc_Forw_C_j = dH_forw_j.array() * Enc_Forw_O_j.array() *
				(MatrixXld::Constant(Enc_Forw_C_j.rows(), Enc_Forw_C_j.cols(), 1).array() - ActivationFunctions::Tanh(Enc_Forw_C_j).array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array()) +
				Enc_Forw__dC_j.array();
			RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (MatrixXld::Constant(Enc_Forw_Ccond_j.rows(), Enc_Forw_Ccond_j.cols(), 1).array() - Enc_Forw_Ccond_j.array() * Enc_Forw_Ccond_j.array());
			RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * Enc_Forw_Ccond_j.array() * (MatrixXld::Constant(Enc_Forw_I_j.rows(), Enc_Forw_I_j.cols(), 1) - Enc_Forw_I_j).array();
			RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (MatrixXld::Constant(Enc_Forw_F_j.rows(), Enc_Forw_F_j.cols(), 1) - Enc_Forw_F_j).array();

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

			Enc_Forw__dC_j = dEnc_Forw_C_j.array() * Enc_Forw_F_j.array();
			MatrixXld U_enc_f(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);
			U_enc_f << this->encoder_->Forward.U_F, this->encoder_->Forward.U_I, this->encoder_->Forward.U_C, this->encoder_->Forward.U_O;
			Enc_Forw__dH_j = U_enc_f.transpose() * dEnc_Forw_Gates_j;

			grads.dV_a_attention += std::move(DV_att_tj);
			grads.dW_e_attention += std::move(DW_att_enc_tj);
			grads.dW_d_attention += std::move(DW_att_dec_tj);

			grads.dW_f_forw_enc += std::move(DW_Enc_Forw_j.leftCols(this->encoder_->Common_Hidden_size));
			grads.dW_i_forw_enc += std::move(DW_Enc_Forw_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dW_ccond_forw_enc += std::move(DW_Enc_Forw_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dW_o_forw_enc += std::move(DW_Enc_Forw_j.rightCols(this->encoder_->Common_Hidden_size));

			grads.dU_f_forw_enc += std::move(DU_Enc_Forw_j.leftCols(this->encoder_->Common_Hidden_size));
			grads.dU_i_forw_enc += std::move(DU_Enc_Forw_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dU_ccond_forw_enc += std::move(DU_Enc_Forw_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dU_o_forw_enc += std::move(DU_Enc_Forw_j.rightCols(this->encoder_->Common_Hidden_size));

			grads.dB_f_forw_enc += std::move(DB_Enc_Forw_j.leftCols(this->encoder_->Common_Hidden_size));
			grads.dB_i_forw_enc += std::move(DB_Enc_Forw_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dB_ccond_forw_enc += std::move(DB_Enc_Forw_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dB_o_forw_enc += std::move(DB_Enc_Forw_j.rightCols(this->encoder_->Common_Hidden_size));
		}

		RowVectorXld Enc_Back__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
		RowVectorXld Enc_Back__dH_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);


		for (Eigen::Index j = 0; j < N; j++) {
			auto dH_Back_j = _dH_Back[j];
			dH_Back_j += Enc_Back__dH_j;
			RowVectorXld Enc_Back_F_j = this->encoder_->Backward.statesForgrads.f[Number_InputState].row(j);
			RowVectorXld Enc_Back_I_j = this->encoder_->Backward.statesForgrads.i[Number_InputState].row(j);
			RowVectorXld Enc_Back_Ccond_j = this->encoder_->Backward.statesForgrads.ccond[Number_InputState].row(j);
			RowVectorXld Enc_Back_O_j = this->encoder_->Backward.statesForgrads.o[Number_InputState].row(j);
			RowVectorXld Enc_Back_C_j = this->encoder_->Backward.statesForgrads.c[Number_InputState].row(j);
			RowVectorXld Enc_Back_C_j_l;
			if (j == 0) {
				Enc_Back_C_j_l = RowVectorXld::Zero(this->encoder_->Backward.statesForgrads.c[Number_InputState].row(j).cols());
			}
			else {
				Enc_Back_C_j_l = this->encoder_->Backward.statesForgrads.c[Number_InputState].row(j - 1);
			}

			RowVectorXld dEnc_Back_O_j = dH_Back_j.array() * ActivationFunctions::Tanh(Enc_Back_C_j).array() * Enc_Back_O_j.array() * (MatrixXld::Constant(Enc_Back_O_j.rows(), Enc_Back_O_j.cols(), 1) - Enc_Back_O_j).array();
			RowVectorXld dEnc_Back_C_j = dH_Back_j.array() * Enc_Back_O_j.array() *
				(MatrixXld::Constant(Enc_Back_C_j.rows(), Enc_Back_C_j.cols(), 1).array() - ActivationFunctions::Tanh(Enc_Back_C_j).array() * ActivationFunctions::Tanh(Enc_Back_C_j).array()) +
				Enc_Back__dC_j.array();
			RowVectorXld dEnc_Back_Ccond_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * (MatrixXld::Constant(Enc_Back_Ccond_j.rows(), Enc_Back_Ccond_j.cols(), 1).array() - Enc_Back_Ccond_j.array() * Enc_Back_Ccond_j.array());
			RowVectorXld dEnc_Back_I_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * Enc_Back_Ccond_j.array() * (MatrixXld::Constant(Enc_Back_I_j.rows(), Enc_Back_I_j.cols(), 1) - Enc_Back_I_j).array();
			RowVectorXld dEnc_Back_F_j = dEnc_Back_C_j.array() * Enc_Back_C_j_l.array() * Enc_Back_F_j.array() * (MatrixXld::Constant(Enc_Back_F_j.rows(), Enc_Back_F_j.cols(), 1) - Enc_Back_F_j).array();

			RowVectorXld dEnc_Back_Gates_j(4 * this->encoder_->Common_Hidden_size);
			dEnc_Back_Gates_j << dEnc_Back_F_j, dEnc_Back_I_j, dEnc_Back_Ccond_j, dEnc_Back_O_j;

			MatrixXld DW_Enc_Back_j = this->encoder_->Common_Input_states[Number_InputState].row(t).transpose() * dEnc_Back_Gates_j;
			MatrixXld DU_Enc_Back_j;
			if (t == 0) {
				DU_Enc_Back_j = MatrixXld::Zero(this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
			}
			else {
				DU_Enc_Back_j = this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Back_Gates_j;
			}
			VectorXld DB_Enc_Back_j = dEnc_Back_Gates_j;

			Enc_Back__dC_j = dEnc_Back_C_j.array() * Enc_Back_F_j.array();
			MatrixXld U_enc_b(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);////////////
			U_enc_b << this->encoder_->Backward.U_F, this->encoder_->Backward.U_I, this->encoder_->Backward.U_C, this->encoder_->Backward.U_O;
			Enc_Back__dH_j = U_enc_b.transpose() * dEnc_Back_Gates_j;

			grads.dW_f_back_enc += std::move(DW_Enc_Back_j.leftCols(this->encoder_->Common_Hidden_size));
			grads.dW_i_back_enc += std::move(DW_Enc_Back_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dW_ccond_back_enc += std::move(DW_Enc_Back_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dW_o_back_enc += std::move(DW_Enc_Back_j.rightCols(this->encoder_->Common_Hidden_size));

			grads.dU_f_back_enc += std::move(DU_Enc_Back_j.leftCols(this->encoder_->Common_Hidden_size));
			grads.dU_i_back_enc += std::move(DU_Enc_Back_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dU_ccond_back_enc += std::move(DU_Enc_Back_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dU_o_back_enc += std::move(DU_Enc_Back_j.rightCols(this->encoder_->Common_Hidden_size));

			grads.dB_f_back_enc += std::move(DB_Enc_Back_j.leftCols(this->encoder_->Common_Hidden_size));
			grads.dB_i_back_enc += std::move(DB_Enc_Back_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dB_ccond_back_enc += std::move(DB_Enc_Back_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size));
			grads.dB_o_back_enc += std::move(DB_Enc_Back_j.rightCols(this->encoder_->Common_Hidden_size));
		}


		grads.dW_out += std::move(DW_out_t);
		grads.dB_out += std::move(DB_out_t);

		grads.dW_gamma_layernorm += std::move(DGamma_t);
		grads.dB_beta_layernorm += std::move(DBeta_t);

		grads.dW_f_dec += std::move(DW_dec_t.leftCols(this->decoder_->Hidden_size));
		grads.dW_i_dec += std::move(DW_dec_t.middleCols(this->decoder_->Hidden_size, this->decoder_->Hidden_size));
		grads.dW_ccond_dec += std::move(DW_dec_t.middleCols(2 * this->decoder_->Hidden_size, this->decoder_->Hidden_size));
		grads.dW_o_dec += std::move(DW_dec_t.rightCols(this->decoder_->Hidden_size));

		grads.dU_f_dec += std::move(DU_dec_t.leftCols(this->decoder_->Hidden_size));
		grads.dU_i_dec += std::move(DU_dec_t.middleCols(this->decoder_->Hidden_size, this->decoder_->Hidden_size));
		grads.dU_ccond_dec += std::move(DU_dec_t.middleCols(2 * this->decoder_->Hidden_size, this->decoder_->Hidden_size));
		grads.dU_o_dec += std::move(DU_dec_t.rightCols(this->decoder_->Hidden_size));

		grads.dB_f_dec += std::move(DB_dec_t.leftCols(this->decoder_->Hidden_size));
		grads.dB_i_dec += std::move(DB_dec_t.middleCols(this->decoder_->Hidden_size, this->decoder_->Hidden_size));
		grads.dB_ccond_dec += std::move(DB_dec_t.middleCols(2 * this->decoder_->Hidden_size, this->decoder_->Hidden_size));
		grads.dB_o_dec += std::move(DB_dec_t.rightCols(this->decoder_->Hidden_size));
	}
}

Seq2SeqWithAttention_ForTrain::grads_Seq2SeqWithAttention Seq2SeqWithAttention_ForTrain::BackwardWithLogging(size_t Number_InputState, MatrixXld Y_True) {
	auto check_nan_inf = [](const MatrixXld& m, const std::string& name) {
		if (!m.allFinite()) {
			std::cerr << "[ERROR] NaN or Inf detected in: " << name << "\n";
		}
		};
	
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
			grads.dW_ccond_dec.conservativeResize(H, X), grads.dU_ccond_dec.conservativeResize(H, H), grads.dB_ccond_dec.conservativeResize(1, H),
			grads.dW_o_dec.conservativeResize(H, X), grads.dU_o_dec.conservativeResize(H, H), grads.dB_o_dec.conservativeResize(1, H);

		grads.dW_f_forw_enc.conservativeResize(HE, EE), grads.dU_f_forw_enc.conservativeResize(HE, HE), grads.dB_f_forw_enc.conservativeResize(1, HE),
			grads.dW_i_forw_enc.conservativeResize(HE, EE), grads.dU_i_forw_enc.conservativeResize(HE, HE), grads.dB_i_forw_enc.conservativeResize(1, HE),
			grads.dW_ccond_forw_enc.conservativeResize(HE, EE), grads.dU_ccond_forw_enc.conservativeResize(HE, HE), grads.dB_ccond_forw_enc.conservativeResize(1, HE),
			grads.dW_o_forw_enc.conservativeResize(HE, EE), grads.dU_o_forw_enc.conservativeResize(HE, HE), grads.dB_o_forw_enc.conservativeResize(1, HE);

		grads.dW_f_back_enc.conservativeResize(HE, EE), grads.dU_f_back_enc.conservativeResize(HE, HE), grads.dB_f_back_enc.conservativeResize(1, HE),
			grads.dW_i_back_enc.conservativeResize(HE, EE), grads.dU_i_back_enc.conservativeResize(HE, HE), grads.dB_i_back_enc.conservativeResize(1, HE),
			grads.dW_ccond_back_enc.conservativeResize(HE, EE), grads.dU_ccond_back_enc.conservativeResize(HE, HE), grads.dB_ccond_back_enc.conservativeResize(1, HE),
			grads.dW_o_back_enc.conservativeResize(HE, EE), grads.dU_o_back_enc.conservativeResize(HE, HE), grads.dB_o_back_enc.conservativeResize(1, HE);
	}

	Eigen::Index T = std::min(this->GetOutputs()[Number_InputState].rows(), Y_True.rows());
	Eigen::Index N = this->encoder_->Common_Input_states[Number_InputState].rows();

	RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	RowVectorXld _dS_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	for (Eigen::Index t = T; t >= 0; t--) {
		RowVectorXld dY_t = Y_True.row(t) - this->GetOutputs()[Number_InputState].row(t); //Y_true_t - Y_t
		MatrixXld DW_out_t = dY_t * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).transpose();
		RowVectorXld dp__t = this->decoder_->W_output.transpose() * dY_t;
		RowVectorXld DB_out_t = dY_t;

		RowVectorXld dS_t = dp__t.leftCols(this->decoder_->Hidden_size) + _dS_t;
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

		RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (MatrixXld::Constant(O_t.rows(), O_t.cols(), 1) - O_t).array();
		RowVectorXld dC_t = dS_t.array() * O_t.array() *
			(MatrixXld::Constant((C_t).rows(), (C_t).cols(), 1).array() - (ActivationFunctions::Tanh(C_t).array() * ActivationFunctions::Tanh(C_t).array())) +
			_dC_t.array();
		RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (MatrixXld::Constant(Ccond_t.rows(), Ccond_t.cols(), 1).array() - (Ccond_t.array() * Ccond_t.array()));
		RowVectorXld dI_t = dC_t.array() * I_t.array() * Ccond_t.array() * (MatrixXld::Constant(I_t.rows(), I_t.cols(), 1) - I_t).array();
		RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (MatrixXld::Constant(F_t.rows(), F_t.cols(), 1) - F_t).array();

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

		_dC_t = dC_t.array() * F_t.array();
		MatrixXld U(this->decoder_->Hidden_size, 4 * this->decoder_->Hidden_size);
		U << this->decoder_->U_F, this->decoder_->U_I, this->decoder_->U_C, this->decoder_->U_O;
		_dS_t = U.transpose() * dGates_t;

		std::vector<MatrixXld> _dH_Back;
		RowVectorXld Enc_Forw__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
		RowVectorXld Enc_Forw__dH_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);

		check_nan_inf(dY_t, "dY_t_" + t);
		check_nan_inf(dp__t, "dp__t_" + t);
		check_nan_inf(dS_t, "dS_t_" + t);
		check_nan_inf(dContext_t, "dContext_t_" + t);
		check_nan_inf(dF_t, "dF_t_" + t);
		check_nan_inf(dI_t, "dI_t_" + t);
		check_nan_inf(dC_t, "dC_t_" + t);
		check_nan_inf(dO_t, "dO_t_" + t);
		check_nan_inf(dCcond_t, "dCcond_t_" + t);

		check_nan_inf(DW_out_t, "DW_out_t_" + t);
		check_nan_inf(DB_out_t, "DB_out_t_" + t);
		check_nan_inf(DGamma_t, "DGamma_t_" + t);
		check_nan_inf(DBeta_t, "DBeta_t_" + t);
		check_nan_inf(DW_dec_t, "DW_dec_t_" + t);
		check_nan_inf(DU_dec_t, "DU_dec_t_" + t);
		check_nan_inf(DB_dec_t, "DB_dec_t_" + t);


		for (Eigen::Index j = N - 1; j >= 0; j--) {
			RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
			RowVectorXld s_t_1;
			if (t > 0) {
				s_t_1 = this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1);
			}
			else {
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
			dH_forw_j += Enc_Forw__dH_j;
			RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);

			_dH_Back.push_back(dH_back_j);


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

			RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (MatrixXld::Constant(Enc_Forw_O_j.rows(), Enc_Forw_O_j.cols(), 1) - Enc_Forw_O_j).array();
			RowVectorXld dEnc_Forw_C_j = dH_forw_j.array() * Enc_Forw_O_j.array() *
				(MatrixXld::Constant(Enc_Forw_C_j.rows(), Enc_Forw_C_j.cols(), 1).array() - ActivationFunctions::Tanh(Enc_Forw_C_j).array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array()) +
				Enc_Forw__dC_j.array();
			RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (MatrixXld::Constant(Enc_Forw_Ccond_j.rows(), Enc_Forw_Ccond_j.cols(), 1).array() - Enc_Forw_Ccond_j.array() * Enc_Forw_Ccond_j.array());
			RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * Enc_Forw_Ccond_j.array() * (MatrixXld::Constant(Enc_Forw_I_j.rows(), Enc_Forw_I_j.cols(), 1) - Enc_Forw_I_j).array();
			RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (MatrixXld::Constant(Enc_Forw_F_j.rows(), Enc_Forw_F_j.cols(), 1) - Enc_Forw_F_j).array();

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

			Enc_Forw__dC_j = dEnc_Forw_C_j.array() * Enc_Forw_F_j.array();
			MatrixXld U_enc_f(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);
			U_enc_f << this->encoder_->Forward.U_F, this->encoder_->Forward.U_I, this->encoder_->Forward.U_C, this->encoder_->Forward.U_O;
			Enc_Forw__dH_j = U_enc_f.transpose() * dEnc_Forw_Gates_j;

			grads.dV_a_attention += DV_att_tj;
			grads.dW_e_attention += DW_att_enc_tj;
			grads.dW_d_attention += DW_att_dec_tj;

			grads.dW_f_forw_enc += DW_Enc_Forw_j.leftCols(this->encoder_->Common_Hidden_size);
			grads.dW_i_forw_enc += DW_Enc_Forw_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dW_ccond_forw_enc += DW_Enc_Forw_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dW_o_forw_enc += DW_Enc_Forw_j.rightCols(this->encoder_->Common_Hidden_size);

			grads.dU_f_forw_enc += DU_Enc_Forw_j.leftCols(this->encoder_->Common_Hidden_size);
			grads.dU_i_forw_enc += DU_Enc_Forw_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dU_ccond_forw_enc += DU_Enc_Forw_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dU_o_forw_enc += DU_Enc_Forw_j.rightCols(this->encoder_->Common_Hidden_size);

			grads.dB_f_forw_enc += DB_Enc_Forw_j.leftCols(this->encoder_->Common_Hidden_size);
			grads.dB_i_forw_enc += DB_Enc_Forw_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dB_ccond_forw_enc += DB_Enc_Forw_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dB_o_forw_enc += DB_Enc_Forw_j.rightCols(this->encoder_->Common_Hidden_size);

			
			check_nan_inf(dU_tj, "dU_tj_" + t + j);
			check_nan_inf(dPreact_tj, "dPreact_tj_" + t + j);
			check_nan_inf(dH_forw_j, "dH_forw_j_" + j);
			check_nan_inf(dH_back_j, "dH_back_j_" + j);
			check_nan_inf(dEnc_Forw_F_j, "dEnc_Forw_F_j_"  + j);
			check_nan_inf(dEnc_Forw_I_j, "dEnc_Forw_I_j_"  + j);
			check_nan_inf(dEnc_Forw_C_j, "dEnc_Forw_C_j_"  + j);
			check_nan_inf(dEnc_Forw_Ccond_j, "dEnc_Forw_Ccond_j_"  + j);
			check_nan_inf(dEnc_Forw_O_j, "dEnc_Forw_O_j_"  + j);

			check_nan_inf(DW_att_enc_tj, "DW_att_enc_tj_" + t + j);
			check_nan_inf(DW_att_dec_tj, "DW_att_dec_tj_" + t + j);
			check_nan_inf(DV_att_tj, "DV_att_tj_" + t + j);
			check_nan_inf(DW_Enc_Forw_j, "DW_Enc_Forw_j_" + j);
			check_nan_inf(DU_Enc_Forw_j, "DU_Enc_Forw_j_" + j);
			check_nan_inf(DB_Enc_Forw_j, "DB_Enc_Forw_j_" + j);
		}

		RowVectorXld Enc_Back__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
		RowVectorXld Enc_Back__dH_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);


		for (Eigen::Index j = 0; j < N; j++) {
			auto dH_Back_j = _dH_Back[j];
			dH_Back_j += Enc_Back__dH_j;
			RowVectorXld Enc_Back_F_j = this->encoder_->Backward.statesForgrads.f[Number_InputState].row(j);
			RowVectorXld Enc_Back_I_j = this->encoder_->Backward.statesForgrads.i[Number_InputState].row(j);
			RowVectorXld Enc_Back_Ccond_j = this->encoder_->Backward.statesForgrads.ccond[Number_InputState].row(j);
			RowVectorXld Enc_Back_O_j = this->encoder_->Backward.statesForgrads.o[Number_InputState].row(j);
			RowVectorXld Enc_Back_C_j = this->encoder_->Backward.statesForgrads.c[Number_InputState].row(j);
			RowVectorXld Enc_Back_C_j_l;
			if (j == 0) {
				Enc_Back_C_j_l = RowVectorXld::Zero(this->encoder_->Backward.statesForgrads.c[Number_InputState].row(j).cols());
			}
			else {
				Enc_Back_C_j_l = this->encoder_->Backward.statesForgrads.c[Number_InputState].row(j - 1);
			}

			RowVectorXld dEnc_Back_O_j = dH_Back_j.array() * ActivationFunctions::Tanh(Enc_Back_C_j).array() * Enc_Back_O_j.array() * (MatrixXld::Constant(Enc_Back_O_j.rows(), Enc_Back_O_j.cols(), 1) - Enc_Back_O_j).array();
			RowVectorXld dEnc_Back_C_j = dH_Back_j.array() * Enc_Back_O_j.array() *
				(MatrixXld::Constant(Enc_Back_C_j.rows(), Enc_Back_C_j.cols(), 1).array() - ActivationFunctions::Tanh(Enc_Back_C_j).array() * ActivationFunctions::Tanh(Enc_Back_C_j).array()) +
				Enc_Back__dC_j.array();
			RowVectorXld dEnc_Back_Ccond_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * (MatrixXld::Constant(Enc_Back_Ccond_j.rows(), Enc_Back_Ccond_j.cols(), 1).array() - Enc_Back_Ccond_j.array() * Enc_Back_Ccond_j.array());
			RowVectorXld dEnc_Back_I_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * Enc_Back_Ccond_j.array() * (MatrixXld::Constant(Enc_Back_I_j.rows(), Enc_Back_I_j.cols(), 1) - Enc_Back_I_j).array();
			RowVectorXld dEnc_Back_F_j = dEnc_Back_C_j.array() * Enc_Back_C_j_l.array() * Enc_Back_F_j.array() * (MatrixXld::Constant(Enc_Back_F_j.rows(), Enc_Back_F_j.cols(), 1) - Enc_Back_F_j).array();

			RowVectorXld dEnc_Back_Gates_j(4 * this->encoder_->Common_Hidden_size);
			dEnc_Back_Gates_j << dEnc_Back_F_j, dEnc_Back_I_j, dEnc_Back_Ccond_j, dEnc_Back_O_j;

			MatrixXld DW_Enc_Back_j = this->encoder_->Common_Input_states[Number_InputState].row(t).transpose() * dEnc_Back_Gates_j;
			MatrixXld DU_Enc_Back_j;
			if (t == 0) {
				DU_Enc_Back_j = MatrixXld::Zero(this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
			}
			else {
				DU_Enc_Back_j = this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Back_Gates_j;
			}
			VectorXld DB_Enc_Back_j = dEnc_Back_Gates_j;

			Enc_Back__dC_j = dEnc_Back_C_j.array() * Enc_Back_F_j.array();
			MatrixXld U_enc_b(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);////////////
			U_enc_b << this->encoder_->Backward.U_F, this->encoder_->Backward.U_I, this->encoder_->Backward.U_C, this->encoder_->Backward.U_O;
			Enc_Back__dH_j = U_enc_b.transpose() * dEnc_Back_Gates_j;

			grads.dW_f_back_enc += DW_Enc_Back_j.leftCols(this->encoder_->Common_Hidden_size);
			grads.dW_i_back_enc += DW_Enc_Back_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dW_ccond_back_enc += DW_Enc_Back_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dW_o_back_enc += DW_Enc_Back_j.rightCols(this->encoder_->Common_Hidden_size);

			grads.dU_f_back_enc += DU_Enc_Back_j.leftCols(this->encoder_->Common_Hidden_size);
			grads.dU_i_back_enc += DU_Enc_Back_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dU_ccond_back_enc += DU_Enc_Back_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dU_o_back_enc += DU_Enc_Back_j.rightCols(this->encoder_->Common_Hidden_size);

			grads.dB_f_back_enc += DB_Enc_Back_j.leftCols(this->encoder_->Common_Hidden_size);
			grads.dB_i_back_enc += DB_Enc_Back_j.middleCols(this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dB_ccond_back_enc += DB_Enc_Back_j.middleCols(2 * this->encoder_->Common_Hidden_size, this->encoder_->Common_Hidden_size);
			grads.dB_o_back_enc += DB_Enc_Back_j.rightCols(this->encoder_->Common_Hidden_size);

			
			check_nan_inf(dEnc_Back_F_j, "dEnc_Back_F_j_" + j);
			check_nan_inf(dEnc_Back_I_j, "dEnc_Back_I_j_" + j);
			check_nan_inf(dEnc_Back_C_j, "dEnc_Back_C_j_" + j);
			check_nan_inf(dEnc_Back_Ccond_j, "dEnc_Back_Ccond_j_" + j);
			check_nan_inf(dEnc_Back_O_j, "dEnc_Back_O_j_" + j);

			check_nan_inf(DW_Enc_Back_j, "DW_Enc_Back_j_" + j);
			check_nan_inf(DU_Enc_Back_j, "DU_Enc_Back_j_" + j);
			check_nan_inf(DB_Enc_Back_j, "DB_Enc_Back_j_" + j);
		}


		grads.dW_out += DW_out_t;
		grads.dB_out += DB_out_t;

		grads.dW_gamma_layernorm += DGamma_t;
		grads.dB_beta_layernorm += DBeta_t;

		grads.dW_f_dec += DW_dec_t.leftCols(this->decoder_->Hidden_size);
		grads.dW_i_dec += DW_dec_t.middleCols(this->decoder_->Hidden_size, this->decoder_->Hidden_size);
		grads.dW_ccond_dec += DW_dec_t.middleCols(2 * this->decoder_->Hidden_size, this->decoder_->Hidden_size);
		grads.dW_o_dec += DW_dec_t.rightCols(this->decoder_->Hidden_size);

		grads.dU_f_dec += DU_dec_t.leftCols(this->decoder_->Hidden_size);
		grads.dU_i_dec += DU_dec_t.middleCols(this->decoder_->Hidden_size, this->decoder_->Hidden_size);
		grads.dU_ccond_dec += DU_dec_t.middleCols(2 * this->decoder_->Hidden_size, this->decoder_->Hidden_size);
		grads.dU_o_dec += DU_dec_t.rightCols(this->decoder_->Hidden_size);

		grads.dB_f_dec += DB_dec_t.leftCols(this->decoder_->Hidden_size);
		grads.dB_i_dec += DB_dec_t.middleCols(this->decoder_->Hidden_size, this->decoder_->Hidden_size);
		grads.dB_ccond_dec += DB_dec_t.middleCols(2 * this->decoder_->Hidden_size, this->decoder_->Hidden_size);
		grads.dB_o_dec += DB_dec_t.rightCols(this->decoder_->Hidden_size);
	}
}

void Seq2SeqWithAttention_ForTrain::UpdateAdamOpt
(
	const std::vector<std::vector<MatrixXld>>& Target_input_output, /*std::vector<MatrixXld> Target_output,*/
	size_t epochs, size_t optima_steps, size_t batch_size,
	long double learning_rate, long double epsilon,
	long double beta1, long double beta2
)
{
	auto clip_by_global_norm = [](auto& grads, long double clip_value) {
		long double total_sq_norm = 0.0;

		// -------- Decoder --------
		total_sq_norm += grads.dW_out.squaredNorm();
		total_sq_norm += grads.dB_out.squaredNorm();

		total_sq_norm += grads.dW_f_dec.squaredNorm();
		total_sq_norm += grads.dU_f_dec.squaredNorm();
		total_sq_norm += grads.dB_f_dec.squaredNorm();

		total_sq_norm += grads.dW_i_dec.squaredNorm();
		total_sq_norm += grads.dU_i_dec.squaredNorm();
		total_sq_norm += grads.dB_i_dec.squaredNorm();

		total_sq_norm += grads.dW_ccond_dec.squaredNorm();
		total_sq_norm += grads.dU_ccond_dec.squaredNorm();
		total_sq_norm += grads.dB_ccond_dec.squaredNorm();

		total_sq_norm += grads.dW_o_dec.squaredNorm();
		total_sq_norm += grads.dU_o_dec.squaredNorm();
		total_sq_norm += grads.dB_o_dec.squaredNorm();

		// -------- LayerNorm --------
		total_sq_norm += grads.dW_gamma_layernorm.squaredNorm();
		total_sq_norm += grads.dB_beta_layernorm.squaredNorm();

		// -------- Attention --------
		total_sq_norm += grads.dV_a_attention.squaredNorm();
		total_sq_norm += grads.dW_e_attention.squaredNorm();
		total_sq_norm += grads.dW_d_attention.squaredNorm();

		// -------- Forward Encoder --------
		total_sq_norm += grads.dW_f_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_f_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_f_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_i_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_i_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_i_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_ccond_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_ccond_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_ccond_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_o_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_o_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_o_forw_enc.squaredNorm();

		// -------- Backward Encoder --------
		total_sq_norm += grads.dW_f_back_enc.squaredNorm();
		total_sq_norm += grads.dU_f_back_enc.squaredNorm();
		total_sq_norm += grads.dB_f_back_enc.squaredNorm();

		total_sq_norm += grads.dW_i_back_enc.squaredNorm();
		total_sq_norm += grads.dU_i_back_enc.squaredNorm();
		total_sq_norm += grads.dB_i_back_enc.squaredNorm();

		total_sq_norm += grads.dW_ccond_back_enc.squaredNorm();
		total_sq_norm += grads.dU_ccond_back_enc.squaredNorm();
		total_sq_norm += grads.dB_ccond_back_enc.squaredNorm();

		total_sq_norm += grads.dW_o_back_enc.squaredNorm();
		total_sq_norm += grads.dU_o_back_enc.squaredNorm();
		total_sq_norm += grads.dB_o_back_enc.squaredNorm();

		long double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		if (global_norm > clip_value) {
			long double scale = clip_value / global_norm;

			// -------- Decoder --------
			grads.dW_out *= scale;
			grads.dB_out *= scale;

			grads.dW_f_dec *= scale;
			grads.dU_f_dec *= scale;
			grads.dB_f_dec *= scale;

			grads.dW_i_dec *= scale;
			grads.dU_i_dec *= scale;
			grads.dB_i_dec *= scale;

			grads.dW_ccond_dec *= scale;
			grads.dU_ccond_dec *= scale;
			grads.dB_ccond_dec *= scale;

			grads.dW_o_dec *= scale;
			grads.dU_o_dec *= scale;
			grads.dB_o_dec *= scale;

			// -------- LayerNorm --------
			grads.dW_gamma_layernorm *= scale;
			grads.dB_beta_layernorm *= scale;

			// -------- Attention --------
			grads.dV_a_attention *= scale;
			grads.dW_e_attention *= scale;
			grads.dW_d_attention *= scale;

			// -------- Forward Encoder --------
			grads.dW_f_forw_enc *= scale;
			grads.dU_f_forw_enc *= scale;
			grads.dB_f_forw_enc *= scale;

			grads.dW_i_forw_enc *= scale;
			grads.dU_i_forw_enc *= scale;
			grads.dB_i_forw_enc *= scale;

			grads.dW_ccond_forw_enc *= scale;
			grads.dU_ccond_forw_enc *= scale;
			grads.dB_ccond_forw_enc *= scale;
			grads.dW_o_forw_enc *= scale;
			grads.dU_o_forw_enc *= scale;
			grads.dB_o_forw_enc *= scale;

			// -------- Backward Encoder --------
			grads.dW_f_back_enc *= scale;
			grads.dU_f_back_enc *= scale;
			grads.dB_f_back_enc *= scale;

			grads.dW_i_back_enc *= scale;
			grads.dU_i_back_enc *= scale;
			grads.dB_i_back_enc *= scale;

			grads.dW_ccond_back_enc *= scale;
			grads.dU_ccond_back_enc *= scale;
			grads.dB_ccond_back_enc *= scale;

			grads.dW_o_back_enc *= scale;
			grads.dU_o_back_enc *= scale;
			grads.dB_o_back_enc *= scale;
		}
};

	std::vector<std::vector<MatrixXld>> shuffle_target = Target_input_output;
	long double notceil_batch_steps_ = (long double)shuffle_target.size() / batch_size;
	size_t batch_steps_ = (size_t)std::ceil(notceil_batch_steps_);
	for (size_t epoch_ = 0; epoch_ < epochs; epoch_++) {
		std::random_device rd;
		std::shuffle(shuffle_target.begin(), shuffle_target.end(), std::mt19937(rd()));

		grads_Seq2SeqWithAttention grads;

		Inference(shuffle_target[0]);

		for (size_t batch_step = 0; batch_step < batch_steps_; batch_step++) {
			for (size_t i = batch_step * batch_size; i < (batch_step + 1) * batch_size && i < shuffle_target.size(); i++) {
				grads += std::move(Backward(i, shuffle_target[1][i]));
			}
			grads /= (batch_step == batch_steps_) ? batch_size * (notceil_batch_steps_ - (int)notceil_batch_steps_) : batch_size;

			clip_by_global_norm(grads, 200L);

			MatrixXld M_W_out; MatrixXld M_B_out;

			MatrixXld M_W_gamma_layernorm; MatrixXld M_B_beta_layernorm;

			MatrixXld M_V_a_attention, M_W_e_attention, M_W_d_attention;

			MatrixXld M_W_f_dec, M_U_f_dec; MatrixXld M_B_f_dec;
			MatrixXld M_W_i_dec, M_U_i_dec; MatrixXld M_B_i_dec;
			MatrixXld M_W_ccond_dec, M_U_ccond_dec; MatrixXld M_B_ccond_dec;
			MatrixXld M_W_o_dec, M_U_o_dec; MatrixXld M_B_o_dec;

			MatrixXld M_W_f_forw_enc, M_U_f_forw_enc; MatrixXld M_B_f_forw_enc;
			MatrixXld M_W_i_forw_enc, M_U_i_forw_enc; MatrixXld M_B_i_forw_enc;
			MatrixXld M_W_ccond_forw_enc, M_U_ccond_forw_enc; MatrixXld M_B_ccond_forw_enc;
			MatrixXld M_W_o_forw_enc, M_U_o_forw_enc; MatrixXld M_B_o_forw_enc;

			MatrixXld M_W_f_back_enc, M_U_f_back_enc; MatrixXld M_B_f_back_enc;
			MatrixXld M_W_i_back_enc, M_U_i_back_enc; MatrixXld M_B_i_back_enc;
			MatrixXld M_W_ccond_back_enc, M_U_ccond_back_enc; MatrixXld M_B_ccond_back_enc;
			MatrixXld M_W_o_back_enc, M_U_o_back_enc; MatrixXld M_B_o_back_enc;


			MatrixXld V_W_out; MatrixXld V_B_out;

			MatrixXld V_W_gamma_layernorm; MatrixXld V_B_beta_layernorm;

			MatrixXld V_V_a_attention, V_W_e_attention, V_W_d_attention;

			MatrixXld V_W_f_dec, V_U_f_dec; MatrixXld V_B_f_dec;
			MatrixXld V_W_i_dec, V_U_i_dec; MatrixXld V_B_i_dec;
			MatrixXld V_W_ccond_dec, V_U_ccond_dec; MatrixXld V_B_ccond_dec;
			MatrixXld V_W_o_dec, V_U_o_dec; MatrixXld V_B_o_dec;

			MatrixXld V_W_f_forw_enc, V_U_f_forw_enc; MatrixXld V_B_f_forw_enc;
			MatrixXld V_W_i_forw_enc, V_U_i_forw_enc; MatrixXld V_B_i_forw_enc;
			MatrixXld V_W_ccond_forw_enc, V_U_ccond_forw_enc; MatrixXld V_B_ccond_forw_enc;
			MatrixXld V_W_o_forw_enc, V_U_o_forw_enc; MatrixXld V_B_o_forw_enc;

			MatrixXld V_W_f_back_enc, V_U_f_back_enc; MatrixXld V_B_f_back_enc;
			MatrixXld V_W_i_back_enc, V_U_i_back_enc; MatrixXld V_B_i_back_enc;
			MatrixXld V_W_ccond_back_enc, V_U_ccond_back_enc; MatrixXld V_B_ccond_back_enc;
			MatrixXld V_W_o_back_enc, V_U_o_back_enc; MatrixXld V_B_o_back_enc;

			for (size_t t_ = 0; t_ < optima_steps; t_++) {
				{
					M_W_out = beta1 * M_W_out + (1 - beta1) * grads.dW_out;
					M_B_out = beta1 * M_B_out + (1 - beta1) * grads.dB_out;
					//
					M_W_gamma_layernorm = beta1 * M_W_gamma_layernorm + (1 - beta1) * grads.dW_gamma_layernorm;
					M_B_beta_layernorm = beta1 * M_B_beta_layernorm + (1 - beta1) * grads.dB_beta_layernorm;
					//
					M_V_a_attention = beta1 * M_V_a_attention + (1 - beta1) * grads.dV_a_attention;
					M_W_e_attention = beta1 * M_W_e_attention + (1 - beta1) * grads.dW_e_attention;
					M_W_d_attention = beta1 * M_W_d_attention + (1 - beta1) * grads.dW_d_attention;
					//
					M_W_f_dec = beta1 * M_W_f_dec + (1 - beta1) * grads.dW_f_dec;
					M_U_f_dec = beta1 * M_U_f_dec + (1 - beta1) * grads.dU_f_dec;
					M_B_f_dec = beta1 * M_B_f_dec + (1 - beta1) * grads.dB_f_dec;

					M_W_i_dec = beta1 * M_W_i_dec + (1 - beta1) * grads.dW_i_dec;
					M_U_i_dec = beta1 * M_U_i_dec + (1 - beta1) * grads.dU_i_dec;
					M_B_i_dec = beta1 * M_B_i_dec + (1 - beta1) * grads.dB_i_dec;

					M_W_ccond_dec = beta1 * M_W_ccond_dec + (1 - beta1) * grads.dW_ccond_dec;
					M_U_ccond_dec = beta1 * M_U_ccond_dec + (1 - beta1) * grads.dU_ccond_dec;
					M_B_ccond_dec = beta1 * M_B_ccond_dec + (1 - beta1) * grads.dB_ccond_dec;

					M_W_o_dec = beta1 * M_W_o_dec + (1 - beta1) * grads.dW_o_dec;
					M_U_o_dec = beta1 * M_U_o_dec + (1 - beta1) * grads.dU_o_dec;
					M_B_o_dec = beta1 * M_B_o_dec + (1 - beta1) * grads.dB_o_dec;
					//
					M_W_f_forw_enc = beta1 * M_W_f_forw_enc + (1 - beta1) * grads.dW_f_forw_enc;
					M_U_f_forw_enc = beta1 * M_U_f_forw_enc + (1 - beta1) * grads.dU_f_forw_enc;
					M_B_f_forw_enc = beta1 * M_B_f_forw_enc + (1 - beta1) * grads.dB_f_forw_enc;

					M_W_i_forw_enc = beta1 * M_W_i_forw_enc + (1 - beta1) * grads.dW_i_forw_enc;
					M_U_i_forw_enc = beta1 * M_U_i_forw_enc + (1 - beta1) * grads.dU_i_forw_enc;
					M_B_i_forw_enc = beta1 * M_B_i_forw_enc + (1 - beta1) * grads.dB_i_forw_enc;

					M_W_ccond_forw_enc = beta1 * M_W_ccond_forw_enc + (1 - beta1) * grads.dW_ccond_forw_enc;
					M_U_ccond_forw_enc = beta1 * M_U_ccond_forw_enc + (1 - beta1) * grads.dU_ccond_forw_enc;
					M_B_ccond_forw_enc = beta1 * M_B_ccond_forw_enc + (1 - beta1) * grads.dB_ccond_forw_enc;

					M_W_o_forw_enc = beta1 * M_W_o_forw_enc + (1 - beta1) * grads.dW_o_forw_enc;
					M_U_o_forw_enc = beta1 * M_U_o_forw_enc + (1 - beta1) * grads.dU_o_forw_enc;
					M_B_o_forw_enc = beta1 * M_B_o_forw_enc + (1 - beta1) * grads.dB_o_forw_enc;
					//
					M_W_f_back_enc = beta1 * M_W_f_back_enc + (1 - beta1) * grads.dW_f_back_enc;
					M_U_f_back_enc = beta1 * M_U_f_back_enc + (1 - beta1) * grads.dU_f_back_enc;
					M_B_f_back_enc = beta1 * M_B_f_back_enc + (1 - beta1) * grads.dB_f_back_enc;

					M_W_i_back_enc = beta1 * M_W_i_back_enc + (1 - beta1) * grads.dW_i_back_enc;
					M_U_i_back_enc = beta1 * M_U_i_back_enc + (1 - beta1) * grads.dU_i_back_enc;
					M_B_i_back_enc = beta1 * M_B_i_back_enc + (1 - beta1) * grads.dB_i_back_enc;

					M_W_ccond_back_enc = beta1 * M_W_ccond_back_enc + (1 - beta1) * grads.dW_ccond_back_enc;
					M_U_ccond_back_enc = beta1 * M_U_ccond_back_enc + (1 - beta1) * grads.dU_ccond_back_enc;
					M_B_ccond_back_enc = beta1 * M_B_ccond_back_enc + (1 - beta1) * grads.dB_ccond_back_enc;

					M_W_o_back_enc = beta1 * M_W_o_back_enc + (1 - beta1) * grads.dW_o_back_enc;
					M_U_o_back_enc = beta1 * M_U_o_back_enc + (1 - beta1) * grads.dU_o_back_enc;
					M_B_o_back_enc = beta1 * M_B_o_back_enc + (1 - beta1) * grads.dB_o_back_enc;
					//
					//
					V_W_out = beta2 * V_W_out.array() + (1 - beta2) * grads.dW_out.array() * grads.dW_out.array();
					V_B_out = beta2 * V_B_out.array() + (1 - beta2) * grads.dB_out.array() * grads.dB_out.array();
					//
					V_W_gamma_layernorm = beta2 * V_W_gamma_layernorm.array() + (1 - beta2) * grads.dW_gamma_layernorm.array() * grads.dW_gamma_layernorm.array();
					V_B_beta_layernorm = beta2 * V_B_beta_layernorm.array() + (1 - beta2) * grads.dB_beta_layernorm.array() * grads.dB_beta_layernorm.array();
					//
					V_V_a_attention = beta2 * V_V_a_attention.array() + (1 - beta2) * grads.dV_a_attention.array() * grads.dV_a_attention.array();
					V_W_e_attention = beta2 * V_W_e_attention.array() + (1 - beta2) * grads.dW_e_attention.array() * grads.dW_e_attention.array();
					V_W_d_attention = beta2 * V_W_d_attention.array() + (1 - beta2) * grads.dW_d_attention.array() * grads.dW_d_attention.array();
					//
					V_W_f_dec = beta2 * V_W_f_dec.array() + (1 - beta2) * grads.dW_f_dec.array() * grads.dW_f_dec.array();
					V_U_f_dec = beta2 * V_U_f_dec.array() + (1 - beta2) * grads.dU_f_dec.array() * grads.dU_f_dec.array();
					V_B_f_dec = beta2 * V_B_f_dec.array() + (1 - beta2) * grads.dB_f_dec.array() * grads.dB_f_dec.array();

					V_W_i_dec = beta2 * V_W_i_dec.array() + (1 - beta2) * grads.dW_i_dec.array() * grads.dW_i_dec.array();
					V_U_i_dec = beta2 * V_U_i_dec.array() + (1 - beta2) * grads.dU_i_dec.array() * grads.dU_i_dec.array();
					V_B_i_dec = beta2 * V_B_i_dec.array() + (1 - beta2) * grads.dB_i_dec.array() * grads.dB_i_dec.array();

					V_W_ccond_dec = beta2 * V_W_ccond_dec.array() + (1 - beta2) * grads.dW_ccond_dec.array() * grads.dW_ccond_dec.array();
					V_U_ccond_dec = beta2 * V_U_ccond_dec.array() + (1 - beta2) * grads.dU_ccond_dec.array() * grads.dU_ccond_dec.array();
					V_B_ccond_dec = beta2 * V_B_ccond_dec.array() + (1 - beta2) * grads.dB_ccond_dec.array() * grads.dB_ccond_dec.array();

					V_W_o_dec = beta2 * V_W_o_dec.array() + (1 - beta2) * grads.dW_o_dec.array() * grads.dW_o_dec.array();
					V_U_o_dec = beta2 * V_U_o_dec.array() + (1 - beta2) * grads.dU_o_dec.array() * grads.dU_o_dec.array();
					V_B_o_dec = beta2 * V_B_o_dec.array() + (1 - beta2) * grads.dB_o_dec.array() * grads.dB_o_dec.array();
					//
					V_W_f_forw_enc = beta2 * V_W_f_forw_enc.array() + (1 - beta2) * grads.dW_f_forw_enc.array() * grads.dW_f_forw_enc.array();
					V_U_f_forw_enc = beta2 * V_U_f_forw_enc.array() + (1 - beta2) * grads.dU_f_forw_enc.array() * grads.dU_f_forw_enc.array();
					V_B_f_forw_enc = beta2 * V_B_f_forw_enc.array() + (1 - beta2) * grads.dB_f_forw_enc.array() * grads.dB_f_forw_enc.array();

					V_W_i_forw_enc = beta2 * V_W_i_forw_enc.array() + (1 - beta2) * grads.dW_i_forw_enc.array() * grads.dW_i_forw_enc.array();
					V_U_i_forw_enc = beta2 * V_U_i_forw_enc.array() + (1 - beta2) * grads.dU_i_forw_enc.array() * grads.dU_i_forw_enc.array();
					V_B_i_forw_enc = beta2 * V_B_i_forw_enc.array() + (1 - beta2) * grads.dB_i_forw_enc.array() * grads.dB_i_forw_enc.array();

					V_W_ccond_forw_enc = beta2 * V_W_ccond_forw_enc.array() + (1 - beta2) * grads.dW_ccond_forw_enc.array() * grads.dW_ccond_forw_enc.array();
					V_U_ccond_forw_enc = beta2 * V_U_ccond_forw_enc.array() + (1 - beta2) * grads.dU_ccond_forw_enc.array() * grads.dU_ccond_forw_enc.array();
					V_B_ccond_forw_enc = beta2 * V_B_ccond_forw_enc.array() + (1 - beta2) * grads.dB_ccond_forw_enc.array() * grads.dB_ccond_forw_enc.array();

					V_W_o_forw_enc = beta2 * V_W_o_forw_enc.array() + (1 - beta2) * grads.dW_o_forw_enc.array() * grads.dW_o_forw_enc.array();
					V_U_o_forw_enc = beta2 * V_U_o_forw_enc.array() + (1 - beta2) * grads.dU_o_forw_enc.array() * grads.dU_o_forw_enc.array();
					V_B_o_forw_enc = beta2 * V_B_o_forw_enc.array() + (1 - beta2) * grads.dB_o_forw_enc.array() * grads.dB_o_forw_enc.array();
					//
					V_W_f_back_enc = beta2 * V_W_f_back_enc.array() + (1 - beta2) * grads.dW_f_back_enc.array() * grads.dW_f_back_enc.array();
					V_U_f_back_enc = beta2 * V_U_f_back_enc.array() + (1 - beta2) * grads.dU_f_back_enc.array() * grads.dU_f_back_enc.array();
					V_B_f_back_enc = beta2 * V_B_f_back_enc.array() + (1 - beta2) * grads.dB_f_back_enc.array() * grads.dB_f_back_enc.array();

					V_W_i_back_enc = beta2 * V_W_i_back_enc.array() + (1 - beta2) * grads.dW_i_back_enc.array() * grads.dW_i_back_enc.array();
					V_U_i_back_enc = beta2 * V_U_i_back_enc.array() + (1 - beta2) * grads.dU_i_back_enc.array() * grads.dU_i_back_enc.array();
					V_B_i_back_enc = beta2 * V_B_i_back_enc.array() + (1 - beta2) * grads.dB_i_back_enc.array() * grads.dB_i_back_enc.array();

					V_W_ccond_back_enc = beta2 * V_W_ccond_back_enc.array() + (1 - beta2) * grads.dW_ccond_back_enc.array() * grads.dW_ccond_back_enc.array();
					V_U_ccond_back_enc = beta2 * V_U_ccond_back_enc.array() + (1 - beta2) * grads.dU_ccond_back_enc.array() * grads.dU_ccond_back_enc.array();
					V_B_ccond_back_enc = beta2 * V_B_ccond_back_enc.array() + (1 - beta2) * grads.dB_ccond_back_enc.array() * grads.dB_ccond_back_enc.array();

					V_W_o_back_enc = beta2 * V_W_o_back_enc.array() + (1 - beta2) * grads.dW_o_back_enc.array() * grads.dW_o_back_enc.array();
					V_U_o_back_enc = beta2 * V_U_o_back_enc.array() + (1 - beta2) * grads.dU_o_back_enc.array() * grads.dU_o_back_enc.array();
					V_B_o_back_enc = beta2 * V_B_o_back_enc.array() + (1 - beta2) * grads.dB_o_back_enc.array() * grads.dB_o_back_enc.array();
				}
				/////
				/////
				/////
				MatrixXld _M_W_out = M_W_out.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_out = M_B_out.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_W_gamma_layernorm = M_W_gamma_layernorm.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_beta_layernorm = M_B_beta_layernorm.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_V_a_attention = M_V_a_attention.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_W_e_attention = M_W_e_attention.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_W_d_attention = M_W_d_attention.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_W_f_dec = M_W_f_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_f_dec = M_U_f_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_f_dec = M_B_f_dec.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_i_dec = M_W_i_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_i_dec = M_U_i_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_i_dec = M_B_i_dec.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_ccond_dec = M_W_ccond_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_ccond_dec = M_U_ccond_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_ccond_dec = M_B_ccond_dec.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_o_dec = M_W_o_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_o_dec = M_U_o_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_o_dec = M_B_o_dec.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_W_f_forw_enc = M_W_f_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_f_forw_enc = M_U_f_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_f_forw_enc = M_B_f_forw_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_i_forw_enc = M_W_i_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_i_forw_enc = M_U_i_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_i_forw_enc = M_B_i_forw_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_ccond_forw_enc = M_W_ccond_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_ccond_forw_enc = M_U_ccond_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_ccond_forw_enc = M_B_ccond_forw_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_o_forw_enc = M_W_o_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_o_forw_enc = M_U_o_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_o_forw_enc = M_B_o_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				//				   
				MatrixXld _M_W_f_back_enc = M_W_f_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_f_back_enc = M_U_f_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_f_back_enc = M_B_f_back_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_i_back_enc = M_W_i_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_i_back_enc = M_U_i_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_i_back_enc = M_B_i_back_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_ccond_back_enc = M_W_ccond_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_ccond_back_enc = M_U_ccond_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_ccond_back_enc = M_B_ccond_back_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_o_back_enc = M_W_o_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_o_back_enc = M_U_o_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_o_back_enc = M_B_o_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				//				  
				//				  
				MatrixXld _V_W_out = V_W_out.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_out = V_B_out.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_W_gamma_layernorm = V_W_gamma_layernorm.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_beta_layernorm = V_B_beta_layernorm.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_V_a_attention = V_V_a_attention.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_W_e_attention = V_W_e_attention.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_W_d_attention = V_W_d_attention.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_W_f_dec = V_W_f_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_f_dec = V_U_f_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_f_dec = V_B_f_dec.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_i_dec = V_W_i_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_i_dec = V_U_i_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_i_dec = V_B_i_dec.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_ccond_dec = V_W_ccond_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_ccond_dec = V_U_ccond_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_ccond_dec = V_B_ccond_dec.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_o_dec = V_W_o_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_o_dec = V_U_o_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_o_dec = V_B_o_dec.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_W_f_forw_enc = V_W_f_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_f_forw_enc = V_U_f_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_f_forw_enc = V_B_f_forw_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_i_forw_enc = V_W_i_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_i_forw_enc = V_U_i_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_i_forw_enc = V_B_i_forw_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_ccond_forw_enc = V_W_ccond_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_ccond_forw_enc = V_U_ccond_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_ccond_forw_enc = V_B_ccond_forw_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_o_forw_enc = V_W_o_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_o_forw_enc = V_U_o_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_o_forw_enc = V_B_o_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				//				   
				MatrixXld _V_W_f_back_enc = V_W_f_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_f_back_enc = V_U_f_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_f_back_enc = V_B_f_back_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_i_back_enc = V_W_i_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_i_back_enc = V_U_i_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_i_back_enc = V_B_i_back_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_ccond_back_enc = V_W_ccond_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_ccond_back_enc = V_U_ccond_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_ccond_back_enc = V_B_ccond_back_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_o_back_enc = V_W_o_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_o_back_enc = V_U_o_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_o_back_enc = V_B_o_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				/////
				/////
				/////
				/////
				grads.dW_out.array() -= learning_rate * _M_W_out.array() / (_V_W_out.array().sqrt() + epsilon);
				grads.dB_out.array() -= learning_rate * _M_B_out.array() / (_V_B_out.array().sqrt() + epsilon);
				//
				grads.dW_gamma_layernorm.array() -= learning_rate * _M_W_gamma_layernorm.array() / (_V_W_gamma_layernorm.array().sqrt() + epsilon);
				grads.dB_beta_layernorm.array() -= learning_rate * _M_B_beta_layernorm.array() / (_V_B_beta_layernorm.array().sqrt() + epsilon);
				//
				grads.dV_a_attention.array() -= learning_rate * _M_V_a_attention.array() / (_V_V_a_attention.array().sqrt() + epsilon);
				grads.dW_e_attention.array() -= learning_rate * _M_W_e_attention.array() / (_V_W_e_attention.array().sqrt() + epsilon);
				grads.dW_d_attention.array() -= learning_rate * _M_W_d_attention.array() / (_V_W_d_attention.array().sqrt() + epsilon);
				//
				grads.dW_f_dec.array() -= learning_rate * _M_W_f_dec.array() / (_V_W_f_dec.array().sqrt() + epsilon);
				grads.dU_f_dec.array() -= learning_rate * _M_U_f_dec.array() / (_V_U_f_dec.array().sqrt() + epsilon);
				grads.dB_f_dec.array() -= learning_rate * _M_B_f_dec.array() / (_V_B_f_dec.array().sqrt() + epsilon);

				grads.dW_i_dec.array() -= learning_rate * _M_W_i_dec.array() / (_V_W_i_dec.array().sqrt() + epsilon);
				grads.dU_i_dec.array() -= learning_rate * _M_U_i_dec.array() / (_V_U_i_dec.array().sqrt() + epsilon);
				grads.dB_i_dec.array() -= learning_rate * _M_B_i_dec.array() / (_V_B_i_dec.array().sqrt() + epsilon);

				grads.dW_ccond_dec.array() -= learning_rate * _M_W_ccond_dec.array() / (_V_W_ccond_dec.array().sqrt() + epsilon);
				grads.dU_ccond_dec.array() -= learning_rate * _M_U_ccond_dec.array() / (_V_U_ccond_dec.array().sqrt() + epsilon);
				grads.dB_ccond_dec.array() -= learning_rate * _M_B_ccond_dec.array() / (_V_B_ccond_dec.array().sqrt() + epsilon);

				grads.dW_o_dec.array() -= learning_rate * _M_W_o_dec.array() / (_V_W_o_dec.array().sqrt() + epsilon);
				grads.dU_o_dec.array() -= learning_rate * _M_U_o_dec.array() / (_V_U_o_dec.array().sqrt() + epsilon);
				grads.dB_o_dec.array() -= learning_rate * _M_B_o_dec.array() / (_V_B_o_dec.array().sqrt() + epsilon);

				//
				grads.dW_f_forw_enc.array() -= learning_rate * _M_W_f_forw_enc.array() / (_V_W_f_forw_enc.array().sqrt() + epsilon);
				grads.dU_f_forw_enc.array() -= learning_rate * _M_U_f_forw_enc.array() / (_V_U_f_forw_enc.array().sqrt() + epsilon);
				grads.dB_f_forw_enc.array() -= learning_rate * _M_B_f_forw_enc.array() / (_V_B_f_forw_enc.array().sqrt() + epsilon);

				grads.dW_i_forw_enc.array() -= learning_rate * _M_W_i_forw_enc.array() / (_V_W_i_forw_enc.array().sqrt() + epsilon);
				grads.dU_i_forw_enc.array() -= learning_rate * _M_U_i_forw_enc.array() / (_V_U_i_forw_enc.array().sqrt() + epsilon);
				grads.dB_i_forw_enc.array() -= learning_rate * _M_B_i_forw_enc.array() / (_V_B_i_forw_enc.array().sqrt() + epsilon);

				grads.dW_ccond_forw_enc.array() -= learning_rate * _M_W_ccond_forw_enc.array() / (_V_W_ccond_forw_enc.array().sqrt() + epsilon);
				grads.dU_ccond_forw_enc.array() -= learning_rate * _M_U_ccond_forw_enc.array() / (_V_U_ccond_forw_enc.array().sqrt() + epsilon);
				grads.dB_ccond_forw_enc.array() -= learning_rate * _M_B_ccond_forw_enc.array() / (_V_B_ccond_forw_enc.array().sqrt() + epsilon);

				grads.dW_o_forw_enc.array() -= learning_rate * _M_W_o_forw_enc.array() / (_V_W_o_forw_enc.array().sqrt() + epsilon);
				grads.dU_o_forw_enc.array() -= learning_rate * _M_U_o_forw_enc.array() / (_V_U_o_forw_enc.array().sqrt() + epsilon);
				grads.dB_o_forw_enc.array() -= learning_rate * _M_B_o_forw_enc.array() / (_V_B_o_forw_enc.array().sqrt() + epsilon);
				//				   
				grads.dW_f_back_enc.array() -= learning_rate * _M_W_f_back_enc.array() / (_V_W_f_back_enc.array().sqrt() + epsilon);
				grads.dU_f_back_enc.array() -= learning_rate * _M_U_f_back_enc.array() / (_V_U_f_back_enc.array().sqrt() + epsilon);
				grads.dB_f_back_enc.array() -= learning_rate * _M_B_f_back_enc.array() / (_V_B_f_back_enc.array().sqrt() + epsilon);

				grads.dW_i_back_enc.array() -= learning_rate * _M_W_i_back_enc.array() / (_V_W_i_back_enc.array().sqrt() + epsilon);
				grads.dU_i_back_enc.array() -= learning_rate * _M_U_i_back_enc.array() / (_V_U_i_back_enc.array().sqrt() + epsilon);
				grads.dB_i_back_enc.array() -= learning_rate * _M_B_i_back_enc.array() / (_V_B_i_back_enc.array().sqrt() + epsilon);

				grads.dW_ccond_back_enc.array() -= learning_rate * _M_W_ccond_back_enc.array() / (_V_W_ccond_back_enc.array().sqrt() + epsilon);
				grads.dU_ccond_back_enc.array() -= learning_rate * _M_U_ccond_back_enc.array() / (_V_U_ccond_back_enc.array().sqrt() + epsilon);
				grads.dB_ccond_back_enc.array() -= learning_rate * _M_B_ccond_back_enc.array() / (_V_B_ccond_back_enc.array().sqrt() + epsilon);

				grads.dW_o_back_enc.array() -= learning_rate * _M_W_o_back_enc.array() / (_V_W_o_back_enc.array().sqrt() + epsilon);
				grads.dU_o_back_enc.array() -= learning_rate * _M_U_o_back_enc.array() / (_V_U_o_back_enc.array().sqrt() + epsilon);
				grads.dB_o_back_enc.array() -= learning_rate * _M_B_o_back_enc.array() / (_V_B_o_back_enc.array().sqrt() + epsilon);
			}
		}
	}
}

void Seq2SeqWithAttention_ForTrain::UpdateAdamOptWithLogging
(
	const std::vector<std::vector<MatrixXld>>& Target_input_output, /*std::vector<MatrixXld> Target_output,*/
	size_t epochs, size_t optima_steps, size_t batch_size,
	long double learning_rate, long double epsilon,
	long double beta1, long double beta2
)
{
	auto get_global_norm = [](auto& grads)  {
		long double total_sq_norm = 0.0;

		// -------- Decoder --------
		total_sq_norm += grads.dW_out.squaredNorm();
		total_sq_norm += grads.dB_out.squaredNorm();

		total_sq_norm += grads.dW_f_dec.squaredNorm();
		total_sq_norm += grads.dU_f_dec.squaredNorm();
		total_sq_norm += grads.dB_f_dec.squaredNorm();

		total_sq_norm += grads.dW_i_dec.squaredNorm();
		total_sq_norm += grads.dU_i_dec.squaredNorm();
		total_sq_norm += grads.dB_i_dec.squaredNorm();

		total_sq_norm += grads.dW_ccond_dec.squaredNorm();
		total_sq_norm += grads.dU_ccond_dec.squaredNorm();
		total_sq_norm += grads.dB_ccond_dec.squaredNorm();

		total_sq_norm += grads.dW_o_dec.squaredNorm();
		total_sq_norm += grads.dU_o_dec.squaredNorm();
		total_sq_norm += grads.dB_o_dec.squaredNorm();

		// -------- LayerNorm --------
		total_sq_norm += grads.dW_gamma_layernorm.squaredNorm();
		total_sq_norm += grads.dB_beta_layernorm.squaredNorm();

		// -------- Attention --------
		total_sq_norm += grads.dV_a_attention.squaredNorm();
		total_sq_norm += grads.dW_e_attention.squaredNorm();
		total_sq_norm += grads.dW_d_attention.squaredNorm();

		// -------- Forward Encoder --------
		total_sq_norm += grads.dW_f_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_f_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_f_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_i_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_i_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_i_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_ccond_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_ccond_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_ccond_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_o_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_o_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_o_forw_enc.squaredNorm();

		// -------- Backward Encoder --------
		total_sq_norm += grads.dW_f_back_enc.squaredNorm();
		total_sq_norm += grads.dU_f_back_enc.squaredNorm();
		total_sq_norm += grads.dB_f_back_enc.squaredNorm();

		total_sq_norm += grads.dW_i_back_enc.squaredNorm();
		total_sq_norm += grads.dU_i_back_enc.squaredNorm();
		total_sq_norm += grads.dB_i_back_enc.squaredNorm();

		total_sq_norm += grads.dW_ccond_back_enc.squaredNorm();
		total_sq_norm += grads.dU_ccond_back_enc.squaredNorm();
		total_sq_norm += grads.dB_ccond_back_enc.squaredNorm();

		total_sq_norm += grads.dW_o_back_enc.squaredNorm();
		total_sq_norm += grads.dU_o_back_enc.squaredNorm();
		total_sq_norm += grads.dB_o_back_enc.squaredNorm();

		long double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		return global_norm;
		};

	auto clip_by_global_norm = [](auto& grads, long double clip_value) {
		long double total_sq_norm = 0.0;

		// -------- Decoder --------
		total_sq_norm += grads.dW_out.squaredNorm();
		total_sq_norm += grads.dB_out.squaredNorm();

		total_sq_norm += grads.dW_f_dec.squaredNorm();
		total_sq_norm += grads.dU_f_dec.squaredNorm();
		total_sq_norm += grads.dB_f_dec.squaredNorm();

		total_sq_norm += grads.dW_i_dec.squaredNorm();
		total_sq_norm += grads.dU_i_dec.squaredNorm();
		total_sq_norm += grads.dB_i_dec.squaredNorm();

		total_sq_norm += grads.dW_ccond_dec.squaredNorm();
		total_sq_norm += grads.dU_ccond_dec.squaredNorm();
		total_sq_norm += grads.dB_ccond_dec.squaredNorm();

		total_sq_norm += grads.dW_o_dec.squaredNorm();
		total_sq_norm += grads.dU_o_dec.squaredNorm();
		total_sq_norm += grads.dB_o_dec.squaredNorm();

		// -------- LayerNorm --------
		total_sq_norm += grads.dW_gamma_layernorm.squaredNorm();
		total_sq_norm += grads.dB_beta_layernorm.squaredNorm();

		// -------- Attention --------
		total_sq_norm += grads.dV_a_attention.squaredNorm();
		total_sq_norm += grads.dW_e_attention.squaredNorm();
		total_sq_norm += grads.dW_d_attention.squaredNorm();

		// -------- Forward Encoder --------
		total_sq_norm += grads.dW_f_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_f_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_f_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_i_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_i_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_i_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_ccond_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_ccond_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_ccond_forw_enc.squaredNorm();

		total_sq_norm += grads.dW_o_forw_enc.squaredNorm();
		total_sq_norm += grads.dU_o_forw_enc.squaredNorm();
		total_sq_norm += grads.dB_o_forw_enc.squaredNorm();

		// -------- Backward Encoder --------
		total_sq_norm += grads.dW_f_back_enc.squaredNorm();
		total_sq_norm += grads.dU_f_back_enc.squaredNorm();
		total_sq_norm += grads.dB_f_back_enc.squaredNorm();

		total_sq_norm += grads.dW_i_back_enc.squaredNorm();
		total_sq_norm += grads.dU_i_back_enc.squaredNorm();
		total_sq_norm += grads.dB_i_back_enc.squaredNorm();

		total_sq_norm += grads.dW_ccond_back_enc.squaredNorm();
		total_sq_norm += grads.dU_ccond_back_enc.squaredNorm();
		total_sq_norm += grads.dB_ccond_back_enc.squaredNorm();

		total_sq_norm += grads.dW_o_back_enc.squaredNorm();
		total_sq_norm += grads.dU_o_back_enc.squaredNorm();
		total_sq_norm += grads.dB_o_back_enc.squaredNorm();

		long double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		if (global_norm > clip_value) {
			long double scale = clip_value / global_norm;

			// -------- Decoder --------
			grads.dW_out *= scale;
			grads.dB_out *= scale;

			grads.dW_f_dec *= scale;
			grads.dU_f_dec *= scale;
			grads.dB_f_dec *= scale;

			grads.dW_i_dec *= scale;
			grads.dU_i_dec *= scale;
			grads.dB_i_dec *= scale;

			grads.dW_ccond_dec *= scale;
			grads.dU_ccond_dec *= scale;
			grads.dB_ccond_dec *= scale;

			grads.dW_o_dec *= scale;
			grads.dU_o_dec *= scale;
			grads.dB_o_dec *= scale;

			// -------- LayerNorm --------
			grads.dW_gamma_layernorm *= scale;
			grads.dB_beta_layernorm *= scale;

			// -------- Attention --------
			grads.dV_a_attention *= scale;
			grads.dW_e_attention *= scale;
			grads.dW_d_attention *= scale;

			// -------- Forward Encoder --------
			grads.dW_f_forw_enc *= scale;
			grads.dU_f_forw_enc *= scale;
			grads.dB_f_forw_enc *= scale;

			grads.dW_i_forw_enc *= scale;
			grads.dU_i_forw_enc *= scale;
			grads.dB_i_forw_enc *= scale;

			grads.dW_ccond_forw_enc *= scale;
			grads.dU_ccond_forw_enc *= scale;
			grads.dB_ccond_forw_enc *= scale;
			grads.dW_o_forw_enc *= scale;
			grads.dU_o_forw_enc *= scale;
			grads.dB_o_forw_enc *= scale;

			// -------- Backward Encoder --------
			grads.dW_f_back_enc *= scale;
			grads.dU_f_back_enc *= scale;
			grads.dB_f_back_enc *= scale;

			grads.dW_i_back_enc *= scale;
			grads.dU_i_back_enc *= scale;
			grads.dB_i_back_enc *= scale;

			grads.dW_ccond_back_enc *= scale;
			grads.dU_ccond_back_enc *= scale;
			grads.dB_ccond_back_enc *= scale;

			grads.dW_o_back_enc *= scale;
			grads.dU_o_back_enc *= scale;
			grads.dB_o_back_enc *= scale;
		}
		};

	std::vector<std::vector<MatrixXld>> shuffle_target = Target_input_output;
	long double notceil_batch_steps_ = (long double)shuffle_target.size() / batch_size;
	size_t batch_steps_ = (size_t)std::ceil(notceil_batch_steps_);

	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point end_time;

	for (size_t epoch_ = 0; epoch_ < epochs; epoch_++) {
		auto get_mean_grads = [](auto & grads) {
			long double total_norm = 0.0;

			// -------- Decoder --------
			total_norm += grads.dW_out.sum();
			total_norm += grads.dB_out.sum();

			total_norm += grads.dW_f_dec.sum();
			total_norm += grads.dU_f_dec.sum();
			total_norm += grads.dB_f_dec.sum();

			total_norm += grads.dW_i_dec.sum();
			total_norm += grads.dU_i_dec.sum();
			total_norm += grads.dB_i_dec.sum();

			total_norm += grads.dW_ccond_dec.sum();
			total_norm += grads.dU_ccond_dec.sum();
			total_norm += grads.dB_ccond_dec.sum();

			total_norm += grads.dW_o_dec.sum();
			total_norm += grads.dU_o_dec.sum();
			total_norm += grads.dB_o_dec.sum();

			// ----- LayerNorm --------
			total_norm += grads.dW_gamma_layernorm.sum();
			total_norm += grads.dB_beta_layernorm.sum();

			// ----- Attention --------
			total_norm += grads.dV_a_attention.sum();
			total_norm += grads.dW_e_attention.sum();
			total_norm += grads.dW_d_attention.sum();

			// ----- Forward Encoder --------
			total_norm += grads.dW_f_forw_enc.sum();
			total_norm += grads.dU_f_forw_enc.sum();
			total_norm += grads.dB_f_forw_enc.sum();

			total_norm += grads.dW_i_forw_enc.sum();
			total_norm += grads.dU_i_forw_enc.sum();
			total_norm += grads.dB_i_forw_enc.sum();

			total_norm += grads.dW_ccond_forw_enc.sum();
			total_norm += grads.dU_ccond_forw_enc.sum();
			total_norm += grads.dB_ccond_forw_enc.sum();

			total_norm += grads.dW_o_forw_enc.sum();
			total_norm += grads.dU_o_forw_enc.sum();
			total_norm += grads.dB_o_forw_enc.sum();

			// ----- Backward Encoder --------
			total_norm += grads.dW_f_back_enc.sum();
			total_norm += grads.dU_f_back_enc.sum();
			total_norm += grads.dB_f_back_enc.sum();

			total_norm += grads.dW_i_back_enc.sum();
			total_norm += grads.dU_i_back_enc.sum();
			total_norm += grads.dB_i_back_enc.sum();

			total_norm += grads.dW_ccond_back_enc.sum();
			total_norm += grads.dU_ccond_back_enc.sum();
			total_norm += grads.dB_ccond_back_enc.sum();

			total_norm += grads.dW_o_back_enc.sum();
			total_norm += grads.dU_o_back_enc.sum();
			total_norm += grads.dB_o_back_enc.sum();

			long double mean = total_norm / 43;
			return mean;
			};

		//long double start_avg_train_loss = 0;
		//long double end_avg_train_loss = 0;

		grads_Seq2SeqWithAttention grads_start_avg_train_loss;
		grads_Seq2SeqWithAttention grads_end_avg_train_loss;

		start_time = std::chrono::steady_clock::now();

		std::random_device rd;
		std::shuffle(shuffle_target.begin(), shuffle_target.end(), std::mt19937(rd()));

		grads_Seq2SeqWithAttention grads;

		Inference(shuffle_target[0]);

		for (size_t batch_step = 0; batch_step < batch_steps_; batch_step++) {
			for (size_t i = batch_step * batch_size; i < (batch_step + 1) * batch_size && i < shuffle_target.size(); i++) {
				grads += std::move(BackwardWithLogging(i, shuffle_target[1][i]));
			}
			grads /= (batch_step == batch_steps_) ? batch_size * (notceil_batch_steps_ - (int)notceil_batch_steps_) : batch_size;

			grads_start_avg_train_loss += std::move(grads);
			grads.SetZero();

			long double clip_threshold = 200L;

			long double grad_norm = get_global_norm(grads);

			clip_by_global_norm(grads, clip_threshold);

			if (!std::isfinite(grad_norm)) {
				std::cerr << "[WARNING] NaN/inf in gradients at batch " << batch_step << "\n";
			}
			else if (grad_norm > clip_threshold) {
				std::cout << "[CLIP] Batch " << batch_step
					<< " gradient norm = " << grad_norm << " clipped\n";
			}
			else {
				std::cout << "[INFO] Batch " << batch_step
					<< " gradient norm = " << grad_norm << "\n";
			}

			MatrixXld M_W_out; MatrixXld M_B_out;

			MatrixXld M_W_gamma_layernorm; MatrixXld M_B_beta_layernorm;

			MatrixXld M_V_a_attention, M_W_e_attention, M_W_d_attention;

			MatrixXld M_W_f_dec, M_U_f_dec; MatrixXld M_B_f_dec;
			MatrixXld M_W_i_dec, M_U_i_dec; MatrixXld M_B_i_dec;
			MatrixXld M_W_ccond_dec, M_U_ccond_dec; MatrixXld M_B_ccond_dec;
			MatrixXld M_W_o_dec, M_U_o_dec; MatrixXld M_B_o_dec;

			MatrixXld M_W_f_forw_enc, M_U_f_forw_enc; MatrixXld M_B_f_forw_enc;
			MatrixXld M_W_i_forw_enc, M_U_i_forw_enc; MatrixXld M_B_i_forw_enc;
			MatrixXld M_W_ccond_forw_enc, M_U_ccond_forw_enc; MatrixXld M_B_ccond_forw_enc;
			MatrixXld M_W_o_forw_enc, M_U_o_forw_enc; MatrixXld M_B_o_forw_enc;

			MatrixXld M_W_f_back_enc, M_U_f_back_enc; MatrixXld M_B_f_back_enc;
			MatrixXld M_W_i_back_enc, M_U_i_back_enc; MatrixXld M_B_i_back_enc;
			MatrixXld M_W_ccond_back_enc, M_U_ccond_back_enc; MatrixXld M_B_ccond_back_enc;
			MatrixXld M_W_o_back_enc, M_U_o_back_enc; MatrixXld M_B_o_back_enc;


			MatrixXld V_W_out; MatrixXld V_B_out;

			MatrixXld V_W_gamma_layernorm; MatrixXld V_B_beta_layernorm;

			MatrixXld V_V_a_attention, V_W_e_attention, V_W_d_attention;

			MatrixXld V_W_f_dec, V_U_f_dec; MatrixXld V_B_f_dec;
			MatrixXld V_W_i_dec, V_U_i_dec; MatrixXld V_B_i_dec;
			MatrixXld V_W_ccond_dec, V_U_ccond_dec; MatrixXld V_B_ccond_dec;
			MatrixXld V_W_o_dec, V_U_o_dec; MatrixXld V_B_o_dec;

			MatrixXld V_W_f_forw_enc, V_U_f_forw_enc; MatrixXld V_B_f_forw_enc;
			MatrixXld V_W_i_forw_enc, V_U_i_forw_enc; MatrixXld V_B_i_forw_enc;
			MatrixXld V_W_ccond_forw_enc, V_U_ccond_forw_enc; MatrixXld V_B_ccond_forw_enc;
			MatrixXld V_W_o_forw_enc, V_U_o_forw_enc; MatrixXld V_B_o_forw_enc;

			MatrixXld V_W_f_back_enc, V_U_f_back_enc; MatrixXld V_B_f_back_enc;
			MatrixXld V_W_i_back_enc, V_U_i_back_enc; MatrixXld V_B_i_back_enc;
			MatrixXld V_W_ccond_back_enc, V_U_ccond_back_enc; MatrixXld V_B_ccond_back_enc;
			MatrixXld V_W_o_back_enc, V_U_o_back_enc; MatrixXld V_B_o_back_enc;

			for (size_t t_ = 0; t_ < optima_steps; t_++) {
				grads.SetZero();
				for (size_t i = batch_step * batch_size; i < (batch_step + 1) * batch_size && i < shuffle_target.size(); i++) {
					grads += std::move(BackwardWithLogging(i, shuffle_target[1][i]));
				}
				grads /= (batch_step == batch_steps_) ? batch_size * (notceil_batch_steps_ - (int)notceil_batch_steps_) : batch_size;

				{
					M_W_out = beta1 * M_W_out + (1 - beta1) * grads.dW_out;
					M_B_out = beta1 * M_B_out + (1 - beta1) * grads.dB_out;
					//
					M_W_gamma_layernorm = beta1 * M_W_gamma_layernorm + (1 - beta1) * grads.dW_gamma_layernorm;
					M_B_beta_layernorm = beta1 * M_B_beta_layernorm + (1 - beta1) * grads.dB_beta_layernorm;
					//
					M_V_a_attention = beta1 * M_V_a_attention + (1 - beta1) * grads.dV_a_attention;
					M_W_e_attention = beta1 * M_W_e_attention + (1 - beta1) * grads.dW_e_attention;
					M_W_d_attention = beta1 * M_W_d_attention + (1 - beta1) * grads.dW_d_attention;
					//
					M_W_f_dec = beta1 * M_W_f_dec + (1 - beta1) * grads.dW_f_dec;
					M_U_f_dec = beta1 * M_U_f_dec + (1 - beta1) * grads.dU_f_dec;
					M_B_f_dec = beta1 * M_B_f_dec + (1 - beta1) * grads.dB_f_dec;

					M_W_i_dec = beta1 * M_W_i_dec + (1 - beta1) * grads.dW_i_dec;
					M_U_i_dec = beta1 * M_U_i_dec + (1 - beta1) * grads.dU_i_dec;
					M_B_i_dec = beta1 * M_B_i_dec + (1 - beta1) * grads.dB_i_dec;

					M_W_ccond_dec = beta1 * M_W_ccond_dec + (1 - beta1) * grads.dW_ccond_dec;
					M_U_ccond_dec = beta1 * M_U_ccond_dec + (1 - beta1) * grads.dU_ccond_dec;
					M_B_ccond_dec = beta1 * M_B_ccond_dec + (1 - beta1) * grads.dB_ccond_dec;

					M_W_o_dec = beta1 * M_W_o_dec + (1 - beta1) * grads.dW_o_dec;
					M_U_o_dec = beta1 * M_U_o_dec + (1 - beta1) * grads.dU_o_dec;
					M_B_o_dec = beta1 * M_B_o_dec + (1 - beta1) * grads.dB_o_dec;
					//
					M_W_f_forw_enc = beta1 * M_W_f_forw_enc + (1 - beta1) * grads.dW_f_forw_enc;
					M_U_f_forw_enc = beta1 * M_U_f_forw_enc + (1 - beta1) * grads.dU_f_forw_enc;
					M_B_f_forw_enc = beta1 * M_B_f_forw_enc + (1 - beta1) * grads.dB_f_forw_enc;

					M_W_i_forw_enc = beta1 * M_W_i_forw_enc + (1 - beta1) * grads.dW_i_forw_enc;
					M_U_i_forw_enc = beta1 * M_U_i_forw_enc + (1 - beta1) * grads.dU_i_forw_enc;
					M_B_i_forw_enc = beta1 * M_B_i_forw_enc + (1 - beta1) * grads.dB_i_forw_enc;

					M_W_ccond_forw_enc = beta1 * M_W_ccond_forw_enc + (1 - beta1) * grads.dW_ccond_forw_enc;
					M_U_ccond_forw_enc = beta1 * M_U_ccond_forw_enc + (1 - beta1) * grads.dU_ccond_forw_enc;
					M_B_ccond_forw_enc = beta1 * M_B_ccond_forw_enc + (1 - beta1) * grads.dB_ccond_forw_enc;

					M_W_o_forw_enc = beta1 * M_W_o_forw_enc + (1 - beta1) * grads.dW_o_forw_enc;
					M_U_o_forw_enc = beta1 * M_U_o_forw_enc + (1 - beta1) * grads.dU_o_forw_enc;
					M_B_o_forw_enc = beta1 * M_B_o_forw_enc + (1 - beta1) * grads.dB_o_forw_enc;
					//
					M_W_f_back_enc = beta1 * M_W_f_back_enc + (1 - beta1) * grads.dW_f_back_enc;
					M_U_f_back_enc = beta1 * M_U_f_back_enc + (1 - beta1) * grads.dU_f_back_enc;
					M_B_f_back_enc = beta1 * M_B_f_back_enc + (1 - beta1) * grads.dB_f_back_enc;

					M_W_i_back_enc = beta1 * M_W_i_back_enc + (1 - beta1) * grads.dW_i_back_enc;
					M_U_i_back_enc = beta1 * M_U_i_back_enc + (1 - beta1) * grads.dU_i_back_enc;
					M_B_i_back_enc = beta1 * M_B_i_back_enc + (1 - beta1) * grads.dB_i_back_enc;

					M_W_ccond_back_enc = beta1 * M_W_ccond_back_enc + (1 - beta1) * grads.dW_ccond_back_enc;
					M_U_ccond_back_enc = beta1 * M_U_ccond_back_enc + (1 - beta1) * grads.dU_ccond_back_enc;
					M_B_ccond_back_enc = beta1 * M_B_ccond_back_enc + (1 - beta1) * grads.dB_ccond_back_enc;

					M_W_o_back_enc = beta1 * M_W_o_back_enc + (1 - beta1) * grads.dW_o_back_enc;
					M_U_o_back_enc = beta1 * M_U_o_back_enc + (1 - beta1) * grads.dU_o_back_enc;
					M_B_o_back_enc = beta1 * M_B_o_back_enc + (1 - beta1) * grads.dB_o_back_enc;
					//
					//
					V_W_out = beta2 * V_W_out.array() + (1 - beta2) * grads.dW_out.array() * grads.dW_out.array();
					V_B_out = beta2 * V_B_out.array() + (1 - beta2) * grads.dB_out.array() * grads.dB_out.array();
					//
					V_W_gamma_layernorm = beta2 * V_W_gamma_layernorm.array() + (1 - beta2) * grads.dW_gamma_layernorm.array() * grads.dW_gamma_layernorm.array();
					V_B_beta_layernorm = beta2 * V_B_beta_layernorm.array() + (1 - beta2) * grads.dB_beta_layernorm.array() * grads.dB_beta_layernorm.array();
					//
					V_V_a_attention = beta2 * V_V_a_attention.array() + (1 - beta2) * grads.dV_a_attention.array() * grads.dV_a_attention.array();
					V_W_e_attention = beta2 * V_W_e_attention.array() + (1 - beta2) * grads.dW_e_attention.array() * grads.dW_e_attention.array();
					V_W_d_attention = beta2 * V_W_d_attention.array() + (1 - beta2) * grads.dW_d_attention.array() * grads.dW_d_attention.array();
					//
					V_W_f_dec = beta2 * V_W_f_dec.array() + (1 - beta2) * grads.dW_f_dec.array() * grads.dW_f_dec.array();
					V_U_f_dec = beta2 * V_U_f_dec.array() + (1 - beta2) * grads.dU_f_dec.array() * grads.dU_f_dec.array();
					V_B_f_dec = beta2 * V_B_f_dec.array() + (1 - beta2) * grads.dB_f_dec.array() * grads.dB_f_dec.array();

					V_W_i_dec = beta2 * V_W_i_dec.array() + (1 - beta2) * grads.dW_i_dec.array() * grads.dW_i_dec.array();
					V_U_i_dec = beta2 * V_U_i_dec.array() + (1 - beta2) * grads.dU_i_dec.array() * grads.dU_i_dec.array();
					V_B_i_dec = beta2 * V_B_i_dec.array() + (1 - beta2) * grads.dB_i_dec.array() * grads.dB_i_dec.array();

					V_W_ccond_dec = beta2 * V_W_ccond_dec.array() + (1 - beta2) * grads.dW_ccond_dec.array() * grads.dW_ccond_dec.array();
					V_U_ccond_dec = beta2 * V_U_ccond_dec.array() + (1 - beta2) * grads.dU_ccond_dec.array() * grads.dU_ccond_dec.array();
					V_B_ccond_dec = beta2 * V_B_ccond_dec.array() + (1 - beta2) * grads.dB_ccond_dec.array() * grads.dB_ccond_dec.array();

					V_W_o_dec = beta2 * V_W_o_dec.array() + (1 - beta2) * grads.dW_o_dec.array() * grads.dW_o_dec.array();
					V_U_o_dec = beta2 * V_U_o_dec.array() + (1 - beta2) * grads.dU_o_dec.array() * grads.dU_o_dec.array();
					V_B_o_dec = beta2 * V_B_o_dec.array() + (1 - beta2) * grads.dB_o_dec.array() * grads.dB_o_dec.array();
					//
					V_W_f_forw_enc = beta2 * V_W_f_forw_enc.array() + (1 - beta2) * grads.dW_f_forw_enc.array() * grads.dW_f_forw_enc.array();
					V_U_f_forw_enc = beta2 * V_U_f_forw_enc.array() + (1 - beta2) * grads.dU_f_forw_enc.array() * grads.dU_f_forw_enc.array();
					V_B_f_forw_enc = beta2 * V_B_f_forw_enc.array() + (1 - beta2) * grads.dB_f_forw_enc.array() * grads.dB_f_forw_enc.array();

					V_W_i_forw_enc = beta2 * V_W_i_forw_enc.array() + (1 - beta2) * grads.dW_i_forw_enc.array() * grads.dW_i_forw_enc.array();
					V_U_i_forw_enc = beta2 * V_U_i_forw_enc.array() + (1 - beta2) * grads.dU_i_forw_enc.array() * grads.dU_i_forw_enc.array();
					V_B_i_forw_enc = beta2 * V_B_i_forw_enc.array() + (1 - beta2) * grads.dB_i_forw_enc.array() * grads.dB_i_forw_enc.array();

					V_W_ccond_forw_enc = beta2 * V_W_ccond_forw_enc.array() + (1 - beta2) * grads.dW_ccond_forw_enc.array() * grads.dW_ccond_forw_enc.array();
					V_U_ccond_forw_enc = beta2 * V_U_ccond_forw_enc.array() + (1 - beta2) * grads.dU_ccond_forw_enc.array() * grads.dU_ccond_forw_enc.array();
					V_B_ccond_forw_enc = beta2 * V_B_ccond_forw_enc.array() + (1 - beta2) * grads.dB_ccond_forw_enc.array() * grads.dB_ccond_forw_enc.array();

					V_W_o_forw_enc = beta2 * V_W_o_forw_enc.array() + (1 - beta2) * grads.dW_o_forw_enc.array() * grads.dW_o_forw_enc.array();
					V_U_o_forw_enc = beta2 * V_U_o_forw_enc.array() + (1 - beta2) * grads.dU_o_forw_enc.array() * grads.dU_o_forw_enc.array();
					V_B_o_forw_enc = beta2 * V_B_o_forw_enc.array() + (1 - beta2) * grads.dB_o_forw_enc.array() * grads.dB_o_forw_enc.array();
					//
					V_W_f_back_enc = beta2 * V_W_f_back_enc.array() + (1 - beta2) * grads.dW_f_back_enc.array() * grads.dW_f_back_enc.array();
					V_U_f_back_enc = beta2 * V_U_f_back_enc.array() + (1 - beta2) * grads.dU_f_back_enc.array() * grads.dU_f_back_enc.array();
					V_B_f_back_enc = beta2 * V_B_f_back_enc.array() + (1 - beta2) * grads.dB_f_back_enc.array() * grads.dB_f_back_enc.array();

					V_W_i_back_enc = beta2 * V_W_i_back_enc.array() + (1 - beta2) * grads.dW_i_back_enc.array() * grads.dW_i_back_enc.array();
					V_U_i_back_enc = beta2 * V_U_i_back_enc.array() + (1 - beta2) * grads.dU_i_back_enc.array() * grads.dU_i_back_enc.array();
					V_B_i_back_enc = beta2 * V_B_i_back_enc.array() + (1 - beta2) * grads.dB_i_back_enc.array() * grads.dB_i_back_enc.array();

					V_W_ccond_back_enc = beta2 * V_W_ccond_back_enc.array() + (1 - beta2) * grads.dW_ccond_back_enc.array() * grads.dW_ccond_back_enc.array();
					V_U_ccond_back_enc = beta2 * V_U_ccond_back_enc.array() + (1 - beta2) * grads.dU_ccond_back_enc.array() * grads.dU_ccond_back_enc.array();
					V_B_ccond_back_enc = beta2 * V_B_ccond_back_enc.array() + (1 - beta2) * grads.dB_ccond_back_enc.array() * grads.dB_ccond_back_enc.array();

					V_W_o_back_enc = beta2 * V_W_o_back_enc.array() + (1 - beta2) * grads.dW_o_back_enc.array() * grads.dW_o_back_enc.array();
					V_U_o_back_enc = beta2 * V_U_o_back_enc.array() + (1 - beta2) * grads.dU_o_back_enc.array() * grads.dU_o_back_enc.array();
					V_B_o_back_enc = beta2 * V_B_o_back_enc.array() + (1 - beta2) * grads.dB_o_back_enc.array() * grads.dB_o_back_enc.array();
				}
				/////
				/////
				/////
				MatrixXld _M_W_out = M_W_out.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_out = M_B_out.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_W_gamma_layernorm = M_W_gamma_layernorm.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_beta_layernorm = M_B_beta_layernorm.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_V_a_attention = M_V_a_attention.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_W_e_attention = M_W_e_attention.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_W_d_attention = M_W_d_attention.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_W_f_dec = M_W_f_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_f_dec = M_U_f_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_f_dec = M_B_f_dec.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_i_dec = M_W_i_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_i_dec = M_U_i_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_i_dec = M_B_i_dec.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_ccond_dec = M_W_ccond_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_ccond_dec = M_U_ccond_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_ccond_dec = M_B_ccond_dec.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_o_dec = M_W_o_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_o_dec = M_U_o_dec.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_o_dec = M_B_o_dec.array() / (1 - std::pow(beta1, optima_steps));
				//
				MatrixXld _M_W_f_forw_enc = M_W_f_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_f_forw_enc = M_U_f_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_f_forw_enc = M_B_f_forw_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_i_forw_enc = M_W_i_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_i_forw_enc = M_U_i_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_i_forw_enc = M_B_i_forw_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_ccond_forw_enc = M_W_ccond_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_ccond_forw_enc = M_U_ccond_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_ccond_forw_enc = M_B_ccond_forw_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_o_forw_enc = M_W_o_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_o_forw_enc = M_U_o_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_o_forw_enc = M_B_o_forw_enc.array() / (1 - std::pow(beta1, optima_steps));
				//				   
				MatrixXld _M_W_f_back_enc = M_W_f_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_f_back_enc = M_U_f_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_f_back_enc = M_B_f_back_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_i_back_enc = M_W_i_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_i_back_enc = M_U_i_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_i_back_enc = M_B_i_back_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_ccond_back_enc = M_W_ccond_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_ccond_back_enc = M_U_ccond_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_ccond_back_enc = M_B_ccond_back_enc.array() / (1 - std::pow(beta1, optima_steps));

				MatrixXld _M_W_o_back_enc = M_W_o_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_U_o_back_enc = M_U_o_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				MatrixXld _M_B_o_back_enc = M_B_o_back_enc.array() / (1 - std::pow(beta1, optima_steps));
				//				  
				//				  
				MatrixXld _V_W_out = V_W_out.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_out = V_B_out.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_W_gamma_layernorm = V_W_gamma_layernorm.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_beta_layernorm = V_B_beta_layernorm.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_V_a_attention = V_V_a_attention.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_W_e_attention = V_W_e_attention.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_W_d_attention = V_W_d_attention.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_W_f_dec = V_W_f_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_f_dec = V_U_f_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_f_dec = V_B_f_dec.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_i_dec = V_W_i_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_i_dec = V_U_i_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_i_dec = V_B_i_dec.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_ccond_dec = V_W_ccond_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_ccond_dec = V_U_ccond_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_ccond_dec = V_B_ccond_dec.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_o_dec = V_W_o_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_o_dec = V_U_o_dec.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_o_dec = V_B_o_dec.array() / (1 - std::pow(beta2, optima_steps));
				//
				MatrixXld _V_W_f_forw_enc = V_W_f_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_f_forw_enc = V_U_f_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_f_forw_enc = V_B_f_forw_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_i_forw_enc = V_W_i_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_i_forw_enc = V_U_i_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_i_forw_enc = V_B_i_forw_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_ccond_forw_enc = V_W_ccond_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_ccond_forw_enc = V_U_ccond_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_ccond_forw_enc = V_B_ccond_forw_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_o_forw_enc = V_W_o_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_o_forw_enc = V_U_o_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_o_forw_enc = V_B_o_forw_enc.array() / (1 - std::pow(beta2, optima_steps));
				//				   
				MatrixXld _V_W_f_back_enc = V_W_f_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_f_back_enc = V_U_f_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_f_back_enc = V_B_f_back_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_i_back_enc = V_W_i_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_i_back_enc = V_U_i_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_i_back_enc = V_B_i_back_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_ccond_back_enc = V_W_ccond_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_ccond_back_enc = V_U_ccond_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_ccond_back_enc = V_B_ccond_back_enc.array() / (1 - std::pow(beta2, optima_steps));

				MatrixXld _V_W_o_back_enc = V_W_o_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_U_o_back_enc = V_U_o_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				MatrixXld _V_B_o_back_enc = V_B_o_back_enc.array() / (1 - std::pow(beta2, optima_steps));
				/////
				/////
				/////
				/////
				grads.dW_out.array() -= learning_rate * _M_W_out.array() / (_V_W_out.array().sqrt() + epsilon);
				grads.dB_out.array() -= learning_rate * _M_B_out.array() / (_V_B_out.array().sqrt() + epsilon);
				//
				grads.dW_gamma_layernorm.array() -= learning_rate * _M_W_gamma_layernorm.array() / (_V_W_gamma_layernorm.array().sqrt() + epsilon);
				grads.dB_beta_layernorm.array() -= learning_rate * _M_B_beta_layernorm.array() / (_V_B_beta_layernorm.array().sqrt() + epsilon);
				//
				grads.dV_a_attention.array() -= learning_rate * _M_V_a_attention.array() / (_V_V_a_attention.array().sqrt() + epsilon);
				grads.dW_e_attention.array() -= learning_rate * _M_W_e_attention.array() / (_V_W_e_attention.array().sqrt() + epsilon);
				grads.dW_d_attention.array() -= learning_rate * _M_W_d_attention.array() / (_V_W_d_attention.array().sqrt() + epsilon);
				//
				grads.dW_f_dec.array() -= learning_rate * _M_W_f_dec.array() / (_V_W_f_dec.array().sqrt() + epsilon);
				grads.dU_f_dec.array() -= learning_rate * _M_U_f_dec.array() / (_V_U_f_dec.array().sqrt() + epsilon);
				grads.dB_f_dec.array() -= learning_rate * _M_B_f_dec.array() / (_V_B_f_dec.array().sqrt() + epsilon);

				grads.dW_i_dec.array() -= learning_rate * _M_W_i_dec.array() / (_V_W_i_dec.array().sqrt() + epsilon);
				grads.dU_i_dec.array() -= learning_rate * _M_U_i_dec.array() / (_V_U_i_dec.array().sqrt() + epsilon);
				grads.dB_i_dec.array() -= learning_rate * _M_B_i_dec.array() / (_V_B_i_dec.array().sqrt() + epsilon);

				grads.dW_ccond_dec.array() -= learning_rate * _M_W_ccond_dec.array() / (_V_W_ccond_dec.array().sqrt() + epsilon);
				grads.dU_ccond_dec.array() -= learning_rate * _M_U_ccond_dec.array() / (_V_U_ccond_dec.array().sqrt() + epsilon);
				grads.dB_ccond_dec.array() -= learning_rate * _M_B_ccond_dec.array() / (_V_B_ccond_dec.array().sqrt() + epsilon);

				grads.dW_o_dec.array() -= learning_rate * _M_W_o_dec.array() / (_V_W_o_dec.array().sqrt() + epsilon);
				grads.dU_o_dec.array() -= learning_rate * _M_U_o_dec.array() / (_V_U_o_dec.array().sqrt() + epsilon);
				grads.dB_o_dec.array() -= learning_rate * _M_B_o_dec.array() / (_V_B_o_dec.array().sqrt() + epsilon);

				//
				grads.dW_f_forw_enc.array() -= learning_rate * _M_W_f_forw_enc.array() / (_V_W_f_forw_enc.array().sqrt() + epsilon);
				grads.dU_f_forw_enc.array() -= learning_rate * _M_U_f_forw_enc.array() / (_V_U_f_forw_enc.array().sqrt() + epsilon);
				grads.dB_f_forw_enc.array() -= learning_rate * _M_B_f_forw_enc.array() / (_V_B_f_forw_enc.array().sqrt() + epsilon);

				grads.dW_i_forw_enc.array() -= learning_rate * _M_W_i_forw_enc.array() / (_V_W_i_forw_enc.array().sqrt() + epsilon);
				grads.dU_i_forw_enc.array() -= learning_rate * _M_U_i_forw_enc.array() / (_V_U_i_forw_enc.array().sqrt() + epsilon);
				grads.dB_i_forw_enc.array() -= learning_rate * _M_B_i_forw_enc.array() / (_V_B_i_forw_enc.array().sqrt() + epsilon);

				grads.dW_ccond_forw_enc.array() -= learning_rate * _M_W_ccond_forw_enc.array() / (_V_W_ccond_forw_enc.array().sqrt() + epsilon);
				grads.dU_ccond_forw_enc.array() -= learning_rate * _M_U_ccond_forw_enc.array() / (_V_U_ccond_forw_enc.array().sqrt() + epsilon);
				grads.dB_ccond_forw_enc.array() -= learning_rate * _M_B_ccond_forw_enc.array() / (_V_B_ccond_forw_enc.array().sqrt() + epsilon);

				grads.dW_o_forw_enc.array() -= learning_rate * _M_W_o_forw_enc.array() / (_V_W_o_forw_enc.array().sqrt() + epsilon);
				grads.dU_o_forw_enc.array() -= learning_rate * _M_U_o_forw_enc.array() / (_V_U_o_forw_enc.array().sqrt() + epsilon);
				grads.dB_o_forw_enc.array() -= learning_rate * _M_B_o_forw_enc.array() / (_V_B_o_forw_enc.array().sqrt() + epsilon);
				//				   
				grads.dW_f_back_enc.array() -= learning_rate * _M_W_f_back_enc.array() / (_V_W_f_back_enc.array().sqrt() + epsilon);
				grads.dU_f_back_enc.array() -= learning_rate * _M_U_f_back_enc.array() / (_V_U_f_back_enc.array().sqrt() + epsilon);
				grads.dB_f_back_enc.array() -= learning_rate * _M_B_f_back_enc.array() / (_V_B_f_back_enc.array().sqrt() + epsilon);

				grads.dW_i_back_enc.array() -= learning_rate * _M_W_i_back_enc.array() / (_V_W_i_back_enc.array().sqrt() + epsilon);
				grads.dU_i_back_enc.array() -= learning_rate * _M_U_i_back_enc.array() / (_V_U_i_back_enc.array().sqrt() + epsilon);
				grads.dB_i_back_enc.array() -= learning_rate * _M_B_i_back_enc.array() / (_V_B_i_back_enc.array().sqrt() + epsilon);

				grads.dW_ccond_back_enc.array() -= learning_rate * _M_W_ccond_back_enc.array() / (_V_W_ccond_back_enc.array().sqrt() + epsilon);
				grads.dU_ccond_back_enc.array() -= learning_rate * _M_U_ccond_back_enc.array() / (_V_U_ccond_back_enc.array().sqrt() + epsilon);
				grads.dB_ccond_back_enc.array() -= learning_rate * _M_B_ccond_back_enc.array() / (_V_B_ccond_back_enc.array().sqrt() + epsilon);

				grads.dW_o_back_enc.array() -= learning_rate * _M_W_o_back_enc.array() / (_V_W_o_back_enc.array().sqrt() + epsilon);
				grads.dU_o_back_enc.array() -= learning_rate * _M_U_o_back_enc.array() / (_V_U_o_back_enc.array().sqrt() + epsilon);
				grads.dB_o_back_enc.array() -= learning_rate * _M_B_o_back_enc.array() / (_V_B_o_back_enc.array().sqrt() + epsilon);
			}

			grads_end_avg_train_loss += grads;
		}

		grads_start_avg_train_loss /= batch_steps_;
		grads_end_avg_train_loss /= batch_steps_;

		end_time = std::chrono::steady_clock::now();

		std::chrono::duration<double> elapsed = end_time - start_time;
	
		std::cout << "Epoch " << epoch_
			<< " finished. Avg train loss [start/end]: " 
			<< get_mean_grads(grads_start_avg_train_loss) << get_mean_grads(grads_end_avg_train_loss)
			//<< ", Val loss: " << val_loss
			<< ", Time_epoch: " << elapsed.count() << "s\n";
	}
}