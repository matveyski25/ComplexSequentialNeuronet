#include "HeaderSeq2seqWithAttention.h"

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train, std::unique_ptr<Decoder> decoder_train)
	: encoder_(std::move(encoder_train)), decoder_(std::move(decoder_train)) {
}

  
Seq2SeqWithAttention_ForTrain::Backward(size_t Number_InputState, MatrixXld Y_True) {
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

	Eigen::Index T = std::min(this->GetDecoderOutputs()[Number_InputState].rows(), Y_True.rows());
	Eigen::Index N = this->encoder_->Common_Input_states[Number_InputState].rows();

	RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	RowVectorXld _dS_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	for (Eigen::Index t = T; t >= 0; t--) {
		RowVectorXld dY_t = Y_True.row(t) - this->GetDecoderOutputs()[Number_InputState].row(t); //Y_true_t - Y_t
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

		RowVectorXld dO_t = dS_t.array() * C_t.array().tanh() * O_t.array() * (MatrixXld::Constant(O_t.rows(), O_t.cols(), 1) - O_t).array();
		RowVectorXld dC_t = dS_t.array() * O_t.array() *
			(MatrixXld::Constant((C_t * C_t).rows(), (C_t * C_t).cols(), 1) - (ActivationFunctions::Tanh(C_t) * ActivationFunctions::Tanh(C_t))).array() +
			_dC_t.array();
		RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (MatrixXld::Constant((Ccond_t * Ccond_t).rows(), (Ccond_t * Ccond_t).cols(), 1) - Ccond_t * Ccond_t).array();
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
		U << this->decoder_->W_F_H, this->decoder_->W_I_H, this->decoder_->W_C_H, this->decoder_->W_O_H;
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
				(MatrixXld::Constant((Enc_Forw_C_j * Enc_Forw_C_j).rows(), (Enc_Forw_C_j * Enc_Forw_C_j).cols(), 1) - ActivationFunctions::Tanh(Enc_Forw_C_j) * ActivationFunctions::Tanh(Enc_Forw_C_j)).array() +
				Enc_Forw__dC_j.array();
			RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (MatrixXld::Constant((Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).rows(), (Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).cols(), 1) - Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).array();
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
			U_enc_f << this->encoder_->Forward.W_F_H, this->encoder_->Forward.W_I_H, this->encoder_->Forward.W_C_H, this->encoder_->Forward.W_O_H;
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
				(MatrixXld::Constant((Enc_Back_C_j * Enc_Back_C_j).rows(), (Enc_Back_C_j * Enc_Back_C_j).cols(), 1) - ActivationFunctions::Tanh(Enc_Back_C_j) * ActivationFunctions::Tanh(Enc_Back_C_j)).array() +
				Enc_Back__dC_j.array();
			RowVectorXld dEnc_Back_Ccond_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * (MatrixXld::Constant((Enc_Back_Ccond_j * Enc_Back_Ccond_j).rows(), (Enc_Back_Ccond_j * Enc_Back_Ccond_j).cols(), 1) - Enc_Back_Ccond_j * Enc_Back_Ccond_j).array();
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
			U_enc_b << this->encoder_->Backward.W_F_H, this->encoder_->Backward.W_I_H, this->encoder_->Backward.W_C_H, this->encoder_->Backward.W_O_H;
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