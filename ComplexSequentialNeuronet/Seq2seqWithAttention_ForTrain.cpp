#include "HeaderSeq2seqWithAttention.h"


void Seq2SeqWithAttention_ForTrain::Inference()
{
	if (this->Input_States.empty()) { throw std::invalid_argument("Вход пустой"); }
	encoder_->Encode(this->Input_States);
	decoder_->Decode(encoder_->GetEncodedHiddenStates());
}

void Seq2SeqWithAttention_ForTrain::Inference(const std::vector<MatrixXld>& input_sequence_batch)
{
	SetInput_states(input_sequence_batch);
	encoder_->Encode(this->Input_States);
	decoder_->Decode(encoder_->GetEncodedHiddenStates());
}

const std::vector<MatrixXld>& Seq2SeqWithAttention_ForTrain::GetOutputs() const {
	return decoder_->GetOutputStates();
}

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train, std::unique_ptr<Decoder> decoder_train)
	: encoder_(std::move(encoder_train)), decoder_(std::move(decoder_train)) {
}

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(
	Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
	Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
	std::unique_ptr<BahdanauAttention> attention_, size_t batch_size)
	:
	encoder_(std::make_unique<Seq2SeqWithAttention_ForTrain::Encoder>(batch_size, Input_size_, Encoder_Hidden_size_)),
	decoder_(std::make_unique<Seq2SeqWithAttention_ForTrain::Decoder>(std::move(attention_), Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {
}

Seq2SeqWithAttention_ForTrain::grads_Seq2SeqWithAttention Seq2SeqWithAttention_ForTrain::Backward(size_t Number_InputState, MatrixXld Y_True_) {
	MatrixXld Y_True(Y_True_.rows() + this->decoder_->end_token.rows(), Y_True_.cols());
	Y_True.transpose() << Y_True_.transpose(), this->decoder_->end_token.transpose();
	grads_Seq2SeqWithAttention grads;
	grads.SetZero(this);

	Eigen::Index T = std::min(this->GetOutputs()[Number_InputState].rows(), Y_True.rows());
	Eigen::Index N = this->encoder_->Common_Hidden_states[Number_InputState].rows();

	RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	RowVectorXld _dS_t = RowVectorXld::Zero(this->decoder_->Hidden_size);

	RowVectorXld dY_ = RowVectorXld::Zero(Y_True.cols());
	for (int64_t t = static_cast<int64_t>(T) - 1; t >= 0; t--) {
		RowVectorXld dY_t = this->GetOutputs()[Number_InputState].row(t) - Y_True.row(t); //Y_t - Y_true
		MatrixXld DW_out_t = dY_t.transpose() * this->decoder_->StatesForgrads.p__[Number_InputState].row(t);
		//RowVectorXld dp__t = this->decoder_->W_Output.transpose() * dY_t;
		RowVectorXld dp_proj = dY_t * this->decoder_->W_Output;
		RowVectorXld DB_out_t = dY_t;

		RowVectorXld d_p_ = dp_proj.array() * this->decoder_->layernorm_gamma.array();
		RowVectorXld DGamma_t = dp_proj.array() * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).array();
		RowVectorXld DBeta_t = dp_proj;

		const RowVectorXld& x_hat = this->decoder_->StatesForgrads.p_[Number_InputState].row(t); // = p_
		double eps = 1e-5;
		double var = (x_hat.array().square().mean()); // Потому что x̂ уже центрирован: mean = 0
		double stddev = std::sqrt(var + eps);

		// скалярные средние
		double mean_dxhat = d_p_.mean();
		double mean_dxhat_xhat = (d_p_.array() * x_hat.array()).mean();

		// полный градиент по входу p
		RowVectorXld dX = (d_p_.array() - mean_dxhat - x_hat.array() * mean_dxhat_xhat) / stddev;

		RowVectorXld dS_t = dX.head(this->decoder_->Hidden_size) + _dS_t;
		RowVectorXld dContext_t = dX.tail(this->decoder_->attention_->duo_encoder_hidden_size_);

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

		RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (1.0 - O_t.array());
		RowVectorXld dC_t = _dC_t.array() + dS_t.array() * O_t.array() * (1.0 - ActivationFunctions::Tanh(C_t).array().square());
		RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (1.0 - Ccond_t.array().square());
		RowVectorXld dI_t = dC_t.array() * Ccond_t.array() * I_t.array() * (1.0 - I_t.array());
		RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (1.0 - F_t.array());

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
		RowVectorXld DB_dec_t = dGates_t;

		_dC_t = dC_t.array() * F_t.array();
		MatrixXld U(this->decoder_->Hidden_size, 4 * this->decoder_->Hidden_size);
		U << this->decoder_->U_F, this->decoder_->U_I, this->decoder_->U_C, this->decoder_->U_O;
		_dS_t = dGates_t * U.transpose();

		MatrixXld W(this->decoder_->Input_size, 4 * this->decoder_->Hidden_size);
		W << this->decoder_->W_F, this->decoder_->W_I, this->decoder_->W_C, this->decoder_->W_O;
		dContext_t += (dGates_t * W.transpose()).tail(this->decoder_->attention_->duo_encoder_hidden_size_);

		std::vector<RowVectorXld> _dH_Back;
		RowVectorXld Enc_Forw__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
		RowVectorXld Enc_Forw__dH_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);

		for (int64_t j = static_cast<int64_t>(N) - 1; j >= 0; j--) {
			double dE_tj = 0.0;

			const VectorXld& alpha = this->decoder_->StatesForgrads.all_alpha[Number_InputState][t];

			RowVectorXld u_tj = this->decoder_->StatesForgrads.all_u[Number_InputState][t][j];

			RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
			RowVectorXld s_t_1 = this->decoder_->StatesForgrads.h[Number_InputState].row(t == 0 ? 0 : t - 1);

			for (int k = 0; k < N; ++k) {
				RowVectorXld h_k = this->encoder_->Common_Hidden_states[Number_InputState].row(k);

				double alpha_k = alpha(k);
				
				double dAlpha_k = dContext_t.dot(h_k);

				double delta = (j == k) ? 1.0 : 0.0;
				dE_tj += dAlpha_k * alpha_k * (delta - alpha(j));
			}

			RowVectorXld dU_tj = dE_tj * this->decoder_->attention_->attention_vector_.transpose();  // [1 x A]
			RowVectorXld dPreact_tj = dU_tj.array() * (1.0 - u_tj.array().square());


			MatrixXld DW_att_enc_tj = dPreact_tj.transpose() * h_j;
			MatrixXld DW_att_dec_tj = dPreact_tj.transpose() * s_t_1;
			MatrixXld DV_att_tj = u_tj.transpose() * dE_tj;

			RowVectorXld dH_j = alpha(j) * dContext_t + dPreact_tj * this->decoder_->attention_->W_encoder_;

			RowVectorXld dS_att_j = dPreact_tj * this->decoder_->attention_->W_decoder_;

			_dS_t += dS_att_j;

			RowVectorXld dH_forw_j = dH_j.leftCols(this->encoder_->Common_Hidden_size);
			dH_forw_j += Enc_Forw__dH_j;
			RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);

			_dH_Back.insert(_dH_Back.begin(), dH_back_j);

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

			RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (1.0 - Enc_Forw_O_j.array());
			RowVectorXld dEnc_Forw_C_j = Enc_Forw__dC_j.array() + dH_forw_j.array() * Enc_Forw_O_j.array() *
				(1.0 - ActivationFunctions::Tanh(Enc_Forw_C_j).array().square());
			RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (1.0 - Enc_Forw_Ccond_j.array().square());
			RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_Ccond_j.array() * Enc_Forw_I_j.array() * (1.0 - Enc_Forw_I_j.array());
			RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (1.0 - Enc_Forw_F_j.array());

			RowVectorXld dEnc_Forw_Gates_j(4 * this->encoder_->Common_Hidden_size);
			dEnc_Forw_Gates_j << dEnc_Forw_F_j, dEnc_Forw_I_j, dEnc_Forw_Ccond_j, dEnc_Forw_O_j;

			MatrixXld DW_Enc_Forw_j = this->encoder_->Forward.Input_states[Number_InputState].row(j).transpose() * dEnc_Forw_Gates_j;
			MatrixXld DU_Enc_Forw_j;
			if (j == 0) {
				DU_Enc_Forw_j = MatrixXld::Zero(this->encoder_->Forward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
			}
			else {
				DU_Enc_Forw_j = this->encoder_->Forward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Forw_Gates_j;
			}
			RowVectorXld DB_Enc_Forw_j = dEnc_Forw_Gates_j;

			Enc_Forw__dC_j = dEnc_Forw_C_j.array() * Enc_Forw_F_j.array();
			MatrixXld U_enc_f(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);
			U_enc_f << this->encoder_->Forward.U_F, this->encoder_->Forward.U_I, this->encoder_->Forward.U_C, this->encoder_->Forward.U_O;
			Enc_Forw__dH_j = dEnc_Forw_Gates_j * U_enc_f.transpose();

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

			RowVectorXld dEnc_Back_O_j = dH_Back_j.array() * ActivationFunctions::Tanh(Enc_Back_C_j).array() * Enc_Back_O_j.array() * (1.0 - Enc_Back_O_j.array());
			RowVectorXld dEnc_Back_C_j = Enc_Back__dC_j.array() + dH_Back_j.array() * Enc_Back_O_j.array() *
				(1.0 - ActivationFunctions::Tanh(Enc_Back_C_j).array().square());
			RowVectorXld dEnc_Back_Ccond_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * (1.0 - Enc_Back_Ccond_j.array().square());
			RowVectorXld dEnc_Back_I_j = dEnc_Back_C_j.array() * Enc_Back_Ccond_j.array() * Enc_Back_I_j.array() * (1.0 - Enc_Back_I_j.array());
			RowVectorXld dEnc_Back_F_j = dEnc_Back_C_j.array() * Enc_Back_C_j_l.array() * Enc_Back_F_j.array() * (1.0 - Enc_Back_F_j.array());

			RowVectorXld dEnc_Back_Gates_j(4 * this->encoder_->Common_Hidden_size);
			dEnc_Back_Gates_j << dEnc_Back_F_j, dEnc_Back_I_j, dEnc_Back_Ccond_j, dEnc_Back_O_j;

			MatrixXld DW_Enc_Back_j = this->encoder_->Backward.Input_states[Number_InputState].row(j).transpose() * dEnc_Back_Gates_j;
			MatrixXld DU_Enc_Back_j;
			if (j == 0) {
				DU_Enc_Back_j = MatrixXld::Zero(this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
			}
			else {
				DU_Enc_Back_j = this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Back_Gates_j;
			}
			RowVectorXld DB_Enc_Back_j = dEnc_Back_Gates_j;

			Enc_Back__dC_j = dEnc_Back_C_j.array() * Enc_Back_F_j.array();
			MatrixXld U_enc_b(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);
			U_enc_b << this->encoder_->Backward.U_F, this->encoder_->Backward.U_I, this->encoder_->Backward.U_C, this->encoder_->Backward.U_O;
			Enc_Back__dH_j = dEnc_Back_Gates_j * U_enc_b.transpose();

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
	return grads;
}

Seq2SeqWithAttention_ForTrain::grads_Seq2SeqWithAttention Seq2SeqWithAttention_ForTrain::BackwardWithLogging(size_t Number_InputState, MatrixXld Y_True_) {
	MatrixXld Y_True(Y_True_.rows() + this->decoder_->end_token.rows(), Y_True_.cols());
	Y_True.transpose() << Y_True_.transpose(), this->decoder_->end_token.transpose();
	auto check_nan_inf = [](const MatrixXld& m, const std::string& name) {
		if (!m.allFinite()) {
			auto lyambda = [](const MatrixXld& m) {
				int nan_count = 0;
				int inf_count = 0;

				for (int i = 0; i < m.size(); ++i) {
					double val = *(m.data() + i);
					if (std::isnan(val)) ++nan_count;
					if (std::isinf(val)) ++inf_count;
				}

				return std::make_pair(nan_count, inf_count);
				};
			//size_t nnan = 0;
			//size_t ninf = 0;
			auto [nan_count, inf_count] = lyambda(m);
			std::cerr << "[ERROR] NaN or Inf detected in: " << name << "\tnan-inf: " << nan_count << "/" << inf_count << "\n";// << "[DEBUG] : " << m << "\n";
			std::abort;
		} 
		};
	grads_Seq2SeqWithAttention grads;
	grads.SetZero(this);
	
	Eigen::Index T = std::min(this->GetOutputs()[Number_InputState].rows(), Y_True.rows());
	Eigen::Index N = this->encoder_->Common_Hidden_states[Number_InputState].rows();

	RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
	RowVectorXld _dS_t = RowVectorXld::Zero(this->decoder_->Hidden_size);

	RowVectorXld dY_ = RowVectorXld::Zero(Y_True.cols());
	for (int64_t t = static_cast<int64_t>(T) - 1; t >= 0; t--) {
		RowVectorXld Y_t = this->GetOutputs()[Number_InputState].row(t);
		RowVectorXld Y_true_t = Y_True.row(t);
		RowVectorXld dY_t = Y_t - Y_true_t; //Y_t - Y_true
		dY_ += dY_t.array().abs().matrix();
		check_nan_inf(Y_True.row(t), "Y_True");
		check_nan_inf(this->GetOutputs()[Number_InputState].row(t), "Y_t");
		MatrixXld DW_out_t = dY_t.transpose() * this->decoder_->StatesForgrads.p__[Number_InputState].row(t);
		//RowVectorXld dp__t = this->decoder_->W_Output.transpose() * dY_t;
		RowVectorXld dp_proj = dY_t * this->decoder_->W_Output;
		RowVectorXld DB_out_t = dY_t;

		RowVectorXld d_p_ = dp_proj.array() * this->decoder_->layernorm_gamma.array();
		RowVectorXld DGamma_t = dp_proj.array() * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).array();
		RowVectorXld DBeta_t = dp_proj;

		const RowVectorXld& x_hat = this->decoder_->StatesForgrads.p_[Number_InputState].row(t); // = p_
		double eps = 1e-5;
		double var = (x_hat.array().square().mean()); // Потому что x̂ уже центрирован: mean = 0
		double stddev = std::sqrt(var + eps);

		// скалярные средние
		double mean_dxhat = d_p_.mean();
		double mean_dxhat_xhat = (d_p_.array() * x_hat.array()).mean();

		// полный градиент по входу p
		RowVectorXld dX = (d_p_.array() - mean_dxhat - x_hat.array() * mean_dxhat_xhat) / stddev;

		RowVectorXld dS_t = dX.head(this->decoder_->Hidden_size) + _dS_t;
		RowVectorXld dContext_t = dX.tail(this->decoder_->attention_->duo_encoder_hidden_size_);

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

		RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (1.0 - O_t.array());
		RowVectorXld dC_t = _dC_t.array() + dS_t.array() * O_t.array() * (1.0 - ActivationFunctions::Tanh(C_t).array().square());
		RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (1.0 - Ccond_t.array().square());
		RowVectorXld dI_t = dC_t.array() * Ccond_t.array() * I_t.array() * (1.0 - I_t.array());
		RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (1.0 - F_t.array());

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
		RowVectorXld DB_dec_t = dGates_t;

		_dC_t = dC_t.array() * F_t.array();
		MatrixXld U(this->decoder_->Hidden_size, 4 * this->decoder_->Hidden_size);
		U << this->decoder_->U_F, this->decoder_->U_I, this->decoder_->U_C, this->decoder_->U_O;
		_dS_t = dGates_t * U.transpose();

		MatrixXld W(this->decoder_->Input_size, 4 * this->decoder_->Hidden_size);
		W << this->decoder_->W_F, this->decoder_->W_I, this->decoder_->W_C, this->decoder_->W_O;
		dContext_t += (dGates_t * W.transpose()).tail(this->decoder_->attention_->duo_encoder_hidden_size_);

		std::vector<RowVectorXld> _dH_Back;
		RowVectorXld Enc_Forw__dC_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
		RowVectorXld Enc_Forw__dH_j = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);

		check_nan_inf(F_t, "F_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(I_t, "I_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(C_t, "C_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(O_t, "O_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(Ccond_t, "Ccond_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(this->decoder_->StatesForgrads.x[Number_InputState].row(t).transpose(), "X_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dGates_t, "dGates_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dY_t, "dY_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(this->decoder_->StatesForgrads.p_[Number_InputState].row(t).transpose(), "p__t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(this->decoder_->W_Output, "W_Output" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(this->decoder_->B_Output, "B_Output" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		
		check_nan_inf(d_p_, "d_p_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dS_t, "dS_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dContext_t, "dContext_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dF_t, "dF_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dI_t, "dI_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dC_t, "dC_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dO_t, "dO_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(dCcond_t, "dCcond_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));

		check_nan_inf(DW_out_t, "DW_out_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(DB_out_t, "DB_out_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(DGamma_t, "DGamma_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(DBeta_t, "DBeta_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(DW_dec_t, "DW_dec_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(DU_dec_t, "DU_dec_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));
		check_nan_inf(DB_dec_t, "DB_dec_t_" + std::to_string(t) + "\tstep : " + std::to_string(Number_InputState));

		for (int64_t j = static_cast<int64_t>(N) - 1; j >= 0; j--) {
			double dE_tj = 0.0;

			const VectorXld& alpha = this->decoder_->StatesForgrads.all_alpha[Number_InputState][t];
			if (Number_InputState >= this->decoder_->StatesForgrads.all_u.size()) {
				throw std::runtime_error("Number_InputState out of range");
			}
			if (t >= this->decoder_->StatesForgrads.all_u[Number_InputState].size()) {
				throw std::runtime_error("t out of range in all_u");
			}
			if (j >= this->decoder_->StatesForgrads.all_u[Number_InputState][t].size()) {
				throw std::runtime_error("j out of range in all_u[t]");
			}

			RowVectorXld u_tj = this->decoder_->StatesForgrads.all_u[Number_InputState][t][j];


			RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
			RowVectorXld s_t_1 = this->decoder_->StatesForgrads.h[Number_InputState].row(t == 0 ? 0 : t - 1);

			for (int k = 0; k < N; ++k) {
				RowVectorXld h_k = this->encoder_->Common_Hidden_states[Number_InputState].row(k);

				double alpha_k = alpha(k);
				if (!std::isfinite(alpha_k)) {
					std::cerr << "[WARN] alpha(" << k << ") = " << alpha_k << " is not finite at t=" << t << ", j=" << j << "\n";
					continue;
				}
				if (!h_k.allFinite()) {
					std::cerr << "[WARN] h_k (k=" << k << ") is not finite at t=" << t << ", j=" << j << "\n";
				}

				double dAlpha_k = dContext_t.dot(h_k);
				if (!std::isfinite(dAlpha_k)) {
					std::cerr << "[WARN] dAlpha_k is not finite (dot of dContext_t and h_k) at t=" << t << ", j=" << j << ", k=" << k << "\n";
				}

				double delta = (j == k) ? 1.0 : 0.0;
				dE_tj += dAlpha_k * alpha_k * (delta - alpha(j));
			}

			if (!std::isfinite(dE_tj)) {
				std::cerr << "[ERROR] dE_tj is NaN or Inf at t=" << t << ", j=" << j << "\n";
			}

			RowVectorXld dU_tj = dE_tj * this->decoder_->attention_->attention_vector_.transpose();  // [1 x A]
			RowVectorXld dPreact_tj = dU_tj.array() * (1.0 - u_tj.array().square());


			MatrixXld DW_att_enc_tj = dPreact_tj.transpose() * h_j;
			MatrixXld DW_att_dec_tj = dPreact_tj.transpose() * s_t_1;
			MatrixXld DV_att_tj = u_tj.transpose() * dE_tj;

			RowVectorXld dH_j = alpha(j) * dContext_t + dPreact_tj * this->decoder_->attention_->W_encoder_;

			RowVectorXld dS_att_j = dPreact_tj * this->decoder_->attention_->W_decoder_;

			_dS_t += dS_att_j;

			RowVectorXld dH_forw_j = dH_j.leftCols(this->encoder_->Common_Hidden_size);
			dH_forw_j += Enc_Forw__dH_j;
			RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);

			_dH_Back.insert(_dH_Back.begin(), dH_back_j);

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

			RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (1.0 - Enc_Forw_O_j.array());
			RowVectorXld dEnc_Forw_C_j = Enc_Forw__dC_j.array() + dH_forw_j.array() * Enc_Forw_O_j.array() *
				(1.0 - ActivationFunctions::Tanh(Enc_Forw_C_j).array().square());
			RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (1.0 - Enc_Forw_Ccond_j.array().square());
			RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_Ccond_j.array() * Enc_Forw_I_j.array() * (1.0 - Enc_Forw_I_j.array());
			RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (1.0 - Enc_Forw_F_j.array());

			RowVectorXld dEnc_Forw_Gates_j(4 * this->encoder_->Common_Hidden_size);
			dEnc_Forw_Gates_j << dEnc_Forw_F_j, dEnc_Forw_I_j, dEnc_Forw_Ccond_j, dEnc_Forw_O_j;

			MatrixXld DW_Enc_Forw_j = this->encoder_->Forward.Input_states[Number_InputState].row(j).transpose() * dEnc_Forw_Gates_j;
			MatrixXld DU_Enc_Forw_j;
			if (j == 0) {
				DU_Enc_Forw_j = MatrixXld::Zero(this->encoder_->Forward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
			}
			else {
				DU_Enc_Forw_j = this->encoder_->Forward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Forw_Gates_j;
			}
			RowVectorXld DB_Enc_Forw_j = dEnc_Forw_Gates_j;

			Enc_Forw__dC_j = dEnc_Forw_C_j.array() * Enc_Forw_F_j.array();
			MatrixXld U_enc_f(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);
			U_enc_f << this->encoder_->Forward.U_F, this->encoder_->Forward.U_I, this->encoder_->Forward.U_C, this->encoder_->Forward.U_O;
			Enc_Forw__dH_j = dEnc_Forw_Gates_j * U_enc_f.transpose();

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
		
			check_nan_inf(dU_tj, "dU_tj_" + std::to_string(t) + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dPreact_tj, "dPreact_tj_" + std::to_string(t) + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dH_forw_j, "dH_forw_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dH_back_j, "dH_back_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Forw_F_j, "dEnc_Forw_F_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Forw_I_j, "dEnc_Forw_I_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Forw_C_j, "dEnc_Forw_C_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Forw_Ccond_j, "dEnc_Forw_Ccond_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Forw_O_j, "dEnc_Forw_O_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));

			check_nan_inf(DW_att_enc_tj, "DW_att_enc_tj_" + std::to_string(t) + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DW_att_dec_tj, "DW_att_dec_tj_" + std::to_string(t) + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DV_att_tj, "DV_att_tj_" + std::to_string(t) + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DW_Enc_Forw_j, "DW_Enc_Forw_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DU_Enc_Forw_j, "DU_Enc_Forw_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DB_Enc_Forw_j, "DB_Enc_Forw_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
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

			RowVectorXld dEnc_Back_O_j = dH_Back_j.array() * ActivationFunctions::Tanh(Enc_Back_C_j).array() * Enc_Back_O_j.array() * (1.0 - Enc_Back_O_j.array());
			RowVectorXld dEnc_Back_C_j = Enc_Back__dC_j.array() + dH_Back_j.array() * Enc_Back_O_j.array() *
				(1.0 - ActivationFunctions::Tanh(Enc_Back_C_j).array().square());
			RowVectorXld dEnc_Back_Ccond_j = dEnc_Back_C_j.array() * Enc_Back_I_j.array() * (1.0 - Enc_Back_Ccond_j.array().square());
			RowVectorXld dEnc_Back_I_j = dEnc_Back_C_j.array() * Enc_Back_Ccond_j.array() * Enc_Back_I_j.array() * (1.0 - Enc_Back_I_j.array());
			RowVectorXld dEnc_Back_F_j = dEnc_Back_C_j.array() * Enc_Back_C_j_l.array() * Enc_Back_F_j.array() * (1.0 - Enc_Back_F_j.array());

			RowVectorXld dEnc_Back_Gates_j(4 * this->encoder_->Common_Hidden_size);
			dEnc_Back_Gates_j << dEnc_Back_F_j, dEnc_Back_I_j, dEnc_Back_Ccond_j, dEnc_Back_O_j;

			MatrixXld DW_Enc_Back_j = this->encoder_->Backward.Input_states[Number_InputState].row(j).transpose() * dEnc_Back_Gates_j;
			MatrixXld DU_Enc_Back_j;
			if (j == 0) {
				DU_Enc_Back_j = MatrixXld::Zero(this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder_->Common_Hidden_size);
			}
			else {
				DU_Enc_Back_j = this->encoder_->Backward.statesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Back_Gates_j;
			}
			RowVectorXld DB_Enc_Back_j = dEnc_Back_Gates_j;

			Enc_Back__dC_j = dEnc_Back_C_j.array() * Enc_Back_F_j.array();
			MatrixXld U_enc_b(this->encoder_->Common_Hidden_size, 4 * this->encoder_->Common_Hidden_size);
			U_enc_b << this->encoder_->Backward.U_F, this->encoder_->Backward.U_I, this->encoder_->Backward.U_C, this->encoder_->Backward.U_O;
			Enc_Back__dH_j = dEnc_Back_Gates_j * U_enc_b.transpose();

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

			check_nan_inf(dEnc_Back_F_j, "dEnc_Back_F_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Back_I_j, "dEnc_Back_I_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Back_C_j, "dEnc_Back_C_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Back_Ccond_j, "dEnc_Back_Ccond_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(dEnc_Back_O_j, "dEnc_Back_O_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));

			check_nan_inf(DW_Enc_Back_j, "DW_Enc_Back_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DU_Enc_Back_j, "DU_Enc_Back_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
			check_nan_inf(DB_Enc_Back_j, "DB_Enc_Back_j_" + std::to_string(j) + "\tstep : " + std::to_string(Number_InputState));
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
	std::cout << "dY_sum : " << dY_.sum() << std::endl;
	return grads;
}

void Seq2SeqWithAttention_ForTrain::UpdateAdamOptWithLogging
(
	const std::vector<std::vector<MatrixXld>>& Target_input_output, /*std::vector<MatrixXld> Target_output,*/
	size_t epochs, size_t optima_steps, size_t batch_size, std::string packname_forsave,
	double learning_rate, double epsilon,
	double beta1, double beta2
)
{
	auto get_global_norm = [](auto& grads)  {
		double total_sq_norm = 0.0;

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

		double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		return global_norm;
		};

	auto clip_by_global_norm = [](auto& grads, double clip_value) {
		double total_sq_norm = 0.0;

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

		double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		if (global_norm > clip_value) {
			double scale = clip_value / global_norm;

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

	auto get_mean_grads = [](auto & grads) {
			double total_norm = 0.0;

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

			double mean = total_norm / 43;
			return mean;
			};

	std::vector<std::vector<MatrixXld>> shuffle_target;
	double notceil_batch_steps_ = (double)Target_input_output.size() / batch_size;
	size_t batch_steps_ = (size_t)std::ceil(notceil_batch_steps_);

	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point end_time;

	for (size_t epoch_ = 0; epoch_ < epochs; epoch_++) {
		grads_Seq2SeqWithAttention grads_start_avg_train_loss;
		grads_Seq2SeqWithAttention grads_end_avg_train_loss;
		grads_start_avg_train_loss.SetZero(this);
		grads_end_avg_train_loss.SetZero(this);

		start_time = std::chrono::steady_clock::now();

		shuffle_target = Target_input_output;

		std::random_device rd;
		std::shuffle(shuffle_target.begin(), shuffle_target.end(), std::mt19937(rd()));
		
		std::vector<std::vector<MatrixXld>> shuffle_(2, std::vector<MatrixXld>(shuffle_target.size()));
		for (int i = 0; i < shuffle_target.size(); i++) {
				shuffle_[0][i] = shuffle_target[i][0];
				shuffle_[1][i] = shuffle_target[i][1];
			}
		shuffle_target = shuffle_;

		double clip_threshold = 200L;


		Inference(shuffle_target[0]);
		for (size_t start_i = 0; start_i < Target_input_output.size(); start_i++) {
			grads_start_avg_train_loss += BackwardWithLogging(start_i, shuffle_target[1][start_i]);
		}
		grads_start_avg_train_loss /= Target_input_output.size();

		for (size_t batch_step = 0; batch_step < batch_steps_; batch_step++) {
			grads_Seq2SeqWithAttention grads;

			grads.SetZero(this);

			MatrixXld M_W_out = MatrixXld::Zero(grads.dW_out.rows(), grads.dW_out.cols());
			MatrixXld M_B_out = MatrixXld::Zero(grads.dB_out.rows(), grads.dB_out.cols());

			MatrixXld M_W_gamma_layernorm = MatrixXld::Zero(grads.dW_gamma_layernorm.rows(), grads.dW_gamma_layernorm.cols());
			MatrixXld M_B_beta_layernorm = MatrixXld::Zero(grads.dB_beta_layernorm.rows(), grads.dB_beta_layernorm.cols());

			MatrixXld M_V_a_attention = MatrixXld::Zero(grads.dV_a_attention.rows(), grads.dV_a_attention.cols());
			MatrixXld M_W_e_attention = MatrixXld::Zero(grads.dW_e_attention.rows(), grads.dW_e_attention.cols());
			MatrixXld M_W_d_attention = MatrixXld::Zero(grads.dW_d_attention.rows(), grads.dW_d_attention.cols());

			MatrixXld M_W_f_dec = MatrixXld::Zero(grads.dW_f_dec.rows(), grads.dW_f_dec.cols());
			MatrixXld M_U_f_dec = MatrixXld::Zero(grads.dU_f_dec.rows(), grads.dU_f_dec.cols());
			MatrixXld M_B_f_dec = MatrixXld::Zero(grads.dB_f_dec.rows(), grads.dB_f_dec.cols());

			MatrixXld M_W_i_dec = MatrixXld::Zero(grads.dW_i_dec.rows(), grads.dW_i_dec.cols());
			MatrixXld M_U_i_dec = MatrixXld::Zero(grads.dU_i_dec.rows(), grads.dU_i_dec.cols());
			MatrixXld M_B_i_dec = MatrixXld::Zero(grads.dB_i_dec.rows(), grads.dB_i_dec.cols());

			MatrixXld M_W_ccond_dec = MatrixXld::Zero(grads.dW_ccond_dec.rows(), grads.dW_ccond_dec.cols());
			MatrixXld M_U_ccond_dec = MatrixXld::Zero(grads.dU_ccond_dec.rows(), grads.dU_ccond_dec.cols());
			MatrixXld M_B_ccond_dec = MatrixXld::Zero(grads.dB_ccond_dec.rows(), grads.dB_ccond_dec.cols());

			MatrixXld M_W_o_dec = MatrixXld::Zero(grads.dW_o_dec.rows(), grads.dW_o_dec.cols());
			MatrixXld M_U_o_dec = MatrixXld::Zero(grads.dU_o_dec.rows(), grads.dU_o_dec.cols());
			MatrixXld M_B_o_dec = MatrixXld::Zero(grads.dB_o_dec.rows(), grads.dB_o_dec.cols());

			MatrixXld M_W_f_forw_enc = MatrixXld::Zero(grads.dW_f_forw_enc.rows(), grads.dW_f_forw_enc.cols());
			MatrixXld M_U_f_forw_enc = MatrixXld::Zero(grads.dU_f_forw_enc.rows(), grads.dU_f_forw_enc.cols());
			MatrixXld M_B_f_forw_enc = MatrixXld::Zero(grads.dB_f_forw_enc.rows(), grads.dB_f_forw_enc.cols());

			MatrixXld M_W_i_forw_enc = MatrixXld::Zero(grads.dW_i_forw_enc.rows(), grads.dW_i_forw_enc.cols());
			MatrixXld M_U_i_forw_enc = MatrixXld::Zero(grads.dU_i_forw_enc.rows(), grads.dU_i_forw_enc.cols());
			MatrixXld M_B_i_forw_enc = MatrixXld::Zero(grads.dB_i_forw_enc.rows(), grads.dB_i_forw_enc.cols());

			MatrixXld M_W_ccond_forw_enc = MatrixXld::Zero(grads.dW_ccond_forw_enc.rows(), grads.dW_ccond_forw_enc.cols());
			MatrixXld M_U_ccond_forw_enc = MatrixXld::Zero(grads.dU_ccond_forw_enc.rows(), grads.dU_ccond_forw_enc.cols());
			MatrixXld M_B_ccond_forw_enc = MatrixXld::Zero(grads.dB_ccond_forw_enc.rows(), grads.dB_ccond_forw_enc.cols());

			MatrixXld M_W_o_forw_enc = MatrixXld::Zero(grads.dW_o_forw_enc.rows(), grads.dW_o_forw_enc.cols());
			MatrixXld M_U_o_forw_enc = MatrixXld::Zero(grads.dU_o_forw_enc.rows(), grads.dU_o_forw_enc.cols());
			MatrixXld M_B_o_forw_enc = MatrixXld::Zero(grads.dB_o_forw_enc.rows(), grads.dB_o_forw_enc.cols());

			MatrixXld M_W_f_back_enc = MatrixXld::Zero(grads.dW_f_back_enc.rows(), grads.dW_f_back_enc.cols());
			MatrixXld M_U_f_back_enc = MatrixXld::Zero(grads.dU_f_back_enc.rows(), grads.dU_f_back_enc.cols());
			MatrixXld M_B_f_back_enc = MatrixXld::Zero(grads.dB_f_back_enc.rows(), grads.dB_f_back_enc.cols());

			MatrixXld M_W_i_back_enc = MatrixXld::Zero(grads.dW_i_back_enc.rows(), grads.dW_i_back_enc.cols());
			MatrixXld M_U_i_back_enc = MatrixXld::Zero(grads.dU_i_back_enc.rows(), grads.dU_i_back_enc.cols());
			MatrixXld M_B_i_back_enc = MatrixXld::Zero(grads.dB_i_back_enc.rows(), grads.dB_i_back_enc.cols());

			MatrixXld M_W_ccond_back_enc = MatrixXld::Zero(grads.dW_ccond_back_enc.rows(), grads.dW_ccond_back_enc.cols());
			MatrixXld M_U_ccond_back_enc = MatrixXld::Zero(grads.dU_ccond_back_enc.rows(), grads.dU_ccond_back_enc.cols());
			MatrixXld M_B_ccond_back_enc = MatrixXld::Zero(grads.dB_ccond_back_enc.rows(), grads.dB_ccond_back_enc.cols());

			MatrixXld M_W_o_back_enc = MatrixXld::Zero(grads.dW_o_back_enc.rows(), grads.dW_o_back_enc.cols());
			MatrixXld M_U_o_back_enc = MatrixXld::Zero(grads.dU_o_back_enc.rows(), grads.dU_o_back_enc.cols());
			MatrixXld M_B_o_back_enc = MatrixXld::Zero(grads.dB_o_back_enc.rows(), grads.dB_o_back_enc.cols());

			// -------- V_ блок --------

			MatrixXld V_W_out = MatrixXld::Zero(grads.dW_out.rows(), grads.dW_out.cols());
			MatrixXld V_B_out = MatrixXld::Zero(grads.dB_out.rows(), grads.dB_out.cols());

			MatrixXld V_W_gamma_layernorm = MatrixXld::Zero(grads.dW_gamma_layernorm.rows(), grads.dW_gamma_layernorm.cols());
			MatrixXld V_B_beta_layernorm = MatrixXld::Zero(grads.dB_beta_layernorm.rows(), grads.dB_beta_layernorm.cols());

			MatrixXld V_V_a_attention = MatrixXld::Zero(grads.dV_a_attention.rows(), grads.dV_a_attention.cols());
			MatrixXld V_W_e_attention = MatrixXld::Zero(grads.dW_e_attention.rows(), grads.dW_e_attention.cols());
			MatrixXld V_W_d_attention = MatrixXld::Zero(grads.dW_d_attention.rows(), grads.dW_d_attention.cols());

			MatrixXld V_W_f_dec = MatrixXld::Zero(grads.dW_f_dec.rows(), grads.dW_f_dec.cols());
			MatrixXld V_U_f_dec = MatrixXld::Zero(grads.dU_f_dec.rows(), grads.dU_f_dec.cols());
			MatrixXld V_B_f_dec = MatrixXld::Zero(grads.dB_f_dec.rows(), grads.dB_f_dec.cols());

			MatrixXld V_W_i_dec = MatrixXld::Zero(grads.dW_i_dec.rows(), grads.dW_i_dec.cols());
			MatrixXld V_U_i_dec = MatrixXld::Zero(grads.dU_i_dec.rows(), grads.dU_i_dec.cols());
			MatrixXld V_B_i_dec = MatrixXld::Zero(grads.dB_i_dec.rows(), grads.dB_i_dec.cols());

			MatrixXld V_W_ccond_dec = MatrixXld::Zero(grads.dW_ccond_dec.rows(), grads.dW_ccond_dec.cols());
			MatrixXld V_U_ccond_dec = MatrixXld::Zero(grads.dU_ccond_dec.rows(), grads.dU_ccond_dec.cols());
			MatrixXld V_B_ccond_dec = MatrixXld::Zero(grads.dB_ccond_dec.rows(), grads.dB_ccond_dec.cols());

			MatrixXld V_W_o_dec = MatrixXld::Zero(grads.dW_o_dec.rows(), grads.dW_o_dec.cols());
			MatrixXld V_U_o_dec = MatrixXld::Zero(grads.dU_o_dec.rows(), grads.dU_o_dec.cols());
			MatrixXld V_B_o_dec = MatrixXld::Zero(grads.dB_o_dec.rows(), grads.dB_o_dec.cols());

			MatrixXld V_W_f_forw_enc = MatrixXld::Zero(grads.dW_f_forw_enc.rows(), grads.dW_f_forw_enc.cols());
			MatrixXld V_U_f_forw_enc = MatrixXld::Zero(grads.dU_f_forw_enc.rows(), grads.dU_f_forw_enc.cols());
			MatrixXld V_B_f_forw_enc = MatrixXld::Zero(grads.dB_f_forw_enc.rows(), grads.dB_f_forw_enc.cols());

			MatrixXld V_W_i_forw_enc = MatrixXld::Zero(grads.dW_i_forw_enc.rows(), grads.dW_i_forw_enc.cols());
			MatrixXld V_U_i_forw_enc = MatrixXld::Zero(grads.dU_i_forw_enc.rows(), grads.dU_i_forw_enc.cols());
			MatrixXld V_B_i_forw_enc = MatrixXld::Zero(grads.dB_i_forw_enc.rows(), grads.dB_i_forw_enc.cols());

			MatrixXld V_W_ccond_forw_enc = MatrixXld::Zero(grads.dW_ccond_forw_enc.rows(), grads.dW_ccond_forw_enc.cols());
			MatrixXld V_U_ccond_forw_enc = MatrixXld::Zero(grads.dU_ccond_forw_enc.rows(), grads.dU_ccond_forw_enc.cols());
			MatrixXld V_B_ccond_forw_enc = MatrixXld::Zero(grads.dB_ccond_forw_enc.rows(), grads.dB_ccond_forw_enc.cols());

			MatrixXld V_W_o_forw_enc = MatrixXld::Zero(grads.dW_o_forw_enc.rows(), grads.dW_o_forw_enc.cols());
			MatrixXld V_U_o_forw_enc = MatrixXld::Zero(grads.dU_o_forw_enc.rows(), grads.dU_o_forw_enc.cols());
			MatrixXld V_B_o_forw_enc = MatrixXld::Zero(grads.dB_o_forw_enc.rows(), grads.dB_o_forw_enc.cols());

			MatrixXld V_W_f_back_enc = MatrixXld::Zero(grads.dW_f_back_enc.rows(), grads.dW_f_back_enc.cols());
			MatrixXld V_U_f_back_enc = MatrixXld::Zero(grads.dU_f_back_enc.rows(), grads.dU_f_back_enc.cols());
			MatrixXld V_B_f_back_enc = MatrixXld::Zero(grads.dB_f_back_enc.rows(), grads.dB_f_back_enc.cols());

			MatrixXld V_W_i_back_enc = MatrixXld::Zero(grads.dW_i_back_enc.rows(), grads.dW_i_back_enc.cols());
			MatrixXld V_U_i_back_enc = MatrixXld::Zero(grads.dU_i_back_enc.rows(), grads.dU_i_back_enc.cols());
			MatrixXld V_B_i_back_enc = MatrixXld::Zero(grads.dB_i_back_enc.rows(), grads.dB_i_back_enc.cols());

			MatrixXld V_W_ccond_back_enc = MatrixXld::Zero(grads.dW_ccond_back_enc.rows(), grads.dW_ccond_back_enc.cols());
			MatrixXld V_U_ccond_back_enc = MatrixXld::Zero(grads.dU_ccond_back_enc.rows(), grads.dU_ccond_back_enc.cols());
			MatrixXld V_B_ccond_back_enc = MatrixXld::Zero(grads.dB_ccond_back_enc.rows(), grads.dB_ccond_back_enc.cols());

			MatrixXld V_W_o_back_enc = MatrixXld::Zero(grads.dW_o_back_enc.rows(), grads.dW_o_back_enc.cols());
			MatrixXld V_U_o_back_enc = MatrixXld::Zero(grads.dU_o_back_enc.rows(), grads.dU_o_back_enc.cols());
			MatrixXld V_B_o_back_enc = MatrixXld::Zero(grads.dB_o_back_enc.rows(), grads.dB_o_back_enc.cols());

			for (size_t t_ = 0; t_ < optima_steps; t_++) {
				Inference(shuffle_target[0]);
				grads.SetZero(this);
				for (size_t i = batch_step * batch_size; i < (batch_step + 1) * batch_size && i < shuffle_target[0].size(); i++) {
					grads += BackwardWithLogging(i, shuffle_target[1][i]);
				}
				grads /= (batch_step == batch_steps_) ? batch_size * (notceil_batch_steps_ - (int)notceil_batch_steps_) : batch_size;
				
				double grad_norm = get_global_norm(grads);

				//clip_by_global_norm(grads, clip_threshold);

				if (!std::isfinite(grad_norm)) {
					auto check_nan_inf = [](const MatrixXld& m, const std::string& name) {
						if (!m.allFinite()) {
							auto lyambda = [](const MatrixXld& m) {
								int nan_count = 0;
								int inf_count = 0;

								for (int i = 0; i < m.size(); ++i) {
									double val = *(m.data() + i);
									if (std::isnan(val)) ++nan_count;
									else if (std::isinf(val)) ++inf_count;
								}

								return std::make_pair(nan_count, inf_count);
								};
							//size_t nnan = 0;
							//size_t ninf = 0;
							auto [nan_count, inf_count] = lyambda(m);
							std::cerr << "[ERROR] NaN or Inf detected in: " << name << "\tnan-inf: " << nan_count << "/" << inf_count << "\n";
						}
						};
					std::cerr << "[WARNING] NaN/inf in gradients at batch " << (batch_step + 1) << "\n";
					check_nan_inf(grads.dW_out, "grads.dW_out");
					check_nan_inf(grads.dB_out, "grads.dB_out");

					check_nan_inf(grads.dW_f_dec, "grads.dW_f_dec");
					check_nan_inf(grads.dU_f_dec, "grads.dU_f_dec");
					check_nan_inf(grads.dB_f_dec, "grads.dB_f_dec");

					check_nan_inf(grads.dW_i_dec, "grads.dW_i_dec");
					check_nan_inf(grads.dU_i_dec, "grads.dU_i_dec");
					check_nan_inf(grads.dB_i_dec, "grads.dB_i_dec");

					check_nan_inf(grads.dW_ccond_dec, "grads.dW_ccond_dec");
					check_nan_inf(grads.dU_ccond_dec, "grads.dU_ccond_dec");
					check_nan_inf(grads.dB_ccond_dec, "grads.dB_ccond_dec");

					check_nan_inf(grads.dW_o_dec, "grads.dW_o_dec");
					check_nan_inf(grads.dU_o_dec, "grads.dU_o_dec");
					check_nan_inf(grads.dB_o_dec, "grads.dB_o_dec");

					check_nan_inf(grads.dW_gamma_layernorm, "grads.dW_gamma_layernorm");
					check_nan_inf(grads.dB_beta_layernorm, "grads.dB_beta_layernorm");

					check_nan_inf(grads.dV_a_attention, "grads.dV_a_attention");
					check_nan_inf(grads.dW_e_attention, "grads.dW_e_attention");
					check_nan_inf(grads.dW_d_attention, "grads.dW_d_attention");

					check_nan_inf(grads.dW_f_forw_enc, "grads.dW_f_forw_enc");
					check_nan_inf(grads.dU_f_forw_enc, "grads.dU_f_forw_enc");
					check_nan_inf(grads.dB_f_forw_enc, "grads.dB_f_forw_enc");

					check_nan_inf(grads.dW_i_forw_enc, "grads.dW_i_forw_enc");
					check_nan_inf(grads.dU_i_forw_enc, "grads.dU_i_forw_enc");
					check_nan_inf(grads.dB_i_forw_enc, "grads.dB_i_forw_enc");

					check_nan_inf(grads.dW_ccond_forw_enc, "grads.dW_ccond_forw_enc");
					check_nan_inf(grads.dU_ccond_forw_enc, "grads.dU_ccond_forw_enc");
					check_nan_inf(grads.dB_ccond_forw_enc, "grads.dB_ccond_forw_enc");

					check_nan_inf(grads.dW_o_forw_enc, "grads.dW_o_forw_enc");
					check_nan_inf(grads.dU_o_forw_enc, "grads.dU_o_forw_enc");
					check_nan_inf(grads.dB_o_forw_enc, "grads.dB_o_forw_enc");

					check_nan_inf(grads.dW_f_back_enc, "grads.dW_f_back_enc");
					check_nan_inf(grads.dU_f_back_enc, "grads.dU_f_back_enc");
					check_nan_inf(grads.dB_f_back_enc, "grads.dB_f_back_enc");

					check_nan_inf(grads.dW_i_back_enc, "grads.dW_i_back_enc");
					check_nan_inf(grads.dU_i_back_enc, "grads.dU_i_back_enc");
					check_nan_inf(grads.dB_i_back_enc, "grads.dB_i_back_enc");

					check_nan_inf(grads.dW_ccond_back_enc, "grads.dW_ccond_back_enc");
					check_nan_inf(grads.dU_ccond_back_enc, "grads.dU_ccond_back_enc");
					check_nan_inf(grads.dB_ccond_back_enc, "grads.dB_ccond_back_enc");

					check_nan_inf(grads.dW_o_back_enc, "grads.dW_o_back_enc");
					check_nan_inf(grads.dU_o_back_enc, "grads.dU_o_back_enc");
					check_nan_inf(grads.dB_o_back_enc, "grads.dB_o_back_enc");
				}
				else if (grad_norm > clip_threshold) {
					std::cout << "[CLIP] Batch " << (batch_step + 1)
						<< " gradient norm = " << grad_norm << " clipped\n";
				}
				else {
					std::cout << "[INFO] Batch " << (batch_step + 1)
						<< " gradient norm = " << grad_norm << "\n";
				}
				std::cout << "Epoch : " << epoch_ << "  step_optimisation : " << t_ << std::endl;


				
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
				
				/////
				/////
				/////
				double bias_corr_1 = (1 - std::pow(beta1, optima_steps + 1));
				double bias_corr_2 = (1 - std::pow(beta2, optima_steps + 1));
				MatrixXld _M_W_out = M_W_out.array() /bias_corr_1;
				MatrixXld _M_B_out = M_B_out.array() /bias_corr_1;
				//
				MatrixXld _M_W_gamma_layernorm = M_W_gamma_layernorm.array() /bias_corr_1;
				MatrixXld _M_B_beta_layernorm = M_B_beta_layernorm.array() /bias_corr_1;
				//
				MatrixXld _M_V_a_attention = M_V_a_attention.array() /bias_corr_1;
				MatrixXld _M_W_e_attention = M_W_e_attention.array() /bias_corr_1;
				MatrixXld _M_W_d_attention = M_W_d_attention.array() /bias_corr_1;
				//
				MatrixXld _M_W_f_dec = M_W_f_dec.array() /bias_corr_1;
				MatrixXld _M_U_f_dec = M_U_f_dec.array() /bias_corr_1;
				MatrixXld _M_B_f_dec = M_B_f_dec.array() /bias_corr_1;

				MatrixXld _M_W_i_dec = M_W_i_dec.array() /bias_corr_1;
				MatrixXld _M_U_i_dec = M_U_i_dec.array() /bias_corr_1;
				MatrixXld _M_B_i_dec = M_B_i_dec.array() /bias_corr_1;

				MatrixXld _M_W_ccond_dec = M_W_ccond_dec.array() /bias_corr_1;
				MatrixXld _M_U_ccond_dec = M_U_ccond_dec.array() /bias_corr_1;
				MatrixXld _M_B_ccond_dec = M_B_ccond_dec.array() /bias_corr_1;

				MatrixXld _M_W_o_dec = M_W_o_dec.array() /bias_corr_1;
				MatrixXld _M_U_o_dec = M_U_o_dec.array() /bias_corr_1;
				MatrixXld _M_B_o_dec = M_B_o_dec.array() /bias_corr_1;
				//
				MatrixXld _M_W_f_forw_enc = M_W_f_forw_enc.array() /bias_corr_1;
				MatrixXld _M_U_f_forw_enc = M_U_f_forw_enc.array() /bias_corr_1;
				MatrixXld _M_B_f_forw_enc = M_B_f_forw_enc.array() /bias_corr_1;

				MatrixXld _M_W_i_forw_enc = M_W_i_forw_enc.array() /bias_corr_1;
				MatrixXld _M_U_i_forw_enc = M_U_i_forw_enc.array() /bias_corr_1;
				MatrixXld _M_B_i_forw_enc = M_B_i_forw_enc.array() /bias_corr_1;

				MatrixXld _M_W_ccond_forw_enc = M_W_ccond_forw_enc.array() /bias_corr_1;
				MatrixXld _M_U_ccond_forw_enc = M_U_ccond_forw_enc.array() /bias_corr_1;
				MatrixXld _M_B_ccond_forw_enc = M_B_ccond_forw_enc.array() /bias_corr_1;

				MatrixXld _M_W_o_forw_enc = M_W_o_forw_enc.array() /bias_corr_1;
				MatrixXld _M_U_o_forw_enc = M_U_o_forw_enc.array() /bias_corr_1;
				MatrixXld _M_B_o_forw_enc = M_B_o_forw_enc.array() /bias_corr_1;
				//				   
				MatrixXld _M_W_f_back_enc = M_W_f_back_enc.array() /bias_corr_1;
				MatrixXld _M_U_f_back_enc = M_U_f_back_enc.array() /bias_corr_1;
				MatrixXld _M_B_f_back_enc = M_B_f_back_enc.array() /bias_corr_1;

				MatrixXld _M_W_i_back_enc = M_W_i_back_enc.array() /bias_corr_1;
				MatrixXld _M_U_i_back_enc = M_U_i_back_enc.array() /bias_corr_1;
				MatrixXld _M_B_i_back_enc = M_B_i_back_enc.array() /bias_corr_1;

				MatrixXld _M_W_ccond_back_enc = M_W_ccond_back_enc.array() /bias_corr_1;
				MatrixXld _M_U_ccond_back_enc = M_U_ccond_back_enc.array() /bias_corr_1;
				MatrixXld _M_B_ccond_back_enc = M_B_ccond_back_enc.array() /bias_corr_1;

				MatrixXld _M_W_o_back_enc = M_W_o_back_enc.array() /bias_corr_1;
				MatrixXld _M_U_o_back_enc = M_U_o_back_enc.array() /bias_corr_1;
				MatrixXld _M_B_o_back_enc = M_B_o_back_enc.array() /bias_corr_1;
				//				  
				//				  
				MatrixXld _V_W_out = V_W_out.array() /bias_corr_2;
				MatrixXld _V_B_out = V_B_out.array() /bias_corr_2;
				//
				MatrixXld _V_W_gamma_layernorm = V_W_gamma_layernorm.array() /bias_corr_2;
				MatrixXld _V_B_beta_layernorm = V_B_beta_layernorm.array() /bias_corr_2;
				//
				MatrixXld _V_V_a_attention = V_V_a_attention.array() /bias_corr_2;
				MatrixXld _V_W_e_attention = V_W_e_attention.array() /bias_corr_2;
				MatrixXld _V_W_d_attention = V_W_d_attention.array() /bias_corr_2;
				//
				MatrixXld _V_W_f_dec = V_W_f_dec.array() /bias_corr_2;
				MatrixXld _V_U_f_dec = V_U_f_dec.array() /bias_corr_2;
				MatrixXld _V_B_f_dec = V_B_f_dec.array() /bias_corr_2;

				MatrixXld _V_W_i_dec = V_W_i_dec.array() /bias_corr_2;
				MatrixXld _V_U_i_dec = V_U_i_dec.array() /bias_corr_2;
				MatrixXld _V_B_i_dec = V_B_i_dec.array() /bias_corr_2;

				MatrixXld _V_W_ccond_dec = V_W_ccond_dec.array() /bias_corr_2;
				MatrixXld _V_U_ccond_dec = V_U_ccond_dec.array() /bias_corr_2;
				MatrixXld _V_B_ccond_dec = V_B_ccond_dec.array() /bias_corr_2;

				MatrixXld _V_W_o_dec = V_W_o_dec.array() /bias_corr_2;
				MatrixXld _V_U_o_dec = V_U_o_dec.array() /bias_corr_2;
				MatrixXld _V_B_o_dec = V_B_o_dec.array() /bias_corr_2;
				//
				MatrixXld _V_W_f_forw_enc = V_W_f_forw_enc.array() /bias_corr_2;
				MatrixXld _V_U_f_forw_enc = V_U_f_forw_enc.array() /bias_corr_2;
				MatrixXld _V_B_f_forw_enc = V_B_f_forw_enc.array() /bias_corr_2;

				MatrixXld _V_W_i_forw_enc = V_W_i_forw_enc.array() /bias_corr_2;
				MatrixXld _V_U_i_forw_enc = V_U_i_forw_enc.array() /bias_corr_2;
				MatrixXld _V_B_i_forw_enc = V_B_i_forw_enc.array() /bias_corr_2;

				MatrixXld _V_W_ccond_forw_enc = V_W_ccond_forw_enc.array() /bias_corr_2;
				MatrixXld _V_U_ccond_forw_enc = V_U_ccond_forw_enc.array() /bias_corr_2;
				MatrixXld _V_B_ccond_forw_enc = V_B_ccond_forw_enc.array() /bias_corr_2;

				MatrixXld _V_W_o_forw_enc = V_W_o_forw_enc.array() /bias_corr_2;
				MatrixXld _V_U_o_forw_enc = V_U_o_forw_enc.array() /bias_corr_2;
				MatrixXld _V_B_o_forw_enc = V_B_o_forw_enc.array() /bias_corr_2;
				//				   
				MatrixXld _V_W_f_back_enc = V_W_f_back_enc.array() /bias_corr_2;
				MatrixXld _V_U_f_back_enc = V_U_f_back_enc.array() /bias_corr_2;
				MatrixXld _V_B_f_back_enc = V_B_f_back_enc.array() /bias_corr_2;

				MatrixXld _V_W_i_back_enc = V_W_i_back_enc.array() /bias_corr_2;
				MatrixXld _V_U_i_back_enc = V_U_i_back_enc.array() /bias_corr_2;
				MatrixXld _V_B_i_back_enc = V_B_i_back_enc.array() /bias_corr_2;

				MatrixXld _V_W_ccond_back_enc = V_W_ccond_back_enc.array() /bias_corr_2;
				MatrixXld _V_U_ccond_back_enc = V_U_ccond_back_enc.array() /bias_corr_2;
				MatrixXld _V_B_ccond_back_enc = V_B_ccond_back_enc.array() /bias_corr_2;

				MatrixXld _V_W_o_back_enc = V_W_o_back_enc.array() /bias_corr_2;
				MatrixXld _V_U_o_back_enc = V_U_o_back_enc.array() /bias_corr_2;
				MatrixXld _V_B_o_back_enc = V_B_o_back_enc.array() /bias_corr_2;
				/////
				/////
				/////
				/////
				this->decoder_->W_Output.array() -= learning_rate * _M_W_out.array() / (_V_W_out.array().sqrt() + epsilon);
				this->decoder_->B_Output.array() -= learning_rate * _M_B_out.array() / (_V_B_out.array().sqrt() + epsilon);
				//
				this->decoder_->layernorm_gamma.array() -= learning_rate * _M_W_gamma_layernorm.array() / (_V_W_gamma_layernorm.array().sqrt() + epsilon);
				this->decoder_->layernorm_beta.array() -= learning_rate * _M_B_beta_layernorm.array() / (_V_B_beta_layernorm.array().sqrt() + epsilon);
				//
				this->decoder_->attention_->attention_vector_.array() -= learning_rate * _M_V_a_attention.array() / (_V_V_a_attention.array().sqrt() + epsilon);
				this->decoder_->attention_->W_encoder_.array() -= learning_rate * _M_W_e_attention.array() / (_V_W_e_attention.array().sqrt() + epsilon);
				this->decoder_->attention_->W_decoder_.array() -= learning_rate * _M_W_d_attention.array() / (_V_W_d_attention.array().sqrt() + epsilon);
				//
				this->decoder_->W_F.array() -= learning_rate * _M_W_f_dec.array() / (_V_W_f_dec.array().sqrt() + epsilon);
				this->decoder_->U_F.array() -= learning_rate * _M_U_f_dec.array() / (_V_U_f_dec.array().sqrt() + epsilon);
				this->decoder_->B_F.array() -= learning_rate * _M_B_f_dec.array() / (_V_B_f_dec.array().sqrt() + epsilon);
				
				this->decoder_->W_I.array() -= learning_rate * _M_W_i_dec.array() / (_V_W_i_dec.array().sqrt() + epsilon);
				this->decoder_->U_I.array() -= learning_rate * _M_U_i_dec.array() / (_V_U_i_dec.array().sqrt() + epsilon);
				this->decoder_->B_I.array() -= learning_rate * _M_B_i_dec.array() / (_V_B_i_dec.array().sqrt() + epsilon);
				
				this->decoder_->W_C.array() -= learning_rate * _M_W_ccond_dec.array() / (_V_W_ccond_dec.array().sqrt() + epsilon);
				this->decoder_->U_C.array() -= learning_rate * _M_U_ccond_dec.array() / (_V_U_ccond_dec.array().sqrt() + epsilon);
				this->decoder_->B_C.array() -= learning_rate * _M_B_ccond_dec.array() / (_V_B_ccond_dec.array().sqrt() + epsilon);
				
				this->decoder_->W_O.array() -= learning_rate * _M_W_o_dec.array() / (_V_W_o_dec.array().sqrt() + epsilon);
				this->decoder_->U_O.array() -= learning_rate * _M_U_o_dec.array() / (_V_U_o_dec.array().sqrt() + epsilon);
				this->decoder_->B_O.array() -= learning_rate * _M_B_o_dec.array() / (_V_B_o_dec.array().sqrt() + epsilon);

				//
				this->encoder_->Forward.W_F.array() -= learning_rate * _M_W_f_forw_enc.array() / (_V_W_f_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.U_F.array() -= learning_rate * _M_U_f_forw_enc.array() / (_V_U_f_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.B_F.array() -= learning_rate * _M_B_f_forw_enc.array() / (_V_B_f_forw_enc.array().sqrt() + epsilon);
									   
				this->encoder_->Forward.W_I.array() -= learning_rate * _M_W_i_forw_enc.array() / (_V_W_i_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.U_I.array() -= learning_rate * _M_U_i_forw_enc.array() / (_V_U_i_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.B_I.array() -= learning_rate * _M_B_i_forw_enc.array() / (_V_B_i_forw_enc.array().sqrt() + epsilon);
									   
				this->encoder_->Forward.W_C.array() -= learning_rate * _M_W_ccond_forw_enc.array() / (_V_W_ccond_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.U_C.array() -= learning_rate * _M_U_ccond_forw_enc.array() / (_V_U_ccond_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.B_C.array() -= learning_rate * _M_B_ccond_forw_enc.array() / (_V_B_ccond_forw_enc.array().sqrt() + epsilon);
									   
				this->encoder_->Forward.W_O.array() -= learning_rate * _M_W_o_forw_enc.array() / (_V_W_o_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.U_O.array() -= learning_rate * _M_U_o_forw_enc.array() / (_V_U_o_forw_enc.array().sqrt() + epsilon);
				this->encoder_->Forward.B_O.array() -= learning_rate * _M_B_o_forw_enc.array() / (_V_B_o_forw_enc.array().sqrt() + epsilon);
				//				   
				this->encoder_->Backward.W_F.array() -= learning_rate * _M_W_f_back_enc.array() / (_V_W_f_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.U_F.array() -= learning_rate * _M_U_f_back_enc.array() / (_V_U_f_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.B_F.array() -= learning_rate * _M_B_f_back_enc.array() / (_V_B_f_back_enc.array().sqrt() + epsilon);
										
				this->encoder_->Backward.W_I.array() -= learning_rate * _M_W_i_back_enc.array() / (_V_W_i_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.U_I.array() -= learning_rate * _M_U_i_back_enc.array() / (_V_U_i_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.B_I.array() -= learning_rate * _M_B_i_back_enc.array() / (_V_B_i_back_enc.array().sqrt() + epsilon);
										
				this->encoder_->Backward.W_C.array() -= learning_rate * _M_W_ccond_back_enc.array() / (_V_W_ccond_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.U_C.array() -= learning_rate * _M_U_ccond_back_enc.array() / (_V_U_ccond_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.B_C.array() -= learning_rate * _M_B_ccond_back_enc.array() / (_V_B_ccond_back_enc.array().sqrt() + epsilon);
										
				this->encoder_->Backward.W_O.array() -= learning_rate * _M_W_o_back_enc.array() / (_V_W_o_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.U_O.array() -= learning_rate * _M_U_o_back_enc.array() / (_V_U_o_back_enc.array().sqrt() + epsilon);
				this->encoder_->Backward.B_O.array() -= learning_rate * _M_B_o_back_enc.array() / (_V_B_o_back_enc.array().sqrt() + epsilon);
			}
		}

		Inference(shuffle_target[0]);
		for (size_t end_i = 0; end_i < Target_input_output.size(); end_i++) {
			grads_end_avg_train_loss += BackwardWithLogging(end_i, shuffle_target[1][end_i]);
		}
		grads_end_avg_train_loss /= Target_input_output.size();

		this->Save(packname_forsave);

		end_time = std::chrono::steady_clock::now();

		std::chrono::duration<double> elapsed = end_time - start_time;
	
		std::cout << "Epoch " << (epoch_ + 1)
			<< " finished. Avg train loss [start/end]: " 
			<< get_mean_grads(grads_start_avg_train_loss) << "/" << get_mean_grads(grads_end_avg_train_loss)
			//<< ", Val loss: " << val_loss
			<< ", Time_epoch: " << elapsed.count() << "s\n";
	}
}

void Seq2SeqWithAttention_ForTrain::UpdateAdamOptWithLogging
(
	const std::vector<std::vector<MatrixXld>>& Target_input_output, /*std::vector<MatrixXld> Target_output,*/
	size_t epochs, size_t optima_steps, std::string packname_forsave,
	double learning_rate, double epsilon,
	double beta1, double beta2
)
{
	auto get_global_norm = [](auto& grads) {
		double total_sq_norm = 0.0;

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

		double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		return global_norm;
		};

	auto clip_by_global_norm = [](auto& grads, double clip_value) {
		double total_sq_norm = 0.0;

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

		double global_norm = std::sqrt(total_sq_norm + 1e-8L);

		if (global_norm > clip_value) {
			double scale = clip_value / global_norm;

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

	auto get_mean_grads = [](auto& grads) {
			double total_norm = 0.0;

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

			double mean = total_norm / 43;
			return mean;
			};

	std::vector<std::vector<MatrixXld>> shuffle_target;

	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point end_time;

	for (size_t epoch_ = 0; epoch_ < epochs; epoch_++) {
		grads_Seq2SeqWithAttention grads_start_avg_train_loss;
		grads_Seq2SeqWithAttention grads_end_avg_train_loss;
		grads_start_avg_train_loss.SetZero(this);
		grads_end_avg_train_loss.SetZero(this);

		start_time = std::chrono::steady_clock::now();

		shuffle_target = Target_input_output;

		std::random_device rd;
		std::shuffle(shuffle_target.begin(), shuffle_target.end(), std::mt19937(rd()));

		std::vector<std::vector<MatrixXld>> shuffle_(2, std::vector<MatrixXld>(shuffle_target.size()));
		for (int i = 0; i < shuffle_target.size(); i++) {
			shuffle_[0][i] = shuffle_target[i][0];
			shuffle_[1][i] = shuffle_target[i][1];
		}
		shuffle_target = shuffle_;

		double clip_threshold = 200L;


		Inference(shuffle_target[0]);
		for (size_t start_i = 0; start_i < Target_input_output.size(); start_i++) {
			grads_start_avg_train_loss += BackwardWithLogging(start_i, shuffle_target[1][start_i]);
		}
		grads_start_avg_train_loss /= Target_input_output.size();

		size_t batch_steps_;
		for (double batch_size = Target_input_output.size(); batch_size > 0.5; batch_size /= 2) {
			batch_steps_ = Target_input_output.size() / std::ceil(batch_size);
			for (size_t batch_step = 0; batch_step < batch_steps_; batch_step++) {
				grads_Seq2SeqWithAttention grads;

				grads.SetZero(this);

				MatrixXld M_W_out = MatrixXld::Zero(grads.dW_out.rows(), grads.dW_out.cols());
				MatrixXld M_B_out = MatrixXld::Zero(grads.dB_out.rows(), grads.dB_out.cols());

				MatrixXld M_W_gamma_layernorm = MatrixXld::Zero(grads.dW_gamma_layernorm.rows(), grads.dW_gamma_layernorm.cols());
				MatrixXld M_B_beta_layernorm = MatrixXld::Zero(grads.dB_beta_layernorm.rows(), grads.dB_beta_layernorm.cols());

				MatrixXld M_V_a_attention = MatrixXld::Zero(grads.dV_a_attention.rows(), grads.dV_a_attention.cols());
				MatrixXld M_W_e_attention = MatrixXld::Zero(grads.dW_e_attention.rows(), grads.dW_e_attention.cols());
				MatrixXld M_W_d_attention = MatrixXld::Zero(grads.dW_d_attention.rows(), grads.dW_d_attention.cols());

				MatrixXld M_W_f_dec = MatrixXld::Zero(grads.dW_f_dec.rows(), grads.dW_f_dec.cols());
				MatrixXld M_U_f_dec = MatrixXld::Zero(grads.dU_f_dec.rows(), grads.dU_f_dec.cols());
				MatrixXld M_B_f_dec = MatrixXld::Zero(grads.dB_f_dec.rows(), grads.dB_f_dec.cols());

				MatrixXld M_W_i_dec = MatrixXld::Zero(grads.dW_i_dec.rows(), grads.dW_i_dec.cols());
				MatrixXld M_U_i_dec = MatrixXld::Zero(grads.dU_i_dec.rows(), grads.dU_i_dec.cols());
				MatrixXld M_B_i_dec = MatrixXld::Zero(grads.dB_i_dec.rows(), grads.dB_i_dec.cols());

				MatrixXld M_W_ccond_dec = MatrixXld::Zero(grads.dW_ccond_dec.rows(), grads.dW_ccond_dec.cols());
				MatrixXld M_U_ccond_dec = MatrixXld::Zero(grads.dU_ccond_dec.rows(), grads.dU_ccond_dec.cols());
				MatrixXld M_B_ccond_dec = MatrixXld::Zero(grads.dB_ccond_dec.rows(), grads.dB_ccond_dec.cols());

				MatrixXld M_W_o_dec = MatrixXld::Zero(grads.dW_o_dec.rows(), grads.dW_o_dec.cols());
				MatrixXld M_U_o_dec = MatrixXld::Zero(grads.dU_o_dec.rows(), grads.dU_o_dec.cols());
				MatrixXld M_B_o_dec = MatrixXld::Zero(grads.dB_o_dec.rows(), grads.dB_o_dec.cols());

				MatrixXld M_W_f_forw_enc = MatrixXld::Zero(grads.dW_f_forw_enc.rows(), grads.dW_f_forw_enc.cols());
				MatrixXld M_U_f_forw_enc = MatrixXld::Zero(grads.dU_f_forw_enc.rows(), grads.dU_f_forw_enc.cols());
				MatrixXld M_B_f_forw_enc = MatrixXld::Zero(grads.dB_f_forw_enc.rows(), grads.dB_f_forw_enc.cols());

				MatrixXld M_W_i_forw_enc = MatrixXld::Zero(grads.dW_i_forw_enc.rows(), grads.dW_i_forw_enc.cols());
				MatrixXld M_U_i_forw_enc = MatrixXld::Zero(grads.dU_i_forw_enc.rows(), grads.dU_i_forw_enc.cols());
				MatrixXld M_B_i_forw_enc = MatrixXld::Zero(grads.dB_i_forw_enc.rows(), grads.dB_i_forw_enc.cols());

				MatrixXld M_W_ccond_forw_enc = MatrixXld::Zero(grads.dW_ccond_forw_enc.rows(), grads.dW_ccond_forw_enc.cols());
				MatrixXld M_U_ccond_forw_enc = MatrixXld::Zero(grads.dU_ccond_forw_enc.rows(), grads.dU_ccond_forw_enc.cols());
				MatrixXld M_B_ccond_forw_enc = MatrixXld::Zero(grads.dB_ccond_forw_enc.rows(), grads.dB_ccond_forw_enc.cols());

				MatrixXld M_W_o_forw_enc = MatrixXld::Zero(grads.dW_o_forw_enc.rows(), grads.dW_o_forw_enc.cols());
				MatrixXld M_U_o_forw_enc = MatrixXld::Zero(grads.dU_o_forw_enc.rows(), grads.dU_o_forw_enc.cols());
				MatrixXld M_B_o_forw_enc = MatrixXld::Zero(grads.dB_o_forw_enc.rows(), grads.dB_o_forw_enc.cols());

				MatrixXld M_W_f_back_enc = MatrixXld::Zero(grads.dW_f_back_enc.rows(), grads.dW_f_back_enc.cols());
				MatrixXld M_U_f_back_enc = MatrixXld::Zero(grads.dU_f_back_enc.rows(), grads.dU_f_back_enc.cols());
				MatrixXld M_B_f_back_enc = MatrixXld::Zero(grads.dB_f_back_enc.rows(), grads.dB_f_back_enc.cols());

				MatrixXld M_W_i_back_enc = MatrixXld::Zero(grads.dW_i_back_enc.rows(), grads.dW_i_back_enc.cols());
				MatrixXld M_U_i_back_enc = MatrixXld::Zero(grads.dU_i_back_enc.rows(), grads.dU_i_back_enc.cols());
				MatrixXld M_B_i_back_enc = MatrixXld::Zero(grads.dB_i_back_enc.rows(), grads.dB_i_back_enc.cols());

				MatrixXld M_W_ccond_back_enc = MatrixXld::Zero(grads.dW_ccond_back_enc.rows(), grads.dW_ccond_back_enc.cols());
				MatrixXld M_U_ccond_back_enc = MatrixXld::Zero(grads.dU_ccond_back_enc.rows(), grads.dU_ccond_back_enc.cols());
				MatrixXld M_B_ccond_back_enc = MatrixXld::Zero(grads.dB_ccond_back_enc.rows(), grads.dB_ccond_back_enc.cols());

				MatrixXld M_W_o_back_enc = MatrixXld::Zero(grads.dW_o_back_enc.rows(), grads.dW_o_back_enc.cols());
				MatrixXld M_U_o_back_enc = MatrixXld::Zero(grads.dU_o_back_enc.rows(), grads.dU_o_back_enc.cols());
				MatrixXld M_B_o_back_enc = MatrixXld::Zero(grads.dB_o_back_enc.rows(), grads.dB_o_back_enc.cols());

				// -------- V_ блок --------

				MatrixXld V_W_out = MatrixXld::Zero(grads.dW_out.rows(), grads.dW_out.cols());
				MatrixXld V_B_out = MatrixXld::Zero(grads.dB_out.rows(), grads.dB_out.cols());

				MatrixXld V_W_gamma_layernorm = MatrixXld::Zero(grads.dW_gamma_layernorm.rows(), grads.dW_gamma_layernorm.cols());
				MatrixXld V_B_beta_layernorm = MatrixXld::Zero(grads.dB_beta_layernorm.rows(), grads.dB_beta_layernorm.cols());

				MatrixXld V_V_a_attention = MatrixXld::Zero(grads.dV_a_attention.rows(), grads.dV_a_attention.cols());
				MatrixXld V_W_e_attention = MatrixXld::Zero(grads.dW_e_attention.rows(), grads.dW_e_attention.cols());
				MatrixXld V_W_d_attention = MatrixXld::Zero(grads.dW_d_attention.rows(), grads.dW_d_attention.cols());

				MatrixXld V_W_f_dec = MatrixXld::Zero(grads.dW_f_dec.rows(), grads.dW_f_dec.cols());
				MatrixXld V_U_f_dec = MatrixXld::Zero(grads.dU_f_dec.rows(), grads.dU_f_dec.cols());
				MatrixXld V_B_f_dec = MatrixXld::Zero(grads.dB_f_dec.rows(), grads.dB_f_dec.cols());

				MatrixXld V_W_i_dec = MatrixXld::Zero(grads.dW_i_dec.rows(), grads.dW_i_dec.cols());
				MatrixXld V_U_i_dec = MatrixXld::Zero(grads.dU_i_dec.rows(), grads.dU_i_dec.cols());
				MatrixXld V_B_i_dec = MatrixXld::Zero(grads.dB_i_dec.rows(), grads.dB_i_dec.cols());

				MatrixXld V_W_ccond_dec = MatrixXld::Zero(grads.dW_ccond_dec.rows(), grads.dW_ccond_dec.cols());
				MatrixXld V_U_ccond_dec = MatrixXld::Zero(grads.dU_ccond_dec.rows(), grads.dU_ccond_dec.cols());
				MatrixXld V_B_ccond_dec = MatrixXld::Zero(grads.dB_ccond_dec.rows(), grads.dB_ccond_dec.cols());

				MatrixXld V_W_o_dec = MatrixXld::Zero(grads.dW_o_dec.rows(), grads.dW_o_dec.cols());
				MatrixXld V_U_o_dec = MatrixXld::Zero(grads.dU_o_dec.rows(), grads.dU_o_dec.cols());
				MatrixXld V_B_o_dec = MatrixXld::Zero(grads.dB_o_dec.rows(), grads.dB_o_dec.cols());

				MatrixXld V_W_f_forw_enc = MatrixXld::Zero(grads.dW_f_forw_enc.rows(), grads.dW_f_forw_enc.cols());
				MatrixXld V_U_f_forw_enc = MatrixXld::Zero(grads.dU_f_forw_enc.rows(), grads.dU_f_forw_enc.cols());
				MatrixXld V_B_f_forw_enc = MatrixXld::Zero(grads.dB_f_forw_enc.rows(), grads.dB_f_forw_enc.cols());

				MatrixXld V_W_i_forw_enc = MatrixXld::Zero(grads.dW_i_forw_enc.rows(), grads.dW_i_forw_enc.cols());
				MatrixXld V_U_i_forw_enc = MatrixXld::Zero(grads.dU_i_forw_enc.rows(), grads.dU_i_forw_enc.cols());
				MatrixXld V_B_i_forw_enc = MatrixXld::Zero(grads.dB_i_forw_enc.rows(), grads.dB_i_forw_enc.cols());

				MatrixXld V_W_ccond_forw_enc = MatrixXld::Zero(grads.dW_ccond_forw_enc.rows(), grads.dW_ccond_forw_enc.cols());
				MatrixXld V_U_ccond_forw_enc = MatrixXld::Zero(grads.dU_ccond_forw_enc.rows(), grads.dU_ccond_forw_enc.cols());
				MatrixXld V_B_ccond_forw_enc = MatrixXld::Zero(grads.dB_ccond_forw_enc.rows(), grads.dB_ccond_forw_enc.cols());

				MatrixXld V_W_o_forw_enc = MatrixXld::Zero(grads.dW_o_forw_enc.rows(), grads.dW_o_forw_enc.cols());
				MatrixXld V_U_o_forw_enc = MatrixXld::Zero(grads.dU_o_forw_enc.rows(), grads.dU_o_forw_enc.cols());
				MatrixXld V_B_o_forw_enc = MatrixXld::Zero(grads.dB_o_forw_enc.rows(), grads.dB_o_forw_enc.cols());

				MatrixXld V_W_f_back_enc = MatrixXld::Zero(grads.dW_f_back_enc.rows(), grads.dW_f_back_enc.cols());
				MatrixXld V_U_f_back_enc = MatrixXld::Zero(grads.dU_f_back_enc.rows(), grads.dU_f_back_enc.cols());
				MatrixXld V_B_f_back_enc = MatrixXld::Zero(grads.dB_f_back_enc.rows(), grads.dB_f_back_enc.cols());

				MatrixXld V_W_i_back_enc = MatrixXld::Zero(grads.dW_i_back_enc.rows(), grads.dW_i_back_enc.cols());
				MatrixXld V_U_i_back_enc = MatrixXld::Zero(grads.dU_i_back_enc.rows(), grads.dU_i_back_enc.cols());
				MatrixXld V_B_i_back_enc = MatrixXld::Zero(grads.dB_i_back_enc.rows(), grads.dB_i_back_enc.cols());

				MatrixXld V_W_ccond_back_enc = MatrixXld::Zero(grads.dW_ccond_back_enc.rows(), grads.dW_ccond_back_enc.cols());
				MatrixXld V_U_ccond_back_enc = MatrixXld::Zero(grads.dU_ccond_back_enc.rows(), grads.dU_ccond_back_enc.cols());
				MatrixXld V_B_ccond_back_enc = MatrixXld::Zero(grads.dB_ccond_back_enc.rows(), grads.dB_ccond_back_enc.cols());

				MatrixXld V_W_o_back_enc = MatrixXld::Zero(grads.dW_o_back_enc.rows(), grads.dW_o_back_enc.cols());
				MatrixXld V_U_o_back_enc = MatrixXld::Zero(grads.dU_o_back_enc.rows(), grads.dU_o_back_enc.cols());
				MatrixXld V_B_o_back_enc = MatrixXld::Zero(grads.dB_o_back_enc.rows(), grads.dB_o_back_enc.cols());

				for (size_t t_ = 0; t_ < optima_steps; t_++) {
					Inference(shuffle_target[0]);
					grads.SetZero(this);
					for (size_t i = batch_step * std::ceil(batch_size); i < (batch_step + 1) * std::ceil(batch_size) && i < shuffle_target[0].size(); i++) {
						grads += BackwardWithLogging(i, shuffle_target[1][i]);
					}
					if (shuffle_target[0].size() % (size_t)std::ceil(batch_size) == 0 || batch_step != batch_steps_) {
						grads /= std::ceil(batch_size);
					}
					else {
						grads /= shuffle_target[0].size() % (size_t)std::ceil(batch_size);
					}


					double grad_norm = get_global_norm(grads);

					//clip_by_global_norm(grads, clip_threshold);

					if (!std::isfinite(grad_norm)) {
						auto check_nan_inf = [](const MatrixXld& m, const std::string& name) {
							if (!m.allFinite()) {
								auto lyambda = [](const MatrixXld& m) {
									int nan_count = 0;
									int inf_count = 0;

									for (int i = 0; i < m.size(); ++i) {
										double val = *(m.data() + i);
										if (std::isnan(val)) ++nan_count;
										else if (std::isinf(val)) ++inf_count;
									}

									return std::make_pair(nan_count, inf_count);
									};
								//size_t nnan = 0;
								//size_t ninf = 0;
								auto [nan_count, inf_count] = lyambda(m);
								std::cerr << "[ERROR] NaN or Inf detected in: " << name << "\tnan-inf: " << nan_count << "/" << inf_count << "\n";
							}
							};
						std::cerr << "[WARNING] NaN/inf in gradients at batch " << batch_step << "\n";
						check_nan_inf(grads.dW_out, "grads.dW_out");
						check_nan_inf(grads.dB_out, "grads.dB_out");

						check_nan_inf(grads.dW_f_dec, "grads.dW_f_dec");
						check_nan_inf(grads.dU_f_dec, "grads.dU_f_dec");
						check_nan_inf(grads.dB_f_dec, "grads.dB_f_dec");

						check_nan_inf(grads.dW_i_dec, "grads.dW_i_dec");
						check_nan_inf(grads.dU_i_dec, "grads.dU_i_dec");
						check_nan_inf(grads.dB_i_dec, "grads.dB_i_dec");

						check_nan_inf(grads.dW_ccond_dec, "grads.dW_ccond_dec");
						check_nan_inf(grads.dU_ccond_dec, "grads.dU_ccond_dec");
						check_nan_inf(grads.dB_ccond_dec, "grads.dB_ccond_dec");

						check_nan_inf(grads.dW_o_dec, "grads.dW_o_dec");
						check_nan_inf(grads.dU_o_dec, "grads.dU_o_dec");
						check_nan_inf(grads.dB_o_dec, "grads.dB_o_dec");

						check_nan_inf(grads.dW_gamma_layernorm, "grads.dW_gamma_layernorm");
						check_nan_inf(grads.dB_beta_layernorm, "grads.dB_beta_layernorm");

						check_nan_inf(grads.dV_a_attention, "grads.dV_a_attention");
						check_nan_inf(grads.dW_e_attention, "grads.dW_e_attention");
						check_nan_inf(grads.dW_d_attention, "grads.dW_d_attention");

						check_nan_inf(grads.dW_f_forw_enc, "grads.dW_f_forw_enc");
						check_nan_inf(grads.dU_f_forw_enc, "grads.dU_f_forw_enc");
						check_nan_inf(grads.dB_f_forw_enc, "grads.dB_f_forw_enc");

						check_nan_inf(grads.dW_i_forw_enc, "grads.dW_i_forw_enc");
						check_nan_inf(grads.dU_i_forw_enc, "grads.dU_i_forw_enc");
						check_nan_inf(grads.dB_i_forw_enc, "grads.dB_i_forw_enc");

						check_nan_inf(grads.dW_ccond_forw_enc, "grads.dW_ccond_forw_enc");
						check_nan_inf(grads.dU_ccond_forw_enc, "grads.dU_ccond_forw_enc");
						check_nan_inf(grads.dB_ccond_forw_enc, "grads.dB_ccond_forw_enc");

						check_nan_inf(grads.dW_o_forw_enc, "grads.dW_o_forw_enc");
						check_nan_inf(grads.dU_o_forw_enc, "grads.dU_o_forw_enc");
						check_nan_inf(grads.dB_o_forw_enc, "grads.dB_o_forw_enc");

						check_nan_inf(grads.dW_f_back_enc, "grads.dW_f_back_enc");
						check_nan_inf(grads.dU_f_back_enc, "grads.dU_f_back_enc");
						check_nan_inf(grads.dB_f_back_enc, "grads.dB_f_back_enc");

						check_nan_inf(grads.dW_i_back_enc, "grads.dW_i_back_enc");
						check_nan_inf(grads.dU_i_back_enc, "grads.dU_i_back_enc");
						check_nan_inf(grads.dB_i_back_enc, "grads.dB_i_back_enc");

						check_nan_inf(grads.dW_ccond_back_enc, "grads.dW_ccond_back_enc");
						check_nan_inf(grads.dU_ccond_back_enc, "grads.dU_ccond_back_enc");
						check_nan_inf(grads.dB_ccond_back_enc, "grads.dB_ccond_back_enc");

						check_nan_inf(grads.dW_o_back_enc, "grads.dW_o_back_enc");
						check_nan_inf(grads.dU_o_back_enc, "grads.dU_o_back_enc");
						check_nan_inf(grads.dB_o_back_enc, "grads.dB_o_back_enc");
					}
					else if (grad_norm > clip_threshold) {
						std::cout << "[CLIP] Batch " << batch_step
							<< " gradient norm = " << grad_norm << " clipped\n";
					}
					else {
						std::cout << "[INFO] Batch " << batch_step
							<< " gradient norm = " << grad_norm << "\n";
					}
					std::cout << "Epoch : " << epoch_ << "  step_optimisation : " << t_ << std::endl;



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

					/////
					/////
					/////
					double bias_corr_1 = (1 - std::pow(beta1, optima_steps + 1));
					double bias_corr_2 = (1 - std::pow(beta2, optima_steps + 1));
					MatrixXld _M_W_out = M_W_out.array() / bias_corr_1;
					MatrixXld _M_B_out = M_B_out.array() / bias_corr_1;
					//
					MatrixXld _M_W_gamma_layernorm = M_W_gamma_layernorm.array() / bias_corr_1;
					MatrixXld _M_B_beta_layernorm = M_B_beta_layernorm.array() / bias_corr_1;
					//
					MatrixXld _M_V_a_attention = M_V_a_attention.array() / bias_corr_1;
					MatrixXld _M_W_e_attention = M_W_e_attention.array() / bias_corr_1;
					MatrixXld _M_W_d_attention = M_W_d_attention.array() / bias_corr_1;
					//
					MatrixXld _M_W_f_dec = M_W_f_dec.array() / bias_corr_1;
					MatrixXld _M_U_f_dec = M_U_f_dec.array() / bias_corr_1;
					MatrixXld _M_B_f_dec = M_B_f_dec.array() / bias_corr_1;

					MatrixXld _M_W_i_dec = M_W_i_dec.array() / bias_corr_1;
					MatrixXld _M_U_i_dec = M_U_i_dec.array() / bias_corr_1;
					MatrixXld _M_B_i_dec = M_B_i_dec.array() / bias_corr_1;

					MatrixXld _M_W_ccond_dec = M_W_ccond_dec.array() / bias_corr_1;
					MatrixXld _M_U_ccond_dec = M_U_ccond_dec.array() / bias_corr_1;
					MatrixXld _M_B_ccond_dec = M_B_ccond_dec.array() / bias_corr_1;

					MatrixXld _M_W_o_dec = M_W_o_dec.array() / bias_corr_1;
					MatrixXld _M_U_o_dec = M_U_o_dec.array() / bias_corr_1;
					MatrixXld _M_B_o_dec = M_B_o_dec.array() / bias_corr_1;
					//
					MatrixXld _M_W_f_forw_enc = M_W_f_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_f_forw_enc = M_U_f_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_f_forw_enc = M_B_f_forw_enc.array() / bias_corr_1;

					MatrixXld _M_W_i_forw_enc = M_W_i_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_i_forw_enc = M_U_i_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_i_forw_enc = M_B_i_forw_enc.array() / bias_corr_1;

					MatrixXld _M_W_ccond_forw_enc = M_W_ccond_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_ccond_forw_enc = M_U_ccond_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_ccond_forw_enc = M_B_ccond_forw_enc.array() / bias_corr_1;

					MatrixXld _M_W_o_forw_enc = M_W_o_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_o_forw_enc = M_U_o_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_o_forw_enc = M_B_o_forw_enc.array() / bias_corr_1;
					//				   
					MatrixXld _M_W_f_back_enc = M_W_f_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_f_back_enc = M_U_f_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_f_back_enc = M_B_f_back_enc.array() / bias_corr_1;

					MatrixXld _M_W_i_back_enc = M_W_i_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_i_back_enc = M_U_i_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_i_back_enc = M_B_i_back_enc.array() / bias_corr_1;

					MatrixXld _M_W_ccond_back_enc = M_W_ccond_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_ccond_back_enc = M_U_ccond_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_ccond_back_enc = M_B_ccond_back_enc.array() / bias_corr_1;

					MatrixXld _M_W_o_back_enc = M_W_o_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_o_back_enc = M_U_o_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_o_back_enc = M_B_o_back_enc.array() / bias_corr_1;
					//				  
					//				  
					MatrixXld _V_W_out = V_W_out.array() / bias_corr_2;
					MatrixXld _V_B_out = V_B_out.array() / bias_corr_2;
					//
					MatrixXld _V_W_gamma_layernorm = V_W_gamma_layernorm.array() / bias_corr_2;
					MatrixXld _V_B_beta_layernorm = V_B_beta_layernorm.array() / bias_corr_2;
					//
					MatrixXld _V_V_a_attention = V_V_a_attention.array() / bias_corr_2;
					MatrixXld _V_W_e_attention = V_W_e_attention.array() / bias_corr_2;
					MatrixXld _V_W_d_attention = V_W_d_attention.array() / bias_corr_2;
					//
					MatrixXld _V_W_f_dec = V_W_f_dec.array() / bias_corr_2;
					MatrixXld _V_U_f_dec = V_U_f_dec.array() / bias_corr_2;
					MatrixXld _V_B_f_dec = V_B_f_dec.array() / bias_corr_2;

					MatrixXld _V_W_i_dec = V_W_i_dec.array() / bias_corr_2;
					MatrixXld _V_U_i_dec = V_U_i_dec.array() / bias_corr_2;
					MatrixXld _V_B_i_dec = V_B_i_dec.array() / bias_corr_2;

					MatrixXld _V_W_ccond_dec = V_W_ccond_dec.array() / bias_corr_2;
					MatrixXld _V_U_ccond_dec = V_U_ccond_dec.array() / bias_corr_2;
					MatrixXld _V_B_ccond_dec = V_B_ccond_dec.array() / bias_corr_2;

					MatrixXld _V_W_o_dec = V_W_o_dec.array() / bias_corr_2;
					MatrixXld _V_U_o_dec = V_U_o_dec.array() / bias_corr_2;
					MatrixXld _V_B_o_dec = V_B_o_dec.array() / bias_corr_2;
					//
					MatrixXld _V_W_f_forw_enc = V_W_f_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_f_forw_enc = V_U_f_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_f_forw_enc = V_B_f_forw_enc.array() / bias_corr_2;

					MatrixXld _V_W_i_forw_enc = V_W_i_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_i_forw_enc = V_U_i_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_i_forw_enc = V_B_i_forw_enc.array() / bias_corr_2;

					MatrixXld _V_W_ccond_forw_enc = V_W_ccond_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_ccond_forw_enc = V_U_ccond_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_ccond_forw_enc = V_B_ccond_forw_enc.array() / bias_corr_2;

					MatrixXld _V_W_o_forw_enc = V_W_o_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_o_forw_enc = V_U_o_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_o_forw_enc = V_B_o_forw_enc.array() / bias_corr_2;
					//				   
					MatrixXld _V_W_f_back_enc = V_W_f_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_f_back_enc = V_U_f_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_f_back_enc = V_B_f_back_enc.array() / bias_corr_2;

					MatrixXld _V_W_i_back_enc = V_W_i_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_i_back_enc = V_U_i_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_i_back_enc = V_B_i_back_enc.array() / bias_corr_2;

					MatrixXld _V_W_ccond_back_enc = V_W_ccond_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_ccond_back_enc = V_U_ccond_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_ccond_back_enc = V_B_ccond_back_enc.array() / bias_corr_2;

					MatrixXld _V_W_o_back_enc = V_W_o_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_o_back_enc = V_U_o_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_o_back_enc = V_B_o_back_enc.array() / bias_corr_2;
					/////
					/////
					/////
					/////
					this->decoder_->W_Output.array() -= learning_rate * _M_W_out.array() / (_V_W_out.array().sqrt() + epsilon);
					this->decoder_->B_Output.array() -= learning_rate * _M_B_out.array() / (_V_B_out.array().sqrt() + epsilon);
					//
					this->decoder_->layernorm_gamma.array() -= learning_rate * _M_W_gamma_layernorm.array() / (_V_W_gamma_layernorm.array().sqrt() + epsilon);
					this->decoder_->layernorm_beta.array() -= learning_rate * _M_B_beta_layernorm.array() / (_V_B_beta_layernorm.array().sqrt() + epsilon);
					//
					this->decoder_->attention_->attention_vector_.array() -= learning_rate * _M_V_a_attention.array() / (_V_V_a_attention.array().sqrt() + epsilon);
					this->decoder_->attention_->W_encoder_.array() -= learning_rate * _M_W_e_attention.array() / (_V_W_e_attention.array().sqrt() + epsilon);
					this->decoder_->attention_->W_decoder_.array() -= learning_rate * _M_W_d_attention.array() / (_V_W_d_attention.array().sqrt() + epsilon);
					//
					this->decoder_->W_F.array() -= learning_rate * _M_W_f_dec.array() / (_V_W_f_dec.array().sqrt() + epsilon);
					this->decoder_->U_F.array() -= learning_rate * _M_U_f_dec.array() / (_V_U_f_dec.array().sqrt() + epsilon);
					this->decoder_->B_F.array() -= learning_rate * _M_B_f_dec.array() / (_V_B_f_dec.array().sqrt() + epsilon);

					this->decoder_->W_I.array() -= learning_rate * _M_W_i_dec.array() / (_V_W_i_dec.array().sqrt() + epsilon);
					this->decoder_->U_I.array() -= learning_rate * _M_U_i_dec.array() / (_V_U_i_dec.array().sqrt() + epsilon);
					this->decoder_->B_I.array() -= learning_rate * _M_B_i_dec.array() / (_V_B_i_dec.array().sqrt() + epsilon);

					this->decoder_->W_C.array() -= learning_rate * _M_W_ccond_dec.array() / (_V_W_ccond_dec.array().sqrt() + epsilon);
					this->decoder_->U_C.array() -= learning_rate * _M_U_ccond_dec.array() / (_V_U_ccond_dec.array().sqrt() + epsilon);
					this->decoder_->B_C.array() -= learning_rate * _M_B_ccond_dec.array() / (_V_B_ccond_dec.array().sqrt() + epsilon);

					this->decoder_->W_O.array() -= learning_rate * _M_W_o_dec.array() / (_V_W_o_dec.array().sqrt() + epsilon);
					this->decoder_->U_O.array() -= learning_rate * _M_U_o_dec.array() / (_V_U_o_dec.array().sqrt() + epsilon);
					this->decoder_->B_O.array() -= learning_rate * _M_B_o_dec.array() / (_V_B_o_dec.array().sqrt() + epsilon);

					//
					this->encoder_->Forward.W_F.array() -= learning_rate * _M_W_f_forw_enc.array() / (_V_W_f_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_F.array() -= learning_rate * _M_U_f_forw_enc.array() / (_V_U_f_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_F.array() -= learning_rate * _M_B_f_forw_enc.array() / (_V_B_f_forw_enc.array().sqrt() + epsilon);

					this->encoder_->Forward.W_I.array() -= learning_rate * _M_W_i_forw_enc.array() / (_V_W_i_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_I.array() -= learning_rate * _M_U_i_forw_enc.array() / (_V_U_i_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_I.array() -= learning_rate * _M_B_i_forw_enc.array() / (_V_B_i_forw_enc.array().sqrt() + epsilon);

					this->encoder_->Forward.W_C.array() -= learning_rate * _M_W_ccond_forw_enc.array() / (_V_W_ccond_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_C.array() -= learning_rate * _M_U_ccond_forw_enc.array() / (_V_U_ccond_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_C.array() -= learning_rate * _M_B_ccond_forw_enc.array() / (_V_B_ccond_forw_enc.array().sqrt() + epsilon);

					this->encoder_->Forward.W_O.array() -= learning_rate * _M_W_o_forw_enc.array() / (_V_W_o_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_O.array() -= learning_rate * _M_U_o_forw_enc.array() / (_V_U_o_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_O.array() -= learning_rate * _M_B_o_forw_enc.array() / (_V_B_o_forw_enc.array().sqrt() + epsilon);
					//				   
					this->encoder_->Backward.W_F.array() -= learning_rate * _M_W_f_back_enc.array() / (_V_W_f_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_F.array() -= learning_rate * _M_U_f_back_enc.array() / (_V_U_f_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_F.array() -= learning_rate * _M_B_f_back_enc.array() / (_V_B_f_back_enc.array().sqrt() + epsilon);

					this->encoder_->Backward.W_I.array() -= learning_rate * _M_W_i_back_enc.array() / (_V_W_i_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_I.array() -= learning_rate * _M_U_i_back_enc.array() / (_V_U_i_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_I.array() -= learning_rate * _M_B_i_back_enc.array() / (_V_B_i_back_enc.array().sqrt() + epsilon);

					this->encoder_->Backward.W_C.array() -= learning_rate * _M_W_ccond_back_enc.array() / (_V_W_ccond_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_C.array() -= learning_rate * _M_U_ccond_back_enc.array() / (_V_U_ccond_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_C.array() -= learning_rate * _M_B_ccond_back_enc.array() / (_V_B_ccond_back_enc.array().sqrt() + epsilon);

					this->encoder_->Backward.W_O.array() -= learning_rate * _M_W_o_back_enc.array() / (_V_W_o_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_O.array() -= learning_rate * _M_U_o_back_enc.array() / (_V_U_o_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_O.array() -= learning_rate * _M_B_o_back_enc.array() / (_V_B_o_back_enc.array().sqrt() + epsilon);
				}
			}
		}

		{
			size_t batch_size = Target_input_output.size();
			batch_steps_ = 1;
			for (size_t batch_step = 0; batch_step < batch_steps_; batch_step++) {
				grads_Seq2SeqWithAttention grads;

				grads.SetZero(this);

				MatrixXld M_W_out = MatrixXld::Zero(grads.dW_out.rows(), grads.dW_out.cols());
				MatrixXld M_B_out = MatrixXld::Zero(grads.dB_out.rows(), grads.dB_out.cols());

				MatrixXld M_W_gamma_layernorm = MatrixXld::Zero(grads.dW_gamma_layernorm.rows(), grads.dW_gamma_layernorm.cols());
				MatrixXld M_B_beta_layernorm = MatrixXld::Zero(grads.dB_beta_layernorm.rows(), grads.dB_beta_layernorm.cols());

				MatrixXld M_V_a_attention = MatrixXld::Zero(grads.dV_a_attention.rows(), grads.dV_a_attention.cols());
				MatrixXld M_W_e_attention = MatrixXld::Zero(grads.dW_e_attention.rows(), grads.dW_e_attention.cols());
				MatrixXld M_W_d_attention = MatrixXld::Zero(grads.dW_d_attention.rows(), grads.dW_d_attention.cols());

				MatrixXld M_W_f_dec = MatrixXld::Zero(grads.dW_f_dec.rows(), grads.dW_f_dec.cols());
				MatrixXld M_U_f_dec = MatrixXld::Zero(grads.dU_f_dec.rows(), grads.dU_f_dec.cols());
				MatrixXld M_B_f_dec = MatrixXld::Zero(grads.dB_f_dec.rows(), grads.dB_f_dec.cols());

				MatrixXld M_W_i_dec = MatrixXld::Zero(grads.dW_i_dec.rows(), grads.dW_i_dec.cols());
				MatrixXld M_U_i_dec = MatrixXld::Zero(grads.dU_i_dec.rows(), grads.dU_i_dec.cols());
				MatrixXld M_B_i_dec = MatrixXld::Zero(grads.dB_i_dec.rows(), grads.dB_i_dec.cols());

				MatrixXld M_W_ccond_dec = MatrixXld::Zero(grads.dW_ccond_dec.rows(), grads.dW_ccond_dec.cols());
				MatrixXld M_U_ccond_dec = MatrixXld::Zero(grads.dU_ccond_dec.rows(), grads.dU_ccond_dec.cols());
				MatrixXld M_B_ccond_dec = MatrixXld::Zero(grads.dB_ccond_dec.rows(), grads.dB_ccond_dec.cols());

				MatrixXld M_W_o_dec = MatrixXld::Zero(grads.dW_o_dec.rows(), grads.dW_o_dec.cols());
				MatrixXld M_U_o_dec = MatrixXld::Zero(grads.dU_o_dec.rows(), grads.dU_o_dec.cols());
				MatrixXld M_B_o_dec = MatrixXld::Zero(grads.dB_o_dec.rows(), grads.dB_o_dec.cols());

				MatrixXld M_W_f_forw_enc = MatrixXld::Zero(grads.dW_f_forw_enc.rows(), grads.dW_f_forw_enc.cols());
				MatrixXld M_U_f_forw_enc = MatrixXld::Zero(grads.dU_f_forw_enc.rows(), grads.dU_f_forw_enc.cols());
				MatrixXld M_B_f_forw_enc = MatrixXld::Zero(grads.dB_f_forw_enc.rows(), grads.dB_f_forw_enc.cols());

				MatrixXld M_W_i_forw_enc = MatrixXld::Zero(grads.dW_i_forw_enc.rows(), grads.dW_i_forw_enc.cols());
				MatrixXld M_U_i_forw_enc = MatrixXld::Zero(grads.dU_i_forw_enc.rows(), grads.dU_i_forw_enc.cols());
				MatrixXld M_B_i_forw_enc = MatrixXld::Zero(grads.dB_i_forw_enc.rows(), grads.dB_i_forw_enc.cols());

				MatrixXld M_W_ccond_forw_enc = MatrixXld::Zero(grads.dW_ccond_forw_enc.rows(), grads.dW_ccond_forw_enc.cols());
				MatrixXld M_U_ccond_forw_enc = MatrixXld::Zero(grads.dU_ccond_forw_enc.rows(), grads.dU_ccond_forw_enc.cols());
				MatrixXld M_B_ccond_forw_enc = MatrixXld::Zero(grads.dB_ccond_forw_enc.rows(), grads.dB_ccond_forw_enc.cols());

				MatrixXld M_W_o_forw_enc = MatrixXld::Zero(grads.dW_o_forw_enc.rows(), grads.dW_o_forw_enc.cols());
				MatrixXld M_U_o_forw_enc = MatrixXld::Zero(grads.dU_o_forw_enc.rows(), grads.dU_o_forw_enc.cols());
				MatrixXld M_B_o_forw_enc = MatrixXld::Zero(grads.dB_o_forw_enc.rows(), grads.dB_o_forw_enc.cols());

				MatrixXld M_W_f_back_enc = MatrixXld::Zero(grads.dW_f_back_enc.rows(), grads.dW_f_back_enc.cols());
				MatrixXld M_U_f_back_enc = MatrixXld::Zero(grads.dU_f_back_enc.rows(), grads.dU_f_back_enc.cols());
				MatrixXld M_B_f_back_enc = MatrixXld::Zero(grads.dB_f_back_enc.rows(), grads.dB_f_back_enc.cols());

				MatrixXld M_W_i_back_enc = MatrixXld::Zero(grads.dW_i_back_enc.rows(), grads.dW_i_back_enc.cols());
				MatrixXld M_U_i_back_enc = MatrixXld::Zero(grads.dU_i_back_enc.rows(), grads.dU_i_back_enc.cols());
				MatrixXld M_B_i_back_enc = MatrixXld::Zero(grads.dB_i_back_enc.rows(), grads.dB_i_back_enc.cols());

				MatrixXld M_W_ccond_back_enc = MatrixXld::Zero(grads.dW_ccond_back_enc.rows(), grads.dW_ccond_back_enc.cols());
				MatrixXld M_U_ccond_back_enc = MatrixXld::Zero(grads.dU_ccond_back_enc.rows(), grads.dU_ccond_back_enc.cols());
				MatrixXld M_B_ccond_back_enc = MatrixXld::Zero(grads.dB_ccond_back_enc.rows(), grads.dB_ccond_back_enc.cols());

				MatrixXld M_W_o_back_enc = MatrixXld::Zero(grads.dW_o_back_enc.rows(), grads.dW_o_back_enc.cols());
				MatrixXld M_U_o_back_enc = MatrixXld::Zero(grads.dU_o_back_enc.rows(), grads.dU_o_back_enc.cols());
				MatrixXld M_B_o_back_enc = MatrixXld::Zero(grads.dB_o_back_enc.rows(), grads.dB_o_back_enc.cols());

				// -------- V_ блок --------

				MatrixXld V_W_out = MatrixXld::Zero(grads.dW_out.rows(), grads.dW_out.cols());
				MatrixXld V_B_out = MatrixXld::Zero(grads.dB_out.rows(), grads.dB_out.cols());

				MatrixXld V_W_gamma_layernorm = MatrixXld::Zero(grads.dW_gamma_layernorm.rows(), grads.dW_gamma_layernorm.cols());
				MatrixXld V_B_beta_layernorm = MatrixXld::Zero(grads.dB_beta_layernorm.rows(), grads.dB_beta_layernorm.cols());

				MatrixXld V_V_a_attention = MatrixXld::Zero(grads.dV_a_attention.rows(), grads.dV_a_attention.cols());
				MatrixXld V_W_e_attention = MatrixXld::Zero(grads.dW_e_attention.rows(), grads.dW_e_attention.cols());
				MatrixXld V_W_d_attention = MatrixXld::Zero(grads.dW_d_attention.rows(), grads.dW_d_attention.cols());

				MatrixXld V_W_f_dec = MatrixXld::Zero(grads.dW_f_dec.rows(), grads.dW_f_dec.cols());
				MatrixXld V_U_f_dec = MatrixXld::Zero(grads.dU_f_dec.rows(), grads.dU_f_dec.cols());
				MatrixXld V_B_f_dec = MatrixXld::Zero(grads.dB_f_dec.rows(), grads.dB_f_dec.cols());

				MatrixXld V_W_i_dec = MatrixXld::Zero(grads.dW_i_dec.rows(), grads.dW_i_dec.cols());
				MatrixXld V_U_i_dec = MatrixXld::Zero(grads.dU_i_dec.rows(), grads.dU_i_dec.cols());
				MatrixXld V_B_i_dec = MatrixXld::Zero(grads.dB_i_dec.rows(), grads.dB_i_dec.cols());

				MatrixXld V_W_ccond_dec = MatrixXld::Zero(grads.dW_ccond_dec.rows(), grads.dW_ccond_dec.cols());
				MatrixXld V_U_ccond_dec = MatrixXld::Zero(grads.dU_ccond_dec.rows(), grads.dU_ccond_dec.cols());
				MatrixXld V_B_ccond_dec = MatrixXld::Zero(grads.dB_ccond_dec.rows(), grads.dB_ccond_dec.cols());

				MatrixXld V_W_o_dec = MatrixXld::Zero(grads.dW_o_dec.rows(), grads.dW_o_dec.cols());
				MatrixXld V_U_o_dec = MatrixXld::Zero(grads.dU_o_dec.rows(), grads.dU_o_dec.cols());
				MatrixXld V_B_o_dec = MatrixXld::Zero(grads.dB_o_dec.rows(), grads.dB_o_dec.cols());

				MatrixXld V_W_f_forw_enc = MatrixXld::Zero(grads.dW_f_forw_enc.rows(), grads.dW_f_forw_enc.cols());
				MatrixXld V_U_f_forw_enc = MatrixXld::Zero(grads.dU_f_forw_enc.rows(), grads.dU_f_forw_enc.cols());
				MatrixXld V_B_f_forw_enc = MatrixXld::Zero(grads.dB_f_forw_enc.rows(), grads.dB_f_forw_enc.cols());

				MatrixXld V_W_i_forw_enc = MatrixXld::Zero(grads.dW_i_forw_enc.rows(), grads.dW_i_forw_enc.cols());
				MatrixXld V_U_i_forw_enc = MatrixXld::Zero(grads.dU_i_forw_enc.rows(), grads.dU_i_forw_enc.cols());
				MatrixXld V_B_i_forw_enc = MatrixXld::Zero(grads.dB_i_forw_enc.rows(), grads.dB_i_forw_enc.cols());

				MatrixXld V_W_ccond_forw_enc = MatrixXld::Zero(grads.dW_ccond_forw_enc.rows(), grads.dW_ccond_forw_enc.cols());
				MatrixXld V_U_ccond_forw_enc = MatrixXld::Zero(grads.dU_ccond_forw_enc.rows(), grads.dU_ccond_forw_enc.cols());
				MatrixXld V_B_ccond_forw_enc = MatrixXld::Zero(grads.dB_ccond_forw_enc.rows(), grads.dB_ccond_forw_enc.cols());

				MatrixXld V_W_o_forw_enc = MatrixXld::Zero(grads.dW_o_forw_enc.rows(), grads.dW_o_forw_enc.cols());
				MatrixXld V_U_o_forw_enc = MatrixXld::Zero(grads.dU_o_forw_enc.rows(), grads.dU_o_forw_enc.cols());
				MatrixXld V_B_o_forw_enc = MatrixXld::Zero(grads.dB_o_forw_enc.rows(), grads.dB_o_forw_enc.cols());

				MatrixXld V_W_f_back_enc = MatrixXld::Zero(grads.dW_f_back_enc.rows(), grads.dW_f_back_enc.cols());
				MatrixXld V_U_f_back_enc = MatrixXld::Zero(grads.dU_f_back_enc.rows(), grads.dU_f_back_enc.cols());
				MatrixXld V_B_f_back_enc = MatrixXld::Zero(grads.dB_f_back_enc.rows(), grads.dB_f_back_enc.cols());

				MatrixXld V_W_i_back_enc = MatrixXld::Zero(grads.dW_i_back_enc.rows(), grads.dW_i_back_enc.cols());
				MatrixXld V_U_i_back_enc = MatrixXld::Zero(grads.dU_i_back_enc.rows(), grads.dU_i_back_enc.cols());
				MatrixXld V_B_i_back_enc = MatrixXld::Zero(grads.dB_i_back_enc.rows(), grads.dB_i_back_enc.cols());

				MatrixXld V_W_ccond_back_enc = MatrixXld::Zero(grads.dW_ccond_back_enc.rows(), grads.dW_ccond_back_enc.cols());
				MatrixXld V_U_ccond_back_enc = MatrixXld::Zero(grads.dU_ccond_back_enc.rows(), grads.dU_ccond_back_enc.cols());
				MatrixXld V_B_ccond_back_enc = MatrixXld::Zero(grads.dB_ccond_back_enc.rows(), grads.dB_ccond_back_enc.cols());

				MatrixXld V_W_o_back_enc = MatrixXld::Zero(grads.dW_o_back_enc.rows(), grads.dW_o_back_enc.cols());
				MatrixXld V_U_o_back_enc = MatrixXld::Zero(grads.dU_o_back_enc.rows(), grads.dU_o_back_enc.cols());
				MatrixXld V_B_o_back_enc = MatrixXld::Zero(grads.dB_o_back_enc.rows(), grads.dB_o_back_enc.cols());

				for (size_t t_ = 0; t_ < optima_steps; t_++) {
					Inference(shuffle_target[0]);
					grads.SetZero(this);
					for (size_t i = batch_step * batch_size; i < (batch_step + 1) * batch_size && i < shuffle_target[0].size(); i++) {
						grads += BackwardWithLogging(i, shuffle_target[1][i]);
					}
					if (shuffle_target[0].size() % batch_size == 0 || batch_step != batch_steps_) {
						grads /= batch_size;
					}
					else {
						grads /= shuffle_target[0].size() % batch_size;
					}


					double grad_norm = get_global_norm(grads);

					//clip_by_global_norm(grads, clip_threshold);

					if (!std::isfinite(grad_norm)) {
						auto check_nan_inf = [](const MatrixXld& m, const std::string& name) {
							if (!m.allFinite()) {
								auto lyambda = [](const MatrixXld& m) {
									int nan_count = 0;
									int inf_count = 0;

									for (int i = 0; i < m.size(); ++i) {
										double val = *(m.data() + i);
										if (std::isnan(val)) ++nan_count;
										else if (std::isinf(val)) ++inf_count;
									}

									return std::make_pair(nan_count, inf_count);
									};
								//size_t nnan = 0;
								//size_t ninf = 0;
								auto [nan_count, inf_count] = lyambda(m);
								std::cerr << "[ERROR] NaN or Inf detected in: " << name << "\tnan-inf: " << nan_count << "/" << inf_count << "\n";
							}
							};
						std::cerr << "[WARNING] NaN/inf in gradients at batch " << (batch_step + 1) << "\n";
						check_nan_inf(grads.dW_out, "grads.dW_out");
						check_nan_inf(grads.dB_out, "grads.dB_out");

						check_nan_inf(grads.dW_f_dec, "grads.dW_f_dec");
						check_nan_inf(grads.dU_f_dec, "grads.dU_f_dec");
						check_nan_inf(grads.dB_f_dec, "grads.dB_f_dec");

						check_nan_inf(grads.dW_i_dec, "grads.dW_i_dec");
						check_nan_inf(grads.dU_i_dec, "grads.dU_i_dec");
						check_nan_inf(grads.dB_i_dec, "grads.dB_i_dec");

						check_nan_inf(grads.dW_ccond_dec, "grads.dW_ccond_dec");
						check_nan_inf(grads.dU_ccond_dec, "grads.dU_ccond_dec");
						check_nan_inf(grads.dB_ccond_dec, "grads.dB_ccond_dec");

						check_nan_inf(grads.dW_o_dec, "grads.dW_o_dec");
						check_nan_inf(grads.dU_o_dec, "grads.dU_o_dec");
						check_nan_inf(grads.dB_o_dec, "grads.dB_o_dec");

						check_nan_inf(grads.dW_gamma_layernorm, "grads.dW_gamma_layernorm");
						check_nan_inf(grads.dB_beta_layernorm, "grads.dB_beta_layernorm");

						check_nan_inf(grads.dV_a_attention, "grads.dV_a_attention");
						check_nan_inf(grads.dW_e_attention, "grads.dW_e_attention");
						check_nan_inf(grads.dW_d_attention, "grads.dW_d_attention");

						check_nan_inf(grads.dW_f_forw_enc, "grads.dW_f_forw_enc");
						check_nan_inf(grads.dU_f_forw_enc, "grads.dU_f_forw_enc");
						check_nan_inf(grads.dB_f_forw_enc, "grads.dB_f_forw_enc");

						check_nan_inf(grads.dW_i_forw_enc, "grads.dW_i_forw_enc");
						check_nan_inf(grads.dU_i_forw_enc, "grads.dU_i_forw_enc");
						check_nan_inf(grads.dB_i_forw_enc, "grads.dB_i_forw_enc");

						check_nan_inf(grads.dW_ccond_forw_enc, "grads.dW_ccond_forw_enc");
						check_nan_inf(grads.dU_ccond_forw_enc, "grads.dU_ccond_forw_enc");
						check_nan_inf(grads.dB_ccond_forw_enc, "grads.dB_ccond_forw_enc");

						check_nan_inf(grads.dW_o_forw_enc, "grads.dW_o_forw_enc");
						check_nan_inf(grads.dU_o_forw_enc, "grads.dU_o_forw_enc");
						check_nan_inf(grads.dB_o_forw_enc, "grads.dB_o_forw_enc");

						check_nan_inf(grads.dW_f_back_enc, "grads.dW_f_back_enc");
						check_nan_inf(grads.dU_f_back_enc, "grads.dU_f_back_enc");
						check_nan_inf(grads.dB_f_back_enc, "grads.dB_f_back_enc");

						check_nan_inf(grads.dW_i_back_enc, "grads.dW_i_back_enc");
						check_nan_inf(grads.dU_i_back_enc, "grads.dU_i_back_enc");
						check_nan_inf(grads.dB_i_back_enc, "grads.dB_i_back_enc");

						check_nan_inf(grads.dW_ccond_back_enc, "grads.dW_ccond_back_enc");
						check_nan_inf(grads.dU_ccond_back_enc, "grads.dU_ccond_back_enc");
						check_nan_inf(grads.dB_ccond_back_enc, "grads.dB_ccond_back_enc");

						check_nan_inf(grads.dW_o_back_enc, "grads.dW_o_back_enc");
						check_nan_inf(grads.dU_o_back_enc, "grads.dU_o_back_enc");
						check_nan_inf(grads.dB_o_back_enc, "grads.dB_o_back_enc");
					}
					else if (grad_norm > clip_threshold) {
						std::cout << "[CLIP] Batch " << (batch_step + 1)
							<< " gradient norm = " << grad_norm << " clipped\n";
					}
					else {
						std::cout << "[INFO] Batch " << (batch_step + 1)
							<< " gradient norm = " << grad_norm << "\n";
					}
					std::cout << "Epoch : " << epoch_ << "  step_optimisation : " << t_ << std::endl;



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

					/////
					/////
					/////
					double bias_corr_1 = (1 - std::pow(beta1, optima_steps + 1));
					double bias_corr_2 = (1 - std::pow(beta2, optima_steps + 1));
					MatrixXld _M_W_out = M_W_out.array() / bias_corr_1;
					MatrixXld _M_B_out = M_B_out.array() / bias_corr_1;
					//
					MatrixXld _M_W_gamma_layernorm = M_W_gamma_layernorm.array() / bias_corr_1;
					MatrixXld _M_B_beta_layernorm = M_B_beta_layernorm.array() / bias_corr_1;
					//
					MatrixXld _M_V_a_attention = M_V_a_attention.array() / bias_corr_1;
					MatrixXld _M_W_e_attention = M_W_e_attention.array() / bias_corr_1;
					MatrixXld _M_W_d_attention = M_W_d_attention.array() / bias_corr_1;
					//
					MatrixXld _M_W_f_dec = M_W_f_dec.array() / bias_corr_1;
					MatrixXld _M_U_f_dec = M_U_f_dec.array() / bias_corr_1;
					MatrixXld _M_B_f_dec = M_B_f_dec.array() / bias_corr_1;

					MatrixXld _M_W_i_dec = M_W_i_dec.array() / bias_corr_1;
					MatrixXld _M_U_i_dec = M_U_i_dec.array() / bias_corr_1;
					MatrixXld _M_B_i_dec = M_B_i_dec.array() / bias_corr_1;

					MatrixXld _M_W_ccond_dec = M_W_ccond_dec.array() / bias_corr_1;
					MatrixXld _M_U_ccond_dec = M_U_ccond_dec.array() / bias_corr_1;
					MatrixXld _M_B_ccond_dec = M_B_ccond_dec.array() / bias_corr_1;

					MatrixXld _M_W_o_dec = M_W_o_dec.array() / bias_corr_1;
					MatrixXld _M_U_o_dec = M_U_o_dec.array() / bias_corr_1;
					MatrixXld _M_B_o_dec = M_B_o_dec.array() / bias_corr_1;
					//
					MatrixXld _M_W_f_forw_enc = M_W_f_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_f_forw_enc = M_U_f_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_f_forw_enc = M_B_f_forw_enc.array() / bias_corr_1;

					MatrixXld _M_W_i_forw_enc = M_W_i_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_i_forw_enc = M_U_i_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_i_forw_enc = M_B_i_forw_enc.array() / bias_corr_1;

					MatrixXld _M_W_ccond_forw_enc = M_W_ccond_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_ccond_forw_enc = M_U_ccond_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_ccond_forw_enc = M_B_ccond_forw_enc.array() / bias_corr_1;

					MatrixXld _M_W_o_forw_enc = M_W_o_forw_enc.array() / bias_corr_1;
					MatrixXld _M_U_o_forw_enc = M_U_o_forw_enc.array() / bias_corr_1;
					MatrixXld _M_B_o_forw_enc = M_B_o_forw_enc.array() / bias_corr_1;
					//				   
					MatrixXld _M_W_f_back_enc = M_W_f_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_f_back_enc = M_U_f_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_f_back_enc = M_B_f_back_enc.array() / bias_corr_1;

					MatrixXld _M_W_i_back_enc = M_W_i_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_i_back_enc = M_U_i_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_i_back_enc = M_B_i_back_enc.array() / bias_corr_1;

					MatrixXld _M_W_ccond_back_enc = M_W_ccond_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_ccond_back_enc = M_U_ccond_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_ccond_back_enc = M_B_ccond_back_enc.array() / bias_corr_1;

					MatrixXld _M_W_o_back_enc = M_W_o_back_enc.array() / bias_corr_1;
					MatrixXld _M_U_o_back_enc = M_U_o_back_enc.array() / bias_corr_1;
					MatrixXld _M_B_o_back_enc = M_B_o_back_enc.array() / bias_corr_1;
					//				  
					//				  
					MatrixXld _V_W_out = V_W_out.array() / bias_corr_2;
					MatrixXld _V_B_out = V_B_out.array() / bias_corr_2;
					//
					MatrixXld _V_W_gamma_layernorm = V_W_gamma_layernorm.array() / bias_corr_2;
					MatrixXld _V_B_beta_layernorm = V_B_beta_layernorm.array() / bias_corr_2;
					//
					MatrixXld _V_V_a_attention = V_V_a_attention.array() / bias_corr_2;
					MatrixXld _V_W_e_attention = V_W_e_attention.array() / bias_corr_2;
					MatrixXld _V_W_d_attention = V_W_d_attention.array() / bias_corr_2;
					//
					MatrixXld _V_W_f_dec = V_W_f_dec.array() / bias_corr_2;
					MatrixXld _V_U_f_dec = V_U_f_dec.array() / bias_corr_2;
					MatrixXld _V_B_f_dec = V_B_f_dec.array() / bias_corr_2;

					MatrixXld _V_W_i_dec = V_W_i_dec.array() / bias_corr_2;
					MatrixXld _V_U_i_dec = V_U_i_dec.array() / bias_corr_2;
					MatrixXld _V_B_i_dec = V_B_i_dec.array() / bias_corr_2;

					MatrixXld _V_W_ccond_dec = V_W_ccond_dec.array() / bias_corr_2;
					MatrixXld _V_U_ccond_dec = V_U_ccond_dec.array() / bias_corr_2;
					MatrixXld _V_B_ccond_dec = V_B_ccond_dec.array() / bias_corr_2;

					MatrixXld _V_W_o_dec = V_W_o_dec.array() / bias_corr_2;
					MatrixXld _V_U_o_dec = V_U_o_dec.array() / bias_corr_2;
					MatrixXld _V_B_o_dec = V_B_o_dec.array() / bias_corr_2;
					//
					MatrixXld _V_W_f_forw_enc = V_W_f_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_f_forw_enc = V_U_f_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_f_forw_enc = V_B_f_forw_enc.array() / bias_corr_2;

					MatrixXld _V_W_i_forw_enc = V_W_i_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_i_forw_enc = V_U_i_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_i_forw_enc = V_B_i_forw_enc.array() / bias_corr_2;

					MatrixXld _V_W_ccond_forw_enc = V_W_ccond_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_ccond_forw_enc = V_U_ccond_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_ccond_forw_enc = V_B_ccond_forw_enc.array() / bias_corr_2;

					MatrixXld _V_W_o_forw_enc = V_W_o_forw_enc.array() / bias_corr_2;
					MatrixXld _V_U_o_forw_enc = V_U_o_forw_enc.array() / bias_corr_2;
					MatrixXld _V_B_o_forw_enc = V_B_o_forw_enc.array() / bias_corr_2;
					//				   
					MatrixXld _V_W_f_back_enc = V_W_f_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_f_back_enc = V_U_f_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_f_back_enc = V_B_f_back_enc.array() / bias_corr_2;

					MatrixXld _V_W_i_back_enc = V_W_i_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_i_back_enc = V_U_i_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_i_back_enc = V_B_i_back_enc.array() / bias_corr_2;

					MatrixXld _V_W_ccond_back_enc = V_W_ccond_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_ccond_back_enc = V_U_ccond_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_ccond_back_enc = V_B_ccond_back_enc.array() / bias_corr_2;

					MatrixXld _V_W_o_back_enc = V_W_o_back_enc.array() / bias_corr_2;
					MatrixXld _V_U_o_back_enc = V_U_o_back_enc.array() / bias_corr_2;
					MatrixXld _V_B_o_back_enc = V_B_o_back_enc.array() / bias_corr_2;
					/////
					/////
					/////
					/////
					this->decoder_->W_Output.array() -= learning_rate * _M_W_out.array() / (_V_W_out.array().sqrt() + epsilon);
					this->decoder_->B_Output.array() -= learning_rate * _M_B_out.array() / (_V_B_out.array().sqrt() + epsilon);
					//
					this->decoder_->layernorm_gamma.array() -= learning_rate * _M_W_gamma_layernorm.array() / (_V_W_gamma_layernorm.array().sqrt() + epsilon);
					this->decoder_->layernorm_beta.array() -= learning_rate * _M_B_beta_layernorm.array() / (_V_B_beta_layernorm.array().sqrt() + epsilon);
					//
					this->decoder_->attention_->attention_vector_.array() -= learning_rate * _M_V_a_attention.array() / (_V_V_a_attention.array().sqrt() + epsilon);
					this->decoder_->attention_->W_encoder_.array() -= learning_rate * _M_W_e_attention.array() / (_V_W_e_attention.array().sqrt() + epsilon);
					this->decoder_->attention_->W_decoder_.array() -= learning_rate * _M_W_d_attention.array() / (_V_W_d_attention.array().sqrt() + epsilon);
					//
					this->decoder_->W_F.array() -= learning_rate * _M_W_f_dec.array() / (_V_W_f_dec.array().sqrt() + epsilon);
					this->decoder_->U_F.array() -= learning_rate * _M_U_f_dec.array() / (_V_U_f_dec.array().sqrt() + epsilon);
					this->decoder_->B_F.array() -= learning_rate * _M_B_f_dec.array() / (_V_B_f_dec.array().sqrt() + epsilon);

					this->decoder_->W_I.array() -= learning_rate * _M_W_i_dec.array() / (_V_W_i_dec.array().sqrt() + epsilon);
					this->decoder_->U_I.array() -= learning_rate * _M_U_i_dec.array() / (_V_U_i_dec.array().sqrt() + epsilon);
					this->decoder_->B_I.array() -= learning_rate * _M_B_i_dec.array() / (_V_B_i_dec.array().sqrt() + epsilon);

					this->decoder_->W_C.array() -= learning_rate * _M_W_ccond_dec.array() / (_V_W_ccond_dec.array().sqrt() + epsilon);
					this->decoder_->U_C.array() -= learning_rate * _M_U_ccond_dec.array() / (_V_U_ccond_dec.array().sqrt() + epsilon);
					this->decoder_->B_C.array() -= learning_rate * _M_B_ccond_dec.array() / (_V_B_ccond_dec.array().sqrt() + epsilon);

					this->decoder_->W_O.array() -= learning_rate * _M_W_o_dec.array() / (_V_W_o_dec.array().sqrt() + epsilon);
					this->decoder_->U_O.array() -= learning_rate * _M_U_o_dec.array() / (_V_U_o_dec.array().sqrt() + epsilon);
					this->decoder_->B_O.array() -= learning_rate * _M_B_o_dec.array() / (_V_B_o_dec.array().sqrt() + epsilon);

					//
					this->encoder_->Forward.W_F.array() -= learning_rate * _M_W_f_forw_enc.array() / (_V_W_f_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_F.array() -= learning_rate * _M_U_f_forw_enc.array() / (_V_U_f_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_F.array() -= learning_rate * _M_B_f_forw_enc.array() / (_V_B_f_forw_enc.array().sqrt() + epsilon);

					this->encoder_->Forward.W_I.array() -= learning_rate * _M_W_i_forw_enc.array() / (_V_W_i_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_I.array() -= learning_rate * _M_U_i_forw_enc.array() / (_V_U_i_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_I.array() -= learning_rate * _M_B_i_forw_enc.array() / (_V_B_i_forw_enc.array().sqrt() + epsilon);

					this->encoder_->Forward.W_C.array() -= learning_rate * _M_W_ccond_forw_enc.array() / (_V_W_ccond_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_C.array() -= learning_rate * _M_U_ccond_forw_enc.array() / (_V_U_ccond_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_C.array() -= learning_rate * _M_B_ccond_forw_enc.array() / (_V_B_ccond_forw_enc.array().sqrt() + epsilon);

					this->encoder_->Forward.W_O.array() -= learning_rate * _M_W_o_forw_enc.array() / (_V_W_o_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.U_O.array() -= learning_rate * _M_U_o_forw_enc.array() / (_V_U_o_forw_enc.array().sqrt() + epsilon);
					this->encoder_->Forward.B_O.array() -= learning_rate * _M_B_o_forw_enc.array() / (_V_B_o_forw_enc.array().sqrt() + epsilon);
					//				   
					this->encoder_->Backward.W_F.array() -= learning_rate * _M_W_f_back_enc.array() / (_V_W_f_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_F.array() -= learning_rate * _M_U_f_back_enc.array() / (_V_U_f_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_F.array() -= learning_rate * _M_B_f_back_enc.array() / (_V_B_f_back_enc.array().sqrt() + epsilon);

					this->encoder_->Backward.W_I.array() -= learning_rate * _M_W_i_back_enc.array() / (_V_W_i_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_I.array() -= learning_rate * _M_U_i_back_enc.array() / (_V_U_i_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_I.array() -= learning_rate * _M_B_i_back_enc.array() / (_V_B_i_back_enc.array().sqrt() + epsilon);

					this->encoder_->Backward.W_C.array() -= learning_rate * _M_W_ccond_back_enc.array() / (_V_W_ccond_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_C.array() -= learning_rate * _M_U_ccond_back_enc.array() / (_V_U_ccond_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_C.array() -= learning_rate * _M_B_ccond_back_enc.array() / (_V_B_ccond_back_enc.array().sqrt() + epsilon);

					this->encoder_->Backward.W_O.array() -= learning_rate * _M_W_o_back_enc.array() / (_V_W_o_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.U_O.array() -= learning_rate * _M_U_o_back_enc.array() / (_V_U_o_back_enc.array().sqrt() + epsilon);
					this->encoder_->Backward.B_O.array() -= learning_rate * _M_B_o_back_enc.array() / (_V_B_o_back_enc.array().sqrt() + epsilon);
				}
			}
		}

		Inference(shuffle_target[0]);
		for (size_t end_i = 0; end_i < Target_input_output.size(); end_i++) {
			grads_end_avg_train_loss += BackwardWithLogging(end_i, shuffle_target[1][end_i]);
		}
		grads_end_avg_train_loss /= Target_input_output.size();


		this->Save(packname_forsave);

		end_time = std::chrono::steady_clock::now();

		std::chrono::duration<double> elapsed = end_time - start_time;

		std::cout << "Epoch " << (epoch_ + 1)
			<< " finished. Avg train loss [start/end]: "
			<< get_mean_grads(grads_start_avg_train_loss) << "/" << get_mean_grads(grads_end_avg_train_loss)
			//<< ", Val loss: " << val_loss
			<< ", Time_epoch: " << elapsed.count() << "s\n";
	}
}
