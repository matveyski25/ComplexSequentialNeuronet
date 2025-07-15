#pragma once
#include "HeaderTemplates_for_Seq2seqWithAttention.h"

class SimpleLSTM_ForTrain_ForDecoder : public Decoder_<SimpleLSTM_ForTrain> {
public:
	struct states_forgrads {
		std::vector<MatrixXld> f, i, o, ccond, c, h, context, z, x, p, p_;
	};

	states_forgrads StatesForgrads;

	friend class Seq2SeqWithAttention_ForTrain;
	/*void Batch_All_state_Ñalculation(
		const std::vector<MatrixXld>& encoder_outputs,
		const std::vector<MatrixXld>& teacher_inputs,        // [B][T_dec x emb_dim]
		const std::vector<std::vector<bool>>& loss_mask,     // [B][T_dec]
		long double teacher_forcing_ratio)
	{
	}*/

	void All_state_Calculation() {}
};

class SimpleLSTM_ForDecoder : public Decoder_<SimpleLSTM> {
public:

	/*void Batch_All_state_Ñalculation(
		const std::vector<MatrixXld>& encoder_outputs,
		const std::vector<MatrixXld>& teacher_inputs,        // [B][T_dec x emb_dim]
		const std::vector<std::vector<bool>>& loss_mask,     // [B][T_dec]
		long double teacher_forcing_ratio)
	{
	}*/

	void All_state_Calculation() {}
};
