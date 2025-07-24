#include "HeaderSeq2seqWithAttention.h"
 

Seq2SeqWithAttention::Seq2SeqWithAttention(
	std::unique_ptr<Encoder> encoder,
	std::unique_ptr<Decoder> decoder)
		: encoder_(std::move(encoder)), decoder_(std::move(decoder)) {
	}

Seq2SeqWithAttention::Seq2SeqWithAttention(
	Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
	Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_,
	std::unique_ptr<BahdanauAttention> attention_)
		:
		encoder_(std::make_unique<Encoder>(Input_size_, Encoder_Hidden_size_)),
	decoder_(std::make_unique<Decoder>(std::move(attention_), Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {
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

const std::vector<MatrixXld>& Seq2SeqWithAttention::GetOutputs() const {
		return decoder_->GetOutputStates();
	}