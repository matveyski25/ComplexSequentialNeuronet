#include "HeaderSeq2seqWithAttention.h"

Seq2SeqWithAttention_ForTrain::Seq2SeqWithAttention_ForTrain(std::unique_ptr<Encoder> encoder_train, std::unique_ptr<Decoder> decoder_train)
	: encoder_(std::move(encoder_train)), decoder_(std::move(decoder_train)) {
}

  
