#include "HeaderBaseRNN.h"

namespace MyNN {
	namespace RNN {
		template<typename T, typename Enable>
		ComputeBlockRNN<T, Enable>& ComputeBlockRNN<T, Enable>::operator=(const ComputeBlockRNN & other) {
			*(this->n_state_) = *(other.n_state_);
			this->hidden_size_ = other.hidden_size_;
			ComputeBlockNN<T, Enable>::operator=(other);
		}
		template<typename T, typename Enable>
		ComputeBlockRNN<T, Enable>& ComputeBlockRNN<T, Enable>::operator=(ComputeBlockRNN && other) noexcept {
			this->n_state_ = std::move(other.n_state_);
			this->hidden_size_ = std::move(other.hidden_size_);
			other.n_state_ = nullptr;
			ComputeBlockNN<T, Enable>::operator=(std::move(other));
		}
	}
}