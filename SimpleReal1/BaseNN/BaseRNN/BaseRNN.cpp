#include "BaseRNN.h"

namespace MyNN {
	namespace RNN {
		template<typename T>
		ComputeBlockRNN<T>& ComputeBlockRNN<T>::operator=(const ComputeBlockRNN & other) {
			if(this != &other){
				*(this->n_state_) = *(other.n_state_);
				this->hidden_size_ = other.hidden_size_;
				this->max_steps_ = other.max_steps_;
				ComputeBlockNN<T>::operator=(other);
			}
			return *this;
		}
		template<typename T>
		ComputeBlockRNN<T>& ComputeBlockRNN<T>::operator=(ComputeBlockRNN && other) noexcept {
			this->n_state_ = std::move(other.n_state_);
			this->hidden_size_ = std::move(other.hidden_size_);
			this->max_steps_ = other.max_steps_;
			other.n_state_ = nullptr;
			ComputeBlockNN<T>::operator=(std::move(other));
			return *this;
		}
	}
}