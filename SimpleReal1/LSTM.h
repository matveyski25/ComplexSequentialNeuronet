#pragma once
#include "HeaderBaseRNN.h"

namespace MyNN {
	namespace RNN {
		template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		class LSTM1 : public BaseRNN<T> {
			void foo() {
				this->forward();
			}
		};
	}
}