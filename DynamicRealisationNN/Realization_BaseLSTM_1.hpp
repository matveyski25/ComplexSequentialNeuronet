#pragma once

#include "BaseRNN.hpp"
#include "RealizationMatrix.hpp"



template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class BaseLSTM : public BaseRNN
{

};

template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class TrainableLSTM : virtual public BaseLSTM<T>, virtual public BaseTrainableRNN
{

};