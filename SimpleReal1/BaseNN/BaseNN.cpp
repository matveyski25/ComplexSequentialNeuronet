#include "BaseNN.h"
namespace MyNN{
    template<typename T>
    ComputeBlockNN<T>& ComputeBlockNN<T>::operator=(const ComputeBlockNN& other) {
        if (this != &other) {
            *this->values_for_compute_ = *(other.values_for_compute_);

            input_state_ = other.input_state_;
            input_size_ = other.input_size_;
            output_size_ = other.output_size_;
        }
        return *this;
    }
    template<typename T>
    ComputeBlockNN<T>& ComputeBlockNN<T>::operator=(ComputeBlockNN&& other) noexcept {
        values_for_compute_ = std::move(other.values_for_compute_);
        input_state_ = std::move(other.input_state_);
        input_size_ = other.input_size_;
        output_size_ = other.output_size_;
        other.values_for_compute_ = nullptr;
        return *this;
    }
    template<typename T>
    void ComputeBlockNN<T>::setInput(LinearAlgebra::BaseMatrix<T> input) {
        input_state_ = input; 
    }

    template<typename T>
    TrainableComputeBlockNN<T>& TrainableComputeBlockNN<T>::operator=(const TrainableComputeBlockNN& other) {
        if (this != &other) {
            *this->saver_ = *(other.saver_);
            *this->loader_ = *(other.loader_);
            *this->optimizer_ = *(other.optimizer_);
            *this->randomizer_ = *(other.randomizer_);
            *this->intermediate_values_ = *(other.intermediate_values_);
            ComputeBlockNN<T>::operator=(other);
        }
        return *this;
    }
    template<typename T>
    TrainableComputeBlockNN<T>& TrainableComputeBlockNN<T>::operator=(TrainableComputeBlockNN && other) noexcept {
        this->saver_ = std::move(other.saver_);
        this->loader_ = std::move(other.loader_);
        this->optimizer_ = std::move(other.optimizer_);
        this->randomizer_ = std::move(other.randomizer_);
        this->intermediate_values_ = std::move(other.intermediate_values_);
        ComputeBlockNN<T>::operator=(std::move(other));
        other.saver_ = nullptr;
        other.loader_ = nullptr;
        other.optimizer_ = nullptr;
        other.randomizer_ = nullptr;
        other.intermediate_values_ = nullptr;
        return *this;
    }

    template<typename T>
    BaseNN<T>::BaseNN(std::unique_ptr<IComputeBlockNN<T>> compute_block, std::unique_ptr<ITranslatorMatrix<T>> translator_matrix)
    {
        this->compute_block_ = std::move(compute_block);
        this->translator_ = std::move(translator_matrix);
    }
    template<typename T>
    BaseNN<T>& BaseNN<T>::operator=(const BaseNN& other) {
        if (this != &other) {
            *compute_block_ = *(other.compute_block_);
            *translator_ = *(other.translator_);
            input_state_ = other.input_state_;
            output_state_ = other.output_state_;
        }
        return *this;
    }
    template<typename T>
    BaseNN<T>& BaseNN<T>::operator=(BaseNN&& other) noexcept {
        compute_block_ = std::move(other.compute_block_);
        translator_ = std::move(other.translator_);
        input_state_ = std::move(other.input_state_);
        output_state_ = std::move(other.output_state_);
        other.compute_block_ = nullptr;
        other.translator_ = nullptr;
        return *this;
    }
    template<typename T>
    void BaseNN<T>::inference() { 
        this->forward(); 
    }
    template<typename T>
    void BaseNN<T>::forward() {
        auto input = (*this->translator_)(input_state_);
        compute_block_->setInput(input);
        compute_block_->compute();
    }
    template<typename T>
    void BaseNN<T>::setComputeBlock(std::unique_ptr<IComputeBlockNN<T>> compute_block) {
        this->compute_block_ = std::move(compute_block);
    }
    template<typename T>
    void BaseNN<T>::setTranslatorMatrix(std::unique_ptr<ITranslatorMatrix<T>> translator) {
        this->translator_ = std::move(translator);
    }
    template<typename T>
    const IComputeBlockNN<T>* BaseNN<T>::getComputeBlock() {
        return this->compute_block_;
    }
    template<typename T>
    const ITranslatorMatrix<T>* BaseNN<T>::getTranslatorMatrix() {
        return this->translator_.get();
    }
}

//TODO Сделать проверку на самоприсваивание в операторах присваивания