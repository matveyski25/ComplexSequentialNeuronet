#include "HeaderBaseNN.h"
namespace MyNN{
    template<typename T, typename Enable>
    ComputeBlockNN<T, Enable>& ComputeBlockNN<T, Enable>::operator=(const ComputeBlockNN& other) {
        if (this != &other) {
            *this->values_for_compute_ = *(other.values_for_compute_);

            input_state_ = other.input_state_;
            input_size_ = other.input_size_;
            output_size_ = other.output_size_;
        }
        return *this;
    }
    template<typename T, typename Enable>
    ComputeBlockNN<T, Enable>& ComputeBlockNN<T, Enable>::operator=(ComputeBlockNN&& other) noexcept {
        values_for_compute_ = std::move(other.values_for_compute_);
        input_state_ = std::move(other.input_state_);
        input_size_ = other.input_size_;
        output_size_ = other.output_size_;
        other.values_for_compute_ = nullptr;
        return *this;
    }
    template<typename T, typename Enable>
    void ComputeBlockNN<T, Enable>::setInput(LinearAlgebra::BaseMatrix<T, Enable> input) {
        input_state_ = input; 
    }

    template<typename T, typename Enable>
    TrainableComputeBlockNN<T, Enable>& TrainableComputeBlockNN<T, Enable>::operator=(const TrainableComputeBlockNN& other) {
        if (this != &other) {
            *this->saver_ = *(other.saver_);
            *this->loader_ = *(other.loader_);
            *this->optimizer_ = *(other.optimizer_);
            *this->randomizer_ = *(other.randomizer_);
            *this->intermediate_values_ = *(other.intermediate_values_);
            ComputeBlockNN<T, Enable>::operator=(other);
        }
        return *this;
    }
    template<typename T, typename Enable>
    TrainableComputeBlockNN<T, Enable>& TrainableComputeBlockNN<T, Enable>::operator=(TrainableComputeBlockNN && other) noexcept {
        this->saver_ = std::move(other.saver_);
        this->loader_ = std::move(other.loader_);
        this->optimizer_ = std::move(other.optimizer_);
        this->randomizer_ = std::move(other.randomizer_);
        this->intermediate_values_ = std::move(other.intermediate_values_);
        ComputeBlockNN<T, Enable>::operator=(std::move(other));
        other.saver_ = nullptr;
        other.loader_ = nullptr;
        other.optimizer_ = nullptr;
        other.randomizer_ = nullptr;
        other.intermediate_values_ = nullptr;
        return *this;
    }

    template<typename T, typename Enable>
    BaseNN<T, Enable>::BaseNN(std::unique_ptr<IComputeBlockNN<T, Enable>> compute_block, std::unique_ptr<ITranslatorMatrix<T, Enable>> translator_matrix)
    {
        this->compute_block_ = std::move(compute_block);
        this->translator_ = std::move(translator_matrix);
    }
    template<typename T, typename Enable>
    BaseNN<T, Enable>& BaseNN<T, Enable>::operator=(const BaseNN& other) {
        if (this != &other) {
            *compute_block_ = *(other.compute_block_);
            *translator_ = *(other.translator_);
            input_state_ = other.input_state_;
            output_state_ = other.output_state_;
        }
        return *this;
    }
    template<typename T, typename Enable>
    BaseNN<T, Enable>& BaseNN<T, Enable>::operator=(BaseNN&& other) noexcept {
        compute_block_ = std::move(other.compute_block_);
        translator_ = std::move(other.translator_);
        input_state_ = std::move(other.input_state_);
        output_state_ = std::move(other.output_state_);
        other.compute_block_ = nullptr;
        other.translator_ = nullptr;
        return *this;
    }
    template<typename T, typename Enable>
    void BaseNN<T, Enable>::inference() { 
        this->forward(); 
    }
    template<typename T, typename Enable>
    void BaseNN<T, Enable>::forward() {
        auto input = (*this->translator_)(input_state_);
        compute_block_->setInput(input);
        compute_block_->compute();
    }
    template<typename T, typename Enable>
    void BaseNN<T, Enable>::setComputeBlock(std::unique_ptr<IComputeBlockNN<T, Enable>> compute_block) {
        this->compute_block_ = std::move(compute_block);
    }
    template<typename T, typename Enable>
    void BaseNN<T, Enable>::setTranslatorMatrix(std::unique_ptr<ITranslatorMatrix<T, Enable>> translator) {
        this->translator_ = std::move(translator);
    }
    template<typename T, typename Enable>
    const IComputeBlockNN<T, Enable>* BaseNN<T, Enable>::getComputeBlock() {
        return this->compute_block_;
    }
    template<typename T, typename Enable>
    const ITranslatorMatrix<T, Enable>* BaseNN<T, Enable>::getTranslatorMatrix() {
        return this->translator_.get();
    }
}

//TODO Сделать проверку на самоприсваивание в операторах присваивания