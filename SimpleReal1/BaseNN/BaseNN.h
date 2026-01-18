#pragma once
#include <cstdint>
#include "RealizationMatrix.hpp"
#include "TemplateLimits.hpp"
#include "MyPtr.hpp"

//ToDo - при копировании не разыменовывались nullptr указатели на компоненты, и если таковые имеются, то делать make_unique(other)
namespace MyNN::Base {
    using std::uint64_t, Utils::MyPtr::copy_ptr,
    Utils::TemplateLimits::IsArithmeticType,
    LinearAlgebra::BaseMatrix, Utils::TemplateLimits::IFeature;

    template<typename T, typename Context>
    struct FeatureSaveLoadManager : IsArithmeticType<T>, IFeature<Context>{
        virtual void save() = 0;
        virtual void load() = 0;
        struct ISaveLoadable {
            virtual void setSaveLoadManager(copy_ptr<FeatureSaveLoadManager<T, Context>>) = 0;
            virtual const FeatureSaveLoadManager<T, Context>* getSaveLoadManager() = 0;
        };
    };
    template<typename T, typename Context>
    struct FeatureComputeBlockNN : IsArithmeticType<T>, IFeature<Context> {
        virtual void setInput(BaseMatrix<T>) = 0;
        virtual BaseMatrix<T> getOutput() = 0;
        virtual void compute() = 0;
        struct IComputable {
            virtual void setComputeBlock(copy_ptr<FeatureComputeBlockNN<T, Context>>) = 0;
            virtual const FeatureComputeBlockNN<T, Context>* getComputeBlock() = 0;
        };
    };
    template<typename T, typename Context>
    struct FeatureRandomizerMatrix : IsArithmeticType<T>, IFeature<Context> {
        virtual void randomize() = 0;
        struct IRandomizable {
            virtual void setRandomizerMatrix(copy_ptr<FeatureRandomizerMatrix<T, Context>>) = 0;
            virtual const FeatureRandomizerMatrix<T, Context>* getRandomizerMatrix() = 0;
        };
    };
    template<typename T, typename Context>
    struct FeatureOptimizerComputeBlock : IsArithmeticType<T, Context> {
        virtual void optimize() = 0;
        struct IOptimizable {
            virtual void setOptimizerComputeBlock(copy_ptr<FeatureOptimizerComputeBlock<T, Context>>) = 0;
            virtual const FeatureOptimizerComputeBlock<T, Context>* getOptimizerComputeBlock() = 0;
        };
    };
    template<typename T, typename Context, typename FromType, typename ToType>
    struct FeatureTranslatorMatrix : IsArithmeticType<T> {
        using FromType_ = FromType;
        using ToType_ = ToType;
        virtual LinearAlgebra::BaseMatrix<T> operator()(const FromType &) = 0;
        virtual ToType& operator()(LinearAlgebra::BaseMatrix<T>) = 0;
        struct ITranslatable {
            virtual void setTranslatorMatrix(copy_ptr<FeatureTranslatorMatrix<T, Context, FromType, ToType>>) = 0;
            virtual const FeatureTranslatorMatrix<T, Context, FromType, ToType>* getTranslatorMatrix() = 0;
        };
    };
    template<typename T, typename FromType, typename ToType>
    struct FeatureBaseNN :
    FeatureSaveLoadManager<T>::ISaveLoadable,
    FeatureComputeBlockNN<T>::IComputable,
    FeatureTranslatorMatrix<T, FromType, ToType>::ITranslatable,
    IsArithmeticType<T> {
        virtual void setInput(const FromType&) = 0;
        virtual ToType getOutput() = 0;
        virtual void inference() = 0;
        template<typename Derived>
        Derived reformat() {
            return static_cast<Derived*>(this)->reformatImpl();
        }
    };

    template<typename T>
    struct FeatureTrainableComputeBlockNN :
    FeatureComputeBlockNN<T>,
    FeatureRandomizerMatrix<T>::IRandomizable,
    FeatureOptimizerComputeBlock<T>::IOptimizable {
        virtual void backward() = 0;
        virtual void getDeltasWeights() = 0;
    };
    template<typename T, typename FromType, typename ToType>
    struct FeatureBaseTrainableNN : public FeatureBaseNN<T, FromType, ToType> {
        virtual void train() = 0;
    };

    template<typename T>
    struct ContextComputeBlockNN {
        BaseMatrix<T> input_state_;
        BaseMatrix<T> output_state_;
        uint64_t input_size_;
        uint64_t output_size_;
    };
    template<typename T, typename FromType, typename ToType>
    struct ContextBaseNN{
        copy_ptr<FeatureSaveLoadManager<T>> save_load_manager_;
        copy_ptr<FeatureOptimizerComputeBlock<T>> compute_block_;
        copy_ptr<FeatureTranslatorMatrix<T, FromType, ToType>> translator_;
    };

    template<typename T>
    struct ContextTrainableComputeBlockNN : ContextComputeBlockNN<T> {
        copy_ptr<FeatureRandomizerMatrix<T>> randomizer_;
        copy_ptr<FeatureOptimizerComputeBlock<T>> optimizer_;
    };
    template<typename T, typename FromType, typename ToType>
    struct ContextBaseTrainableNN : public ContextBaseNN<T, FromType, ToType> {};

}
