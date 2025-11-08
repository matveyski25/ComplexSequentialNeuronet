#pragma once
#include <type_traits>
#include <memory>
#include "RealizationMatrix.hpp"
//ToDo - при копировании не разыменовывались nullptr указатели на компоненты, и если таковые имеются, то делать make_unique(other)
namespace MyNN {

    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ITranslatorMatrix;
    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IComputeBlockNN;

    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ILoadable {
    public:
        class ILoader;
        virtual void setLoader(std::unique_ptr <ILoader>) = 0;
        virtual ILoader* getLoader() = 0;
    };
    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ISaveable {
    public:
        class ISaver;
        virtual void setSaver(std::unique_ptr <ISaver>) = 0;
        virtual ISaver* getSaver() = 0;
    };
    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IOptimizeable {
    public:
        class IOptimizer;
        virtual void setOptimizer(std::unique_ptr <IOptimizer>) = 0;
        virtual IOptimizer* getOptimizer() = 0;
    };
    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IRandomizeable {
    public:
        class IRandomizer;
        virtual void setRandomizer(std::unique_ptr <IRandomizer>) = 0;
        virtual IRandomizer* getRandomizer() = 0;
    };

    template<typename T, typename Enable>
    class IComputeBlockNN : public ILoadable<T> {
        friend class IOptimizeable<T>::IOptimizer;
        friend class IRandomizeable<T>::IRandomizer;
    protected:
        struct ValuesForCompute {};
    public:
        virtual const ValuesForCompute * getValuesForCompute() = 0;
        virtual void setValuesForCompute(const ValuesForCompute*) = 0;
        virtual void setInput(LinearAlgebra::BaseMatrix<T> input) = 0;
        virtual LinearAlgebra::BaseMatrix<T> getOutput() = 0;
        virtual void compute() = 0;
        virtual ~IComputeBlockNN() = default;
    };

    template<typename T, typename Enable>
    class ILoadable<T, Enable>::ILoader {
    public:
        struct ArgsLoader {};
        virtual void load(IComputeBlockNN<T>*) = 0;
        virtual void setArgsForLoad(const ArgsLoader*) = 0;
    };
    template<typename T, typename Enable>
    class ISaveable<T, Enable>::ISaver {
        struct ArgsSaver {};
        virtual void save(IComputeBlockNN<T>*) = 0;
        virtual void setArgsForSave(const ArgsSaver*) = 0;
    };
    template<typename T, typename Enable>
    class IOptimizeable<T, Enable>::IOptimizer {
    public:
        struct ArgsOptimizer {};
        struct Gradients : IComputeBlockNN<T>::ValuesForCompute {};
        virtual void optimize(const Gradients*, typename IComputeBlockNN<T>::ValuesForCompute*) = 0;
        virtual void setArgsForOptimize(const ArgsOptimizer*) = 0;
    };
    template<typename T, typename Enable>
    class IRandomizeable<T, Enable>::IRandomizer {
    public:
        struct ArgsRandomizer {};
        virtual void randomize(typename IComputeBlockNN<T>::ValuesForCompute*) = 0;
        virtual void setArgsForRandomize(const ArgsRandomizer*) = 0;
    };

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ITrainableComputeBlockNN : public IComputeBlockNN<T>, public ISaveable<T>, IOptimizeable<T>, IRandomizeable<T> {
    protected:
        struct IntermediateValues {};
    };

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IBaseNN {
    protected:
        virtual void forward() = 0;
    public:
        struct InputValue {};
        struct OutputValue {};

        virtual void inference() = 0;
        virtual void setInputState(InputValue) = 0;
        virtual InputValue getInputState() = 0;
        virtual OutputValue getOutputState() = 0;
        virtual void setComputeBlock(std::unique_ptr<IComputeBlockNN<T>>) = 0;
        virtual const IComputeBlockNN<T>* getComputeBlock() = 0;

        virtual void setTranslatorMatrix(std::unique_ptr<ITranslatorMatrix<T>>) = 0;
        virtual const ITranslatorMatrix<T>* getTranslatorMatrix() = 0;
        virtual ~IBaseNN() = default;
    };

    template<typename T, typename Enable>
    class ITranslatorMatrix {
    public:
        virtual LinearAlgebra::BaseMatrix<T> operator()(typename IBaseNN<T>::InputValue) = 0;
        virtual typename IBaseNN<T>::OutputValue operator()(LinearAlgebra::BaseMatrix<T>) = 0;
    };

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IBaseTrainableNN : public IBaseNN<T> {};

    template<typename T>
    class ComputeBlockNN : public IComputeBlockNN<T> {
    protected:
        using Values = typename IComputeBlockNN<T>::ValuesForCompute;

        std::unique_ptr<Values> values_for_compute_;
        LinearAlgebra::BaseMatrix<T> input_state_;
        std::uint64_t input_size_;
        std::uint64_t output_size_;
    public:
        ComputeBlockNN& operator=(const ComputeBlockNN& other);
        ComputeBlockNN& operator=(ComputeBlockNN&& other) noexcept;
        void setInput(LinearAlgebra::BaseMatrix<T> input) override;

        class Loader : public IComputeBlockNN<T>::ILoader {
        protected:
            std::unique_ptr<typename IComputeBlockNN<T>::ILoader::ArgsLoader> args_;
        };
        class Saver : public IComputeBlockNN<T>::ISaver {
        protected:
            std::unique_ptr<typename IComputeBlockNN<T>::ISaver::ArgsSaver> args_;
        };

        std::uint64_t getInputSize() {
            return this->input_size_;
        }
        std::uint64_t getOutputSize() {
            return this->output_size_;
        }
    };

    template<typename T>
    class TrainableComputeBlockNN : public ComputeBlockNN<T>, public ITrainableComputeBlockNN<T> {
    protected:
        using Base = ITrainableComputeBlockNN<T>;
        using Saver = typename Base::ISaver;
        using Loader = typename Base::ILoader;
        using Optim = typename Base::IOptimizer;
        using Random = typename Base::IRandomizer;
        using Intermediate = typename Base::IntermediateValues;

        std::unique_ptr<Saver> saver_;
        std::unique_ptr<Loader> loader_;
        std::unique_ptr<Optim> optimizer_;
        std::unique_ptr<Random> randomizer_;
        std::unique_ptr<Intermediate> intermediate_values;

    public:
        TrainableComputeBlockNN& operator=(const TrainableComputeBlockNN& other);
        TrainableComputeBlockNN& operator=(TrainableComputeBlockNN&& other) noexcept;
    };

    template<typename T>
    class BaseNN : public IBaseNN<T> {
    protected:
        using Base = IBaseNN<T>;

        std::unique_ptr<IComputeBlockNN<T>> compute_block_;
        std::unique_ptr<ITranslatorMatrix<T>> translator_;
        typename Base::InputValue input_state_;
        typename Base::OutputValue output_state_;

        void forward() override;
    public:
        BaseNN(std::unique_ptr<IComputeBlockNN<T>>, std::unique_ptr<ITranslatorMatrix<T>>);

        BaseNN& operator=(const BaseNN& other);
        BaseNN& operator=(BaseNN&& other) noexcept;
        void inference() override;
        void setComputeBlock(std::unique_ptr<IComputeBlockNN<T>> compute_block) override;
        void setTranslatorMatrix(std::unique_ptr<ITranslatorMatrix<T>> translator) override;
        const IComputeBlockNN<T>* getComputeBlock() override;
        const ITranslatorMatrix<T>* getTranslatorMatrix() override;
    };

    template<typename T>
    class BaseTrainableNN : public IBaseTrainableNN<T>, public BaseNN<T> {};

} // namespace MyNN
