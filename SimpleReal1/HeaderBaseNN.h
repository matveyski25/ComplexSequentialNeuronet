#pragma once
#include <type_traits>
#include <memory>
#include "RealizationMatrix.hpp"
//ToDo - при копировании не разыменовывались nullptr указатели на компоненты, и если таковые имеются, то делать make_unique(other)
namespace MyNN {

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ITranslatorMatrix;
    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IComputeBlockNN;

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ILoadable {
    public:
        class ILoader;
        virtual void setLoader(std::unique_ptr <ILoader>) = 0;
        virtual ILoader* getLoader() = 0;
    };
    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ISaveable {
    public:
        class ISaver;
        virtual void setSaver(std::unique_ptr <ISaver>) = 0;
        virtual ISaver* getSaver() = 0;
    };
    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IOptimizeable {
    public:
        class IOptimizer;
        virtual void setOptimizer(std::unique_ptr <IOptimizer>) = 0;
        virtual IOptimizer* getOptimizer() = 0;
    };
    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IRandomizeable {
    public:
        class IRandomizer;
        virtual void setRandomizer(std::unique_ptr <IRandomizer>) = 0;
        virtual IRandomizer* getRandomizer() = 0;
    };

    template<typename T, typename Enable>
    class IComputeBlockNN : public ILoadable<T, Enable> {
        friend class IOptimizeable<T, Enable>::IOptimizer;
        friend class IRandomizeable<T, Enable>::IRandomizer;
    protected:
        struct ValuesForCompute {};
    public:
        virtual const ValuesForCompute * getValuesForCompute() = 0;
        virtual void setValuesForCompute(const ValuesForCompute*) = 0;
        virtual void setInput(LinearAlgebra::BaseMatrix<T, Enable> input) = 0;
        virtual LinearAlgebra::BaseMatrix<T, Enable> getOutput() = 0;
        virtual void compute() = 0;
        virtual ~IComputeBlockNN() = default;
    };

    template<typename T, typename Enable>
    class ILoadable<T, Enable>::ILoader {
    public:
        struct ArgsLoader {};
        virtual void load(IComputeBlockNN<T, Enable>*) = 0;
        virtual void setArgsForLoad(const ArgsLoader*) = 0;
    };
    template<typename T, typename Enable>
    class ISaveable<T, Enable>::ISaver {
        struct ArgsSaver {};
        virtual void save(IComputeBlockNN<T, Enable>*) = 0;
        virtual void setArgsForSave(const ArgsSaver*) = 0;
    };
    template<typename T, typename Enable>
    class IOptimizeable<T, Enable>::IOptimizer {
    public:
        struct ArgsOptimizer {};
        struct Gradients : IComputeBlockNN<T, Enable>::ValuesForCompute {};
        virtual void optimize(const Gradients*, typename IComputeBlockNN<T, Enable>::ValuesForCompute*) = 0;
        virtual void setArgsForOptimize(const ArgsOptimizer*) = 0;
    };
    template<typename T, typename Enable>
    class IRandomizeable<T, Enable>::IRandomizer {
    public:
        struct ArgsRandomizer {};
        virtual void randomize(typename IComputeBlockNN<T, Enable>::ValuesForCompute*) = 0;
        virtual void setArgsForRandomize(const ArgsRandomizer*) = 0;
    };

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ITrainableComputeBlockNN : public IComputeBlockNN<T, Enable>, public ISaveable<T, Enable>, IOptimizeable<T, Enable>, IRandomizeable<T, Enable> {
    protected:
        struct IntermediateValues {};
    };

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
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
        virtual void setComputeBlock(std::unique_ptr<IComputeBlockNN<T, Enable>>) = 0;
        virtual const IComputeBlockNN<T, Enable>* getComputeBlock() = 0;

        virtual void setTranslatorMatrix(std::unique_ptr<ITranslatorMatrix<T, Enable>>) = 0;
        virtual const ITranslatorMatrix<T, Enable>* getTranslatorMatrix() = 0;
        virtual ~IBaseNN() = default;
    };

    template<typename T, typename Enable>
    class ITranslatorMatrix {
    public:
        virtual LinearAlgebra::BaseMatrix<T, Enable> operator()(typename IBaseNN<T, Enable>::InputValue) = 0;
        virtual typename IBaseNN<T, Enable>::OutputValue operator()(LinearAlgebra::BaseMatrix<T, Enable>) = 0;
    };

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class IBaseTrainableNN : public IBaseNN<T, Enable> {};

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class ComputeBlockNN : public IComputeBlockNN<T, Enable> {
    protected:
        using Values = typename IComputeBlockNN<T, Enable>::ValuesForCompute;

        std::unique_ptr<Values> values_for_compute_;
        LinearAlgebra::BaseMatrix<T, Enable> input_state_;
        std::uint64_t input_size_;
        std::uint64_t output_size_;
    public:
        ComputeBlockNN& operator=(const ComputeBlockNN& other);
        ComputeBlockNN& operator=(ComputeBlockNN&& other) noexcept;
        void setInput(LinearAlgebra::BaseMatrix<T, Enable> input) override;

        class Loader : public IComputeBlockNN<T, Enable>::ILoader {
        protected:
            std::unique_ptr<typename IComputeBlockNN<T, Enable>::ILoader::ArgsLoader> args_;
        };
        class Saver : public IComputeBlockNN<T, Enable>::ISaver {
        protected:
            std::unique_ptr<typename IComputeBlockNN<T, Enable>::ISaver::ArgsSaver> args_;
        };

        std::uint64_t getInputSize() {
            return this->input_size_;
        }
        std::uint64_t getOutputSize() {
            return this->output_size_;
        }
    };

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class TrainableComputeBlockNN : public ComputeBlockNN<T, Enable>, public ITrainableComputeBlockNN<T, Enable> {
    protected:
        using Base = ITrainableComputeBlockNN<T, Enable>;
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

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class BaseNN : public IBaseNN<T, Enable> {
    protected:
        using Base = IBaseNN<T, Enable>;

        std::unique_ptr<IComputeBlockNN<T, Enable>> compute_block_;
        std::unique_ptr<ITranslatorMatrix<T, Enable>> translator_;
        typename Base::InputValue input_state_;
        typename Base::OutputValue output_state_;

        void forward() override;
    public:
        BaseNN(std::unique_ptr<IComputeBlockNN<T, Enable>>, std::unique_ptr<ITranslatorMatrix<T, Enable>>);

        BaseNN& operator=(const BaseNN& other);
        BaseNN& operator=(BaseNN&& other) noexcept;
        void inference() override;
        void setComputeBlock(std::unique_ptr<IComputeBlockNN<T, Enable>> compute_block) override;
        void setTranslatorMatrix(std::unique_ptr<ITranslatorMatrix<T, Enable>> translator) override;
        const IComputeBlockNN<T, Enable>* getComputeBlock() override;
        const ITranslatorMatrix<T, Enable>* getTranslatorMatrix() override;
    };

    template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>>
    class BaseTrainableNN : public IBaseTrainableNN<T, Enable>, public BaseNN<T, Enable> {};

} // namespace MyNN
