#pragma once
#include<type_traits>
#include<memory>
#include "RealizationMatrix.hpp"


namespace MyNN {
	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>> 
	class IComputeBlockNN {
	public:
		/*The structure inherited from the base interface IBaseArgs - Структура наследующаяся от общей IBaseArgs*/
		struct ValuesForCompute {
			struct Weights {};
			struct Bias {};
		};
		virtual void setValuesForCompute(ValuesForCompute* values) = 0;
		virtual void setInput(LinearAlgebra::BaseMatrix<T>  input) = 0;
		virtual LinearAlgebra::BaseMatrix<T> getOutput() = 0;
		virtual void compute() = 0;

	};
	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>> 
	class ITrainableComputeBlockNN : public IComputeBlockNN<T> {
	protected:
		/*The struct for intermediate values from computing for future trining - Структура для промежуточных значений для будущего обучения*/
		struct IntermediateValues {};
	public:
		class ISaver {
			struct ArgsSaver {};
			virtual void save(IComputeBlockNN<T>* compute_block) = 0;
			virtual void setArgsForSave(ArgsSaver* args) = 0;
		};
		
		class ILoader {
		public:
			struct ArgsLoader {};
			virtual void load(IComputeBlockNN<T>* compute_block) = 0;
			virtual void setArgsForLoad(ArgsLoader* args) = 0;
		};

		class IOptimizer {
		public:
			struct ArgsOptimizer {};
			struct Gradients : IComputeBlockNN<T>::ValuesForCompute {};
			virtual void optimize(ArgsOptimizer* values) = 0;
			virtual void setArgsForOptimize(const Gradients* gradients, IComputeBlockNN<T>::ValuesForCompute* values_for_compute) = 0;
		};

		class IRandomizer {
		public:
			struct ArgsRandomizer {};
			virtual void randomize(IComputeBlockNN<T>::ValuesForCompute* values_for_compute) = 0;
			virtual void setArgsForRandomize(ArgsRandomizer* args) = 0;
		};

		virtual void setSaver(ISaver* saver) = 0;
		virtual const ISaver* getSaver() = 0;

		virtual void setLoader(ILoader* loader) = 0;
		virtual const ILoader* getLoader() = 0;

		virtual void setOptimizer(IOptimizer* optimizer) = 0;
		virtual const IOptimizer* getOptimizer() = 0;

		virtual void setRandomizer(IRandomizer* randomizer) = 0;
		virtual const IRandomizer* getRandomizer() = 0;
	};

	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>> 
	class IBaseNN {
	protected:
		virtual void forward() = 0;
	public:
		struct InputValue {};
		struct OutputValue {};

		virtual void inference() = 0;
		virtual void setInputState(InputValue input_state) = 0;
		virtual InputValue getInputState() = 0;
		virtual OutputValue getOutputState() = 0;
		virtual void setComputeBlock(IComputeBlockNN<T>* compute_block) = 0;
		virtual const IComputeBlockNN<T>* getComputeBlock() = 0;
	};
	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>> 
	class IBaseTrainableNN : public IBaseNN<T> {};

	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	class ComputeBlockNN : public IComputeBlockNN<T> {
	protected:
		std::uint64_t input_size;
		std::uint64_t output_size;
		ValuesForCompute values_for_compute_;
	public:
		ComputeBlockNN<T>& operator=(const ComputeBlockNN<T>& other) {
			this->values_for_compute_ = other.values_for_compute_;
			this->input_size = other.input_size;
			this->output_size = other.output_size;
		}
		ComputeBlockNN<T>& operator=(ComputeBlockNN<T>&& other) {
			this->values_for_compute_ = std::move(other.values_for_compute_);
			this->input_size = std::move(other.input_size);
			this->output_size = std::move(other.output_size);
		}
	};
	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	class TrainableComputeBlockNN : public ComputeBlockNN<T>, public ITrainableComputeBlockNN<T> {
	protected:
		std::unique_ptr<ISaver> saver_;
		std::unique_ptr<ILoader> loader_;
		std::unique_ptr<IOptimizer> optimizer_;
		std::unique_ptr<IRandomizer> randomizer_;
		IntermediateValues intermediate_values_;
	public:
		TrainableComputeBlockNN<T>& operator=(const TrainableComputeBlockNN<T> & other) {
			*(this->saver_) = *(other.saver_);
			*(this->loader_) = *(other.loader_);
			*(this->optimizer_) = *(other.optimizer_);
			*(this->randomizer_) = *(other.randomizer_);
			this->intermediate_values_ = other.intermediate_values_;
			ComputeBlockNN<T>::operator=(other);
		}
		TrainableComputeBlockNN<T>& operator=(TrainableComputeBlockNN<T>&& other) {
			this->saver_ = std::move(other.saver_);
			this->loader_ = std::move(other.loader_);
			this->optimizer_ = std::move(other.optimizer_);
			this->randomizer_ = std::move(other.randomizer_);
			this->intermediate_values_ = other.intermediate_values_;
			ComputeBlockNN<T>::operator=(std::move(other));
			other.saver_ = nullptr;
			other.loader_ = nullptr;
			other.optimizer_ = nullptr;
			other.randomizer_ = nullptr;
		}
	};
	
	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	class BaseNN : public IBaseNN<T> {
	protected:
		InputValue input_state_;
		OutputValue output_state_;
		std::unique_ptr<ComputeBlockNN> compute_block_;
		BaseNN& operator=(const BaseNN& other) {
			*(this->compute_block_) = *(other.compute_block_);
			this->input_state_ = other.input_state_;
			this->ouput_state_ = other.ouput_state_;
			ComputeBlockNN<T>::operator=(other);
		}
		BaseNN& operator=(BaseNN&& other) {
			this->compute_block = std::move(other.compute_block_);
			this->input_state_ = std::move(other.input_state_);
			this->ouput_state_ = std::move(other.ouput_state_);
			ComputeBlockNN<T>::operator=(std::move(other));
			other.compute_block_ = nullptr;
		}
	};
	template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	class BaseTrainableNN : public IBaseTrainableNN<T>, public BaseNN<T> {};
}