#include "HeaderLib_ComplexSequentialNeuronet.h"

//using namespace std;
namespace FunctionsAtivate {
	unsigned short StepFunction(long double value) {
		if (value >= 1) {
			return 1;
		}
		else {
			return 0;
		}
	}
	long double Sigmoid(long double value) {
		return 1 / (1 + std::expl((-1 * value)));
	}
	long double Tanh(long double value) {
		return ((std::expl(value) - std::expl(-1 * value)) / (std::expl(value) + std::expl(-1 * value)));
	}
	long double ReLU(long double value) {
		return std::fmaxl(0, value);
	}
	long double LeakyReLU(long double value, long double a = 0.001) {
		if (value >= 0) {
			return value;
		}
		else {
			return (a * value);
		}
	}
	long double Swish(long double value, long double b = 0.001) {
		return (value * Sigmoid((value * b)));
	}
	std::vector<long double> Softmax(std::vector<long double> values) {
		long double sum = 0;
		long double max_val = *std::max_element(values.begin(), values.end());
		for (auto& v : values) {
			v = (v > 709) ? 709 : v; // Ограничение сверху
			sum += std::exp(v - max_val);
		}
		std::vector<long double> result;
		if (sum == 0) {
			// Например, равномерное распределение:
			result.resize(values.size(), 1.0 / values.size());
		}
		else {
			for (size_t j = 0; j < values.size(); j++) {
				result.push_back(std::exp(values[j] - max_val) / sum);
			}
		}
		return result;
	}
}

class SimpleSNT {
public:
	SimpleSNT(size_t NumberNeurons_ = 1, const std::vector <long double> weights = {0, 0, 0, 0, 0, 0, 0, 0}, const std::vector <long double> displacements = { 0, 0, 0, 0}) {
		if (weights.size() > 0 && displacements.size() > 0) {
			std::vector <long double> weights_ = weights;
			if (weights.size() < 8) {
				weights_ = weights;
				weights_.resize(8);
				for (short i = weights.size() - 1; i < 8; i++) {
					weights_[i] = 0;
				}
			}
			else {
				weights_ = weights;
			}

			std::vector <long double> displacements_ = displacements;
			if (displacements.size() < 4) {
				displacements_ = displacements;
				displacements_.resize(4);
				for (short i = displacements.size() - 1; i < 4; i++) {
					displacements_[i] = 0.5;
				}
			}
			else {
				displacements_ = displacements;
			}

			this->weight_ForgetGate_for_HiddenState = weights_[0];
			this->weight_InputGate_for_HiddenState = weights_[1];
			this->weight_CandidateGate_for_HiddenState = weights_[2];
			this->weight_OutputGate_for_HiddenState = weights_[3];

			this->weight_ForgetGate_for_InputState = weights_[4];
			this->weight_InputGate_for_InputState = weights_[5];
			this->weight_CandidateGate_for_InputState = weights_[6];
			this->weight_OutputGate_for_InputState = weights_[7];

			this->displacement_ForgetState = displacements_[0];
			this->displacement_InputState = displacements_[1];
			this->displacement_CandidateState = displacements_[2];
			this->displacement_OutputState = displacements_[3];

			this->NumberNeurons = NumberNeurons_;
			this->OutputNeurons.resize(NumberNeurons, 0.0);
			this->InputNeurons.resize(NumberNeurons, 0.0);
			this->Hidden_state.resize(NumberNeurons, 0.0);
		}
	}
	SimpleSNT(const std::vector <long double> InputNeurons_, const std::vector <long double> weights, const std::vector <long double> displacements) {
		if (weights.size() > 0 && displacements.size() > 0) {
			std::vector <long double> weights_ = weights;
			if (weights.size() < 8) {
				weights_ = weights;
				weights_.resize(8);
				for (short i = weights.size() - 1; i < 8; i++) {
					weights_[i] = 0;
				}
			}
			else {
				weights_ = weights;
			}

			std::vector <long double> displacements_ = displacements;
			if (displacements.size() < 4) {
				displacements_ = displacements;
				displacements_.resize(4);
				for (short i = displacements.size() - 1; i < 4; i++) {
					displacements_[i] = 0.5;
				}
			}
			else {
				displacements_ = displacements;
			}

			this->weight_ForgetGate_for_HiddenState = weights_[0];
			this->weight_InputGate_for_HiddenState = weights_[1];
			this->weight_CandidateGate_for_HiddenState = weights_[2];
			this->weight_OutputGate_for_HiddenState = weights_[3];

			this->weight_ForgetGate_for_InputState = weights_[4];
			this->weight_InputGate_for_InputState = weights_[5];
			this->weight_CandidateGate_for_InputState = weights_[6];
			this->weight_OutputGate_for_InputState = weights_[7];

			this->displacement_ForgetState = displacements_[0];
			this->displacement_InputState = displacements_[1];
			this->displacement_CandidateState = displacements_[2];
			this->displacement_OutputState = displacements_[3];

			this->NumberNeurons = InputNeurons_.size();
			this->InputNeurons = InputNeurons_;
			this->OutputNeurons.resize(NumberNeurons, 0.0);
			this->Hidden_state.resize(NumberNeurons, 0.0);
		}
	}
	SimpleSNT(std::vector <long double>&& InputNeurons_, const std::vector <long double> weights, const std::vector <long double> displacements) {
		if (weights.size() > 0 && displacements.size() > 0) {
			std::vector <long double> weights_ = weights;
			if (weights.size() < 8) {
				weights_ = weights;
				weights_.resize(8);
				for (short i = weights.size() - 1; i < 8; i++) {
					weights_[i] = 0;
				}
			}
			else {
				weights_ = weights;
			}

			std::vector <long double> displacements_ = displacements;
			if (displacements.size() < 4) {
				displacements_ = displacements;
				displacements_.resize(4);
				for (short i = displacements.size() - 1; i < 4; i++) {
					displacements_[i] = 0.5;
				}
			}
			else {
				displacements_ = displacements;
			}

			this->weight_ForgetGate_for_HiddenState = weights_[0];
			this->weight_InputGate_for_HiddenState = weights_[1];
			this->weight_CandidateGate_for_HiddenState = weights_[2];
			this->weight_OutputGate_for_HiddenState = weights_[3];

			this->weight_ForgetGate_for_InputState = weights_[4];
			this->weight_InputGate_for_InputState = weights_[5];
			this->weight_CandidateGate_for_InputState = weights_[6];
			this->weight_OutputGate_for_InputState = weights_[7];

			this->displacement_ForgetState = displacements_[0];
			this->displacement_InputState = displacements_[1];
			this->displacement_CandidateState = displacements_[2];
			this->displacement_OutputState = displacements_[3];

			this->NumberNeurons = InputNeurons_.size();
			this->InputNeurons = std::move(InputNeurons_);
			this->OutputNeurons.resize(NumberNeurons, 0.0);
			this->Hidden_state.resize(NumberNeurons, 0.0);
		}
	}
	~SimpleSNT() = default;
	void SetInputNeurons(const std::vector<long double> InputNeurons_) {
		this->InputNeurons = InputNeurons_;	
	}
	void SetWeights(const std::vector <long double> weights) {
		std::vector <long double> weights_ = weights;
		if (weights.size() < 8) {
			weights_ = weights;
			weights_.resize(8);
			for (short i = weights.size() - 1; i < 8; i++) {
				weights_[i] = 0;
			}
		}
		else {
			weights_ = weights;
		}
		this->weight_ForgetGate_for_HiddenState = weights_[0];
		this->weight_InputGate_for_HiddenState = weights_[1];
		this->weight_CandidateGate_for_HiddenState = weights_[2];
		this->weight_OutputGate_for_HiddenState = weights_[3];

		this->weight_ForgetGate_for_InputState = weights_[4];
		this->weight_InputGate_for_InputState = weights_[5];
		this->weight_CandidateGate_for_InputState = weights_[6];
		this->weight_OutputGate_for_InputState = weights_[7];
	}
	void SetDisplacements(const std::vector <long double> displacements) {
		std::vector <long double> displacements_ = displacements;
		if (displacements.size() < 4) {
			displacements_ = displacements;
			displacements_.resize(4);
			for (short i = displacements.size() - 1; i < 4; i++) {
				displacements_[i] = 0.5;
			}
		}
		else {
			displacements_ = displacements;
		}

		this->displacement_ForgetState = displacements_[0];
		this->displacement_InputState = displacements_[1];
		this->displacement_CandidateState = displacements_[2];
		this->displacement_OutputState = displacements_[3];
	}
	void SetAll(const std::vector <long double> InputNeurons_, const std::vector <long double> weights, const std::vector <long double> displacements) {
		SetInputNeurons(InputNeurons_);
		SetWeights(weights);
		SetDisplacements(displacements);
	}
	void CalculationAllNeurons() {
		for (size_t i = 0; i < this->InputNeurons.size(); i++) {
			neuron_stateСalculation(i);
		}
	}
	std::vector <long double> GetOutputNeurons() {
		std::vector <long double> result = this->OutputNeurons;
		return result;
	}
private:
	size_t NumberNeurons;
	std::vector <long double> InputNeurons;
	std::vector <long double> Hidden_state;
	std::vector <long double> OutputNeurons;
	long double weight_ForgetGate_for_HiddenState, weight_InputGate_for_HiddenState, weight_CandidateGate_for_HiddenState, weight_OutputGate_for_HiddenState;
	long double weight_ForgetGate_for_InputState, weight_InputGate_for_InputState, weight_CandidateGate_for_InputState, weight_OutputGate_for_InputState;
	long double displacement_ForgetState, displacement_InputState, displacement_CandidateState, displacement_OutputState;

	std::vector <long double> neuronСalculation(long double Hidden_State, long double Last_State, long double Input_State) {
		// Forget Gate: решаем, что забыть
		long double ForgetGate = FunctionsAtivate::Sigmoid(
			this->weight_ForgetGate_for_HiddenState * Hidden_State
			+ this->weight_ForgetGate_for_InputState * Input_State
			+ this->displacement_ForgetState
		);

		// Input Gate: решаем, что обновить
		long double InputGate = FunctionsAtivate::Sigmoid(
			this->weight_InputGate_for_HiddenState * Hidden_State
			+ this->weight_InputGate_for_InputState * Input_State 
			+ this->displacement_InputState
		);

		// Новый кандидат для ячейки
		long double Ct_candidate = FunctionsAtivate::Tanh(
			this->weight_CandidateGate_for_HiddenState * Hidden_State
			+ this->weight_CandidateGate_for_InputState * Input_State
			+ this->displacement_CandidateState
		);

		// Обновляем состояние ячейки
		auto Neuron = ForgetGate * Last_State + InputGate * Ct_candidate;

		// Output Gate: решаем, что передать в hidden_state
		long double OutputGate = FunctionsAtivate::Sigmoid(
			this->weight_OutputGate_for_HiddenState * Hidden_State
			+ this->weight_OutputGate_for_InputState * Input_State
			+ this->displacement_OutputState
		);

		// Новое скрытое состояние
		auto NewHidden_state = OutputGate * FunctionsAtivate::Tanh(Neuron);

		return { Neuron, NewHidden_state };
	}

	void neuron_stateСalculation(size_t number) {
		if (number == 0) {
			auto vec_ = std::move(neuronСalculation(0, 0, this->InputNeurons[number]));
			this->OutputNeurons[number] = std::move(vec_[0]);
			this->Hidden_state[number] = std::move(vec_[1]);
		}
		else{
			auto vec_ = std::move(neuronСalculation(this->Hidden_state[number - 1], this->OutputNeurons[number - 1], this->InputNeurons[number]));
			this->OutputNeurons[number] = std::move(vec_[0]);
			this->Hidden_state[number] = std::move(vec_[1]);
		}
	}
};

void main() {
	SimpleSNT a(3);
	a.CalculationAllNeurons();
	auto b = a.GetOutputNeurons();
	for (int i = 0; i < b.size(); i++) {
		std::cout << b[i] << "\t";
	}
	a.SetInputNeurons({ 50, 3, 4 });
	a.CalculationAllNeurons();
	b = a.GetOutputNeurons();
	for (int i = 0; i < b.size(); i++) {
		std::cout << b[i] << "\t";
	}
}