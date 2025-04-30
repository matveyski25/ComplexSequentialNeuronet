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
	struct weights_and_displacement_Neuron {
		long double weight_ForgetGate_for_HiddenState, weight_InputGate_for_HiddenState, weight_CandidateGate_for_HiddenState, weight_OutputGate_for_HiddenState;
		long double weight_ForgetGate_for_InputState, weight_InputGate_for_InputState, weight_CandidateGate_for_InputState, weight_OutputGate_for_InputState;
		long double displacement_ForgetState, displacement_InputState, displacement_CandidateState, displacement_OutputState;
		weights_and_displacement_Neuron(const std::vector <long double> weights, const std::vector <long double> displacements) {
			if(weights.size() > 0 && displacements.size() > 0) {
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
				if (weights.size() < 4) {
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
			}
		}
		weights_and_displacement_Neuron(const std::vector <long double> weights_and_displacements) {
			if(weights_and_displacements.size() > 0){
				std::vector <long double> weights_and_displacements_ = weights_and_displacements;
				if (weights_and_displacements.size() < 12) {
					weights_and_displacements_ = weights_and_displacements;
					weights_and_displacements_.resize(12);
					for (short i = weights_and_displacements.size() - 1; i < 8; i++) {
						weights_and_displacements_[i] = 0;
					}
					for (short j = 8; j < 12; j++) {
						weights_and_displacements_[j] = 0.5;
					}
				}
				else {
					weights_and_displacements_ = weights_and_displacements;
				}

				this->weight_ForgetGate_for_HiddenState = weights_and_displacements_[0];
				this->weight_InputGate_for_HiddenState = weights_and_displacements_[1];
				this->weight_CandidateGate_for_HiddenState = weights_and_displacements_[2];
				this->weight_OutputGate_for_HiddenState = weights_and_displacements_[3];

				this->weight_ForgetGate_for_InputState = weights_and_displacements_[4];
				this->weight_InputGate_for_InputState = weights_and_displacements_[5];
				this->weight_CandidateGate_for_InputState = weights_and_displacements_[6];
				this->weight_OutputGate_for_InputState = weights_and_displacements_[7];

				this->displacement_ForgetState = weights_and_displacements_[8];
				this->displacement_InputState = weights_and_displacements_[9];
				this->displacement_CandidateState = weights_and_displacements_[10];
				this->displacement_OutputState = weights_and_displacements_[11];
			}
		}
		weights_and_displacement_Neuron() = default;
	};
	SimpleSNT(size_t NumberNeurons_ = 1, const std::vector <weights_and_displacement_Neuron> weights_and_displacements = { { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} } }) {
		std::vector <weights_and_displacement_Neuron> weights_and_displacements_ = weights_and_displacements;
		if (NumberNeurons_ > weights_and_displacements.size()) {
			weights_and_displacements_.resize(NumberNeurons_);
			for (size_t i = weights_and_displacements.size() - 1; i < NumberNeurons_; i++) {
				weights_and_displacements_[i] = { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} };
			}
		}
		else if (NumberNeurons_ < weights_and_displacements.size()) {
			weights_and_displacements_.resize(NumberNeurons_);
		}
		this->Weights_and_Displacements_ = std::move(weights_and_displacements_);

		this->NumberNeurons = NumberNeurons_;
		this->OutputNeurons.resize(NumberNeurons, 0.0);
		this->InputNeurons.resize(NumberNeurons, 0.0);
		this->Hidden_state.resize(NumberNeurons, 0.0);
	}
	SimpleSNT(const std::vector <long double> InputNeurons_, const std::vector <weights_and_displacement_Neuron> weights_and_displacements = { { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} } }) {
		std::vector <weights_and_displacement_Neuron> weights_and_displacements_ = weights_and_displacements;
		if (InputNeurons_.size() > weights_and_displacements.size()) {
			weights_and_displacements_.resize(InputNeurons_.size());
			for (size_t i = weights_and_displacements.size() - 1; i < InputNeurons_.size(); i++) {
				weights_and_displacements_[i] = { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} };
			}
		}
		else if (InputNeurons_.size() < weights_and_displacements.size()) {
			weights_and_displacements_.resize(InputNeurons_.size());
		}

		this->Weights_and_Displacements_ = std::move(weights_and_displacements_);

		this->NumberNeurons = InputNeurons_.size();
		this->InputNeurons = InputNeurons_;
		this->OutputNeurons.resize(NumberNeurons, 0.0);
		this->Hidden_state.resize(NumberNeurons, 0.0);
	}
	SimpleSNT(std::vector <long double>&& InputNeurons_, const std::vector <weights_and_displacement_Neuron> weights_and_displacements = { { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} } }) {
		std::vector <weights_and_displacement_Neuron> weights_and_displacements_ = weights_and_displacements;
		if (InputNeurons_.size() > weights_and_displacements.size()) {
			weights_and_displacements_.resize(InputNeurons_.size());
			for (size_t i = weights_and_displacements.size() - 1; i < InputNeurons_.size(); i++) {
				weights_and_displacements_[i] = { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} };
			}
		}
		else if (InputNeurons_.size() < weights_and_displacements.size()) {
			weights_and_displacements_.resize(InputNeurons_.size());
		}

		this->Weights_and_Displacements_ = std::move(weights_and_displacements_);

		this->NumberNeurons = InputNeurons_.size();
		this->InputNeurons = std::move(InputNeurons_);
		this->OutputNeurons.resize(NumberNeurons, 0.0);
		this->Hidden_state.resize(NumberNeurons, 0.0);
	}
	~SimpleSNT() = default;
	void SetInputNeurons(const std::vector<long double> InputNeurons_, const std::vector <weights_and_displacement_Neuron> weights_and_displacements = { { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} } }) {
		std::vector <weights_and_displacement_Neuron> weights_and_displacements_ = weights_and_displacements;
		if (InputNeurons_.size() > weights_and_displacements.size()) {
			weights_and_displacements_.resize(InputNeurons_.size());
			for (size_t i = weights_and_displacements.size() - 1; i < InputNeurons_.size(); i++) {
				weights_and_displacements_[i] = { { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, {0.5, 0.5, 0.5, 0.5} };
			}
		}
		else if (InputNeurons_.size() < weights_and_displacements.size()) {
			weights_and_displacements_.resize(InputNeurons_.size());
		}

		this->Weights_and_Displacements_ = std::move(weights_and_displacements_);
		this->InputNeurons = InputNeurons_;	
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
	std::vector <weights_and_displacement_Neuron> Weights_and_Displacements_;

	std::vector <long double> neuronСalculation(long double Hidden_State, long double Last_State, long double Input_State, weights_and_displacement_Neuron weights_and_displacement) {
		// Forget Gate: решаем, что забыть
		long double ForgetGate = FunctionsAtivate::Sigmoid(
			weights_and_displacement.weight_ForgetGate_for_HiddenState * Hidden_State
			+ weights_and_displacement.weight_ForgetGate_for_InputState * Input_State
			+ weights_and_displacement.displacement_ForgetState
		);

		// Input Gate: решаем, что обновить
		long double InputGate = FunctionsAtivate::Sigmoid(
			weights_and_displacement.weight_InputGate_for_HiddenState * Hidden_State
			+ weights_and_displacement.weight_InputGate_for_InputState * Input_State 
			+ weights_and_displacement.displacement_InputState
		);

		// Новый кандидат для ячейки
		long double Ct_candidate = FunctionsAtivate::Tanh(
			weights_and_displacement.weight_CandidateGate_for_HiddenState * Hidden_State
			+ weights_and_displacement.weight_CandidateGate_for_InputState * Input_State
			+ weights_and_displacement.displacement_CandidateState
		);

		// Обновляем состояние ячейки
		auto Neuron = ForgetGate * Last_State + InputGate * Ct_candidate;

		// Output Gate: решаем, что передать в hidden_state
		long double OutputGate = FunctionsAtivate::Sigmoid(
			weights_and_displacement.weight_OutputGate_for_HiddenState * Hidden_State
			+ weights_and_displacement.weight_OutputGate_for_InputState * Input_State
			+ weights_and_displacement.displacement_OutputState
		);

		// Новое скрытое состояние
		auto NewHidden_state = OutputGate * FunctionsAtivate::Tanh(Neuron);

		return { Neuron, NewHidden_state };
	}

	void neuron_stateСalculation(size_t number) {
		if (number == 0) {
			auto vec_ = std::move(neuronСalculation(0, 0, this->InputNeurons[number], this->Weights_and_Displacements_[number]));
			this->OutputNeurons[number] = std::move(vec_[0]);
			this->Hidden_state[number] = std::move(vec_[1]);
		}
		else{
			auto vec_ = std::move(neuronСalculation(this->Hidden_state[number - 1], this->OutputNeurons[number - 1], this->InputNeurons[number], this->Weights_and_Displacements_[number]));
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