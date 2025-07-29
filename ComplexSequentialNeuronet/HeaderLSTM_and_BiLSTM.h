#pragma once

#include "ActivateFunctionsForNN/HeaderActivateFunctionsForNN.h"

#include <fstream>
#include <filesystem>

using ActivationFunctions::MatrixXld;
using ActivationFunctions::RowVectorXld; // Вектор-строка
using ActivationFunctions::VectorXld;    // Вектор-столбец

class SimpleLSTM {
	friend class BiLSTM;
public:

	SimpleLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_);

	SimpleLSTM() = default;

	~SimpleLSTM();

	void SetInput_states(const std::vector<MatrixXld>& Input_states_);

	void SetWeights(const MatrixXld& weights_I_F, const MatrixXld& weights_I_I, const MatrixXld& weights_I_C, const MatrixXld& weights_I_O, 
		const MatrixXld& weights_H_F, const MatrixXld& weights_H_I, const MatrixXld& weights_H_C, const MatrixXld& weights_H_O);

	void SetDisplacements(const MatrixXld& displacements_FG, const MatrixXld& displacements_IG, const MatrixXld& displacements_CT, const MatrixXld& displacements_OG);

	void SetRandomWeights(double a = -0.01, double b = 0.01);

	void SetRandomDisplacements(double a = -0.5L, double b = 0.5L);

	virtual void All_state_Сalculation();

	std::vector<RowVectorXld> GetLastOutputs() const;


	/*std::vector<MatrixXld> GetWeightsAndDisplacement() {
		return {
			this->U_F, this->W_F, this->B_F,
			this->U_I, this->W_I, this->B_I,
			this->U_C, this->W_C, this->B_C,
			this->U_O, this->W_O, this->B_O
		};
	}*/

	static std::vector<std::vector<char>> denormalize(const MatrixXld& val);

	void save(const std::string& filename) const;

	void load(const std::string& filename);

	void save_matrix(std::ofstream& file, const MatrixXld& m) const;

	void load_matrix(std::ifstream& file, MatrixXld& m);

//protected:

	Eigen::Index Input_size;
	Eigen::Index Hidden_size;
	std::vector<MatrixXld> Input_states;
protected:
	MatrixXld U_F;  // Forget gate hidden state weights
	MatrixXld U_I;  // Input gate hidden state weights
	MatrixXld U_C;  // Cell state hidden state weights
	MatrixXld U_O;  // Output gate hidden state weights

	MatrixXld W_F;  // Forget gate input weights
	MatrixXld W_I;  // Input gate input weights
	MatrixXld W_C;  // Cell state input weights
	MatrixXld W_O;  // Output gate input weights

	MatrixXld B_F;  // Матрица 1xHidden_size
	MatrixXld B_I;  // Матрица 1xHidden_size
	MatrixXld B_C;  // Матрица 1xHidden_size
	MatrixXld B_O;  // Матрица 1xHidden_size

	//MatrixXld Output_weights; // (Hidden_size x 1)
	//MatrixXld Output_bias;    // (1 x 1)

private:
	/*void n_state_Сalculation(size_t timestep, size_t nstep) {
		RowVectorXld x_t(this->Input_size);
		RowVectorXld h_t_l = RowVectorXld::Zero(this->Hidden_size);
		RowVectorXld c_t_l = RowVectorXld::Zero(this->Hidden_size);

		// Получение входа
		x_t = this->Input_states[nstep].row(timestep);

		// Если timestep > 0, берём предыдущие состояния
		if (timestep > 0) {
			h_t_l = this->Hidden_states[nstep].row(timestep - 1);
			c_t_l = this->Cell_states[nstep].row(timestep - 1);
		}

		// Объединенные веса и смещения
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F, this->W_I, this->W_C, this->W_O;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->U_F, this->U_I, this->U_C, this->U_O;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		// Расчёт выхода
		RowVectorXld Z_t = x_t * W_x + h_t_l * W_h;
		Z_t += b;

		RowVectorXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(this->Hidden_size));
		RowVectorXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(this->Hidden_size, this->Hidden_size));
		RowVectorXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * this->Hidden_size, this->Hidden_size));
		RowVectorXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(this->Hidden_size));

		RowVectorXld new_c_t = f_t.array() * c_t_l.array() + i_t.array() * c_t_bar.array();
		RowVectorXld new_h_t = o_t.array() * ActivationFunctions::Tanh(new_c_t).array();

		// Обеспечение размеров
		size_t total_sequences = this->Input_states.size();
		this->Hidden_states.resize(total_sequences);
		this->Cell_states.resize(total_sequences);

		for (size_t i = 0; i < total_sequences; ++i) {
			Eigen::Index T = timestep + 1;
			if (this->Hidden_states[i].rows() < T) {
				this->Hidden_states[i].conservativeResize(T, this->Hidden_size);
				this->Hidden_states[i].row(T - 1).setZero();
			}
			if (this->Cell_states[i].rows() < T) {
				this->Cell_states[i].conservativeResize(T, this->Hidden_size);
				this->Cell_states[i].row(T - 1).setZero();
			}
		}

		// Запись новых состояний
		this->Hidden_states[nstep].row(timestep) = new_h_t;
		this->Cell_states[nstep].row(timestep) = new_c_t;
	}*/
	std::vector<MatrixXld> Hidden_states;
};

class BiLSTM {

public:
	BiLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_);

	BiLSTM() = default;

	~BiLSTM();

	void All_state_Сalculation();

	void SetInput_states(const std::vector<MatrixXld>& inputs);

	std::vector<RowVectorXld> GetFinalHidden_ForwardBackward() const;

	void Save(const std::string& filename);

	void Load(const std::string& filename);
protected:
	Eigen::Index Common_Input_size;
	Eigen::Index Common_Hidden_size;
	std::vector<MatrixXld> Common_Input_states;
	std::vector<MatrixXld> Common_Hidden_states;
private:
	SimpleLSTM Forward;
	SimpleLSTM Backward;
};

class SimpleLSTM_ForTrain : public SimpleLSTM {
	friend class BiLSTM_ForTrain;////////////////
	friend class Seq2SeqWithAttention_ForTrain;//////////////
public:
	SimpleLSTM_ForTrain(size_t Batch_size_, Eigen::Index Number_states, Eigen::Index Hidden_size_);

	SimpleLSTM_ForTrain() = default;

	~SimpleLSTM_ForTrain();

	void SetInput_states(const std::vector<MatrixXld>& Input_states_);

	void save(const std::string& filename) const;

	void load(const std::string& filename);

protected:
	size_t Batch_size;

	void Batch_All_state_Сalculation();

	struct states_forgrads { std::vector<MatrixXld> f, i, ccond, o, c, h; };

	states_forgrads statesForgrads;
};

class BiLSTM_ForTrain : public BiLSTM {
	friend class Seq2SeqWithAttention_ForTrain;////////////////
public:
	BiLSTM_ForTrain(size_t Batch_size_, Eigen::Index Number_states, Eigen::Index Hidden_size_);

	BiLSTM_ForTrain() = default;

	void SetInput_states(const std::vector<MatrixXld>& inputs);

	std::vector<RowVectorXld> GetFinalHidden_ForwardBackward() const;

	void Save(const std::string& filename);

	void Load(const std::string& filename);

	~BiLSTM_ForTrain();

	void Batch_All_state_Сalculation();
protected:
	SimpleLSTM_ForTrain Forward;
	SimpleLSTM_ForTrain Backward;
	size_t Common_Batch_size;
};

class Attention {
public:
	struct AttnOutput {
		RowVectorXld context;
		std::vector<RowVectorXld> u_t;  // size = time_steps_enc
		VectorXld      alpha;          // size = time_steps_enc
	};

	virtual ~Attention() = default;

	// Абстрактный метод: вычисляет контекст по шагу
	virtual AttnOutput ComputeContext(
		const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden
	) = 0;
};

class BahdanauAttention : public Attention {
public:
	friend class Seq2SeqWithAttention_ForTrain;////////////
	BahdanauAttention(Eigen::Index encoder_hidden_size, Eigen::Index decoder_hidden_size, Eigen::Index attention_size);

	AttnOutput ComputeContext(
		const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden
	) override;

	void SetRandomWeights(double a = -0.01, double b = 0.01);
protected:
	Eigen::Index duo_encoder_hidden_size_;    // 2H
	Eigen::Index decoder_hidden_size_;    // H_dec
	Eigen::Index attention_size_;         // A

	MatrixXld W_encoder_;       // [A x 2H]
	MatrixXld W_decoder_;       // [A x H_dec]
	VectorXld attention_vector_; // [A x 1]

};