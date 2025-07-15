#pragma once

#include <ActivateFunctionsForNN/HeaderActivateFunctionsForNN.h>

#include <fstream>

using ActivationFunctions::MatrixXld;
using ActivationFunctions::RowVectorXld; // ������-������
using ActivationFunctions::VectorXld;    // ������-�������

class SimpleLSTM {
	friend class BiLSTM;
public:

	SimpleLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_) {}

	SimpleLSTM();

	~SimpleLSTM() {}

	void SetInput_states(const std::vector<MatrixXld>& Input_states_) {}

	void SetWeights(const MatrixXld& weights_I_F, const MatrixXld& weights_I_I, const MatrixXld& weights_I_C, const MatrixXld& weights_I_O, 
		const MatrixXld& weights_H_F, const MatrixXld& weights_H_I, const MatrixXld& weights_H_C, const MatrixXld& weights_H_O){}

	void SetDisplacements(const MatrixXld& displacements_FG, const MatrixXld& displacements_IG, const MatrixXld& displacements_CT, const MatrixXld& displacements_OG) {}

	void SetRandomWeights(long double a = -0.2L, long double b = 0.2L) {}

	void SetRandomDisplacements(long double a = -0.5L, long double b = 0.5L) {}

	void All_state_�alculation() {}

	std::vector<RowVectorXld> GetLastOutputs() const {}


	/*std::vector<MatrixXld> GetWeightsAndDisplacement() {
		return {
			this->W_F_H, this->W_F_I, this->B_F,
			this->W_I_H, this->W_I_I, this->B_I,
			this->W_C_H, this->W_C_I, this->B_C,
			this->W_O_H, this->W_O_I, this->B_O
		};
	}*/

	static std::vector<std::vector<char>> denormalize(const MatrixXld& val) {}

	void save(const std::string& filename) const {}

	void load(const std::string& filename) {}

	void save_matrix(std::ofstream& file, const MatrixXld& m) const {}

	void load_matrix(std::ifstream& file, MatrixXld& m) {}

protected:
	/*struct LSTMGradients {
		// ��������� ��� Forget Gate
		MatrixXld dW_fg_hs;  // �� ����� hidden-state
		MatrixXld dW_fg_is;  // �� ����� input
		MatrixXld db_fg;     // �� ��������

		// ��������� ��� Input Gate
		MatrixXld dW_ig_hs;
		MatrixXld dW_ig_is;
		MatrixXld db_ig;

		// ��������� ��� Cell State
		MatrixXld dW_ct_hs;
		MatrixXld dW_ct_is;
		MatrixXld db_ct;

		// ��������� ��� Output Gate
		MatrixXld dW_og_hs;
		MatrixXld dW_og_is;
		MatrixXld db_og;
		std::vector<MatrixXld*> GetAll() {
			return {
				&dW_fg_hs, &dW_fg_is, &db_fg,
				&dW_ig_hs, &dW_ig_is, &db_ig,
				&dW_ct_hs, &dW_ct_is, &db_ct,
				&dW_og_hs, &dW_og_is, &db_og
			};
		}
	};*/

	Eigen::Index Input_size;
	Eigen::Index Hidden_size;
	std::vector<MatrixXld> Input_states;

	MatrixXld W_F_H;  // Forget gate hidden state weights
	MatrixXld W_I_H;  // Input gate hidden state weights
	MatrixXld W_C_H;  // Cell state hidden state weights
	MatrixXld W_O_H;  // Output gate hidden state weights

	MatrixXld W_F_I;  // Forget gate input weights
	MatrixXld W_I_I;  // Input gate input weights
	MatrixXld W_C_I;  // Cell state input weights
	MatrixXld W_O_I;  // Output gate input weights

	MatrixXld B_F;  // ������� 1xHidden_size
	MatrixXld B_I;  // ������� 1xHidden_size
	MatrixXld B_C;  // ������� 1xHidden_size
	MatrixXld B_O;  // ������� 1xHidden_size

	//MatrixXld Output_weights; // (Hidden_size x 1)
	//MatrixXld Output_bias;    // (1 x 1)

private:
	/*void n_state_�alculation(size_t timestep, size_t nstep) {
		RowVectorXld x_t(this->Input_size);
		RowVectorXld h_t_l = RowVectorXld::Zero(this->Hidden_size);
		RowVectorXld c_t_l = RowVectorXld::Zero(this->Hidden_size);

		// ��������� �����
		x_t = this->Input_states[nstep].row(timestep);

		// ���� timestep > 0, ���� ���������� ���������
		if (timestep > 0) {
			h_t_l = this->Hidden_states[nstep].row(timestep - 1);
			c_t_l = this->Cell_states[nstep].row(timestep - 1);
		}

		// ������������ ���� � ��������
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F_I, this->W_I_I, this->W_C_I, this->W_O_I;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->W_F_H, this->W_I_H, this->W_C_H, this->W_O_H;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		// ������ ������
		RowVectorXld Z_t = x_t * W_x + h_t_l * W_h;
		Z_t += b;

		RowVectorXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(this->Hidden_size));
		RowVectorXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(this->Hidden_size, this->Hidden_size));
		RowVectorXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * this->Hidden_size, this->Hidden_size));
		RowVectorXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(this->Hidden_size));

		RowVectorXld new_c_t = f_t.array() * c_t_l.array() + i_t.array() * c_t_bar.array();
		RowVectorXld new_h_t = o_t.array() * ActivationFunctions::Tanh(new_c_t).array();

		// ����������� ��������
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

		// ������ ����� ���������
		this->Hidden_states[nstep].row(timestep) = new_h_t;
		this->Cell_states[nstep].row(timestep) = new_c_t;
	}*/
	std::vector<MatrixXld> Hidden_states;
};

class BiLSTM {

public:
	BiLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_) {}

	BiLSTM();

	~BiLSTM() {}

	void All_state_�alculation() {}

	void SetInput_states(const std::vector<MatrixXld>& inputs) {}

	std::vector<RowVectorXld> GetFinalHidden_ForwardBackward() const {}

	void Save(const std::string& filename) {}

	void Load(const std::string& filename) {}
protected:
	Eigen::Index Common_Input_size;
	Eigen::Index Common_Hidden_size;
	std::vector<MatrixXld> Common_Input_states;
	std::vector<MatrixXld> Common_Hidden_states;
private:
	SimpleLSTM Forward;
	SimpleLSTM Backward;
};
