#include "HeaderLSTM_and_BiLSTM.h"

class SimpleLSTM {
	friend class BiLSTM;
public:

	SimpleLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_) {
		if (Hidden_size_ <= 0) {
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}

		this->Input_size = Number_states;
		this->Hidden_size = Hidden_size_;

		// Инициализация весов
		SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

		// Инициализация смещений (1xHidden_size)
		SetRandomDisplacements(-1.5L, 1.5L);

		// Инициализация состояний
		//Input_states = MatrixXld(0, this->Input_size); // Пустая матрица
		//Cell_states = MatrixXld::Zero(1, Hidden_size_);
		//Hidden_states = MatrixXld::Zero(1, Hidden_size_);
	}

	SimpleLSTM() = default;

	~SimpleLSTM() {
		save("LSTM_state.txt");
	}

	void SetInput_states(const std::vector<MatrixXld>& Input_states_) {
		for (const auto& b : Input_states_) {
			if (b.rows() == 0 || b.cols() != Input_size) {
				throw std::invalid_argument("Invalid input matrix dimensions");
			}
		}

		Input_states = Input_states_;
		//size_t steps = Input_states.rows();

		// Корректная инициализация размеров
		//Cell_states = MatrixXld::Zero(steps + 1, Hidden_size);
		//Hidden_states = MatrixXld::Zero(steps + 1, Hidden_size);
	}

	void SetWeights(const MatrixXld& weights_I_F, const MatrixXld& weights_I_I, const MatrixXld& weights_I_C, const MatrixXld& weights_I_O, const MatrixXld& weights_H_F, const MatrixXld& weights_H_I, const MatrixXld& weights_H_C, const MatrixXld& weights_H_O)
	{
		// Проверка размеров весов
		auto check_weights = [&](const MatrixXld& W, size_t rows, size_t cols) {
			return W.rows() == rows && W.cols() == cols;
			};

		if (!check_weights(weights_I_F, Hidden_size, Input_size) ||
			!check_weights(weights_I_I, Hidden_size, Input_size) ||
			!check_weights(weights_I_C, Hidden_size, Input_size) ||
			!check_weights(weights_I_O, Hidden_size, Input_size) ||
			!check_weights(weights_H_F, Hidden_size, Hidden_size) ||
			!check_weights(weights_H_I, Hidden_size, Hidden_size) ||
			!check_weights(weights_H_C, Hidden_size, Hidden_size) ||
			!check_weights(weights_H_O, Hidden_size, Hidden_size))
		{
			throw std::invalid_argument("Invalid weights dimensions");
		}

		this->W_F_I = weights_I_F;
		this->W_I_I = weights_I_I;
		this->W_C_I = weights_I_C;
		this->W_O_I = weights_I_O;

		this->W_F_H = weights_H_F;
		this->W_I_H = weights_H_I;
		this->W_C_H = weights_H_C;
		this->W_O_H = weights_H_O;
	}

	void SetDisplacements(const MatrixXld& displacements_FG, const MatrixXld& displacements_IG, const MatrixXld& displacements_CT, const MatrixXld& displacements_OG) {
		auto check_displacement = [&](const MatrixXld& m) {
			return m.rows() == 1 && m.cols() == Hidden_size;
			};

		if (!check_displacement(displacements_FG) ||
			!check_displacement(displacements_IG) ||
			!check_displacement(displacements_CT) ||
			!check_displacement(displacements_OG))
		{
			throw std::invalid_argument("Displacements must be 1xHidden_size matrices");
		}

		B_F = displacements_FG;
		B_I = displacements_IG;
		B_C = displacements_CT;
		B_O = displacements_OG;
	}

	void SetRandomWeights(long double a = -0.2L, long double b = 0.2L) {
		//ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		auto xavier_init = [](size_t rows, size_t cols, long double c = 2.0L) {
			long double range = sqrt(c / (rows + cols));
			return ActivationFunctions::matrix_random(rows, cols, -range, range);
			};
		auto orthogonal_init = [](size_t rows, size_t cols) {
			MatrixXld mat = MatrixXld::Random(rows, cols);
			Eigen::HouseholderQR<MatrixXld> qr(mat);
			return qr.householderQ() * MatrixXld::Identity(rows, cols);
			};
		auto random_init = [](size_t rows, size_t cols, long double a_, long double b_) { return ActivationFunctions::matrix_random(rows, cols, a_, b_); };
		auto init_h = orthogonal_init;
		auto init_i = xavier_init;
		this->W_F_H = init_h(Hidden_size, Hidden_size);
		this->W_I_H = init_h(Hidden_size, Hidden_size);
		this->W_C_H = init_h(Hidden_size, Hidden_size);
		this->W_O_H = init_h(Hidden_size, Hidden_size);

		this->W_F_I = init_i(Hidden_size, Input_size);
		this->W_I_I = init_i(Hidden_size, Input_size);
		this->W_C_I = init_i(Hidden_size, Input_size);
		this->W_O_I = init_i(Hidden_size, Input_size);

		//Output_weights = ActivationFunctions::matrix_random(Hidden_size, 1, -0.1L, 0.1L);
	}

	void SetRandomDisplacements(long double a = -0.5L, long double b = 0.5L) {
		this->B_F = MatrixXld::Constant(1, Hidden_size, 1.0L);
		this->B_I = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->B_C = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->B_O = ActivationFunctions::matrix_random(1, Hidden_size, a, b);

		//Output_bias = MatrixXld::Zero(1, 1);
	}

	void All_state_Сalculation() {
		// Подготовка весов вне цикла
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F_I, this->W_I_I, this->W_C_I, this->W_O_I;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->W_F_H, this->W_I_H, this->W_C_H, this->W_O_H;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		for (size_t nstep = 0; nstep < this->Input_states.size(); ++nstep) {
			size_t sequence_length = this->Input_states[nstep].rows();

			RowVectorXld h_t_l = RowVectorXld::Zero(this->Hidden_size);
			RowVectorXld c_t_l = RowVectorXld::Zero(this->Hidden_size);

			for (size_t timestep = 0; timestep < sequence_length; ++timestep) {
				RowVectorXld x_t = this->Input_states[nstep].row(timestep);

				RowVectorXld Z_t = x_t * W_x + h_t_l * W_h;
				Z_t += b;

				RowVectorXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(this->Hidden_size));
				RowVectorXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(this->Hidden_size, this->Hidden_size));
				RowVectorXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * this->Hidden_size, this->Hidden_size));
				RowVectorXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(this->Hidden_size));

				RowVectorXld new_c_t = f_t.array() * c_t_l.array() + i_t.array() * c_t_bar.array();
				RowVectorXld new_h_t = o_t.array() * ActivationFunctions::Tanh(new_c_t).array();

				this->Hidden_states[nstep].col(timestep) = new_h_t;
				h_t_l = new_h_t;
				c_t_l = new_c_t;
			}
		}
	}

	std::vector<RowVectorXld> GetLastOutputs() const {
		std::vector<RowVectorXld> outputs;
		for (const auto& state : this->Hidden_states) {
			if (state.rows() > 0) {
				outputs.push_back(state.row(state.rows() - 1));
			}
		}
		return outputs;
	}


	/*std::vector<MatrixXld> GetWeightsAndDisplacement() {
		return {
			this->W_F_H, this->W_F_I, this->B_F,
			this->W_I_H, this->W_I_I, this->B_I,
			this->W_C_H, this->W_C_I, this->B_C,
			this->W_O_H, this->W_O_I, this->B_O
		};
	}*/

	static long double normalize(char c) {
		return static_cast<long double>(static_cast<unsigned char>(c)) / 127.5L - 1.0L;
	}

	static char denormalize(const long double val) {
		long double denorm_value = (val * 127.5L) + 1L;
		denorm_value = std::max<long double>(
			denorm_value,
			static_cast<long double>(std::numeric_limits<char>::min())
		);
		denorm_value = std::min<long double>(
			denorm_value,
			static_cast<long double>(std::numeric_limits<char>::max())
		);
		return static_cast<char>(std::round(denorm_value));
	}

	static std::vector<std::vector<char>> denormalize(const MatrixXld& val) {
		if (val.rows() == 0 || val.cols() == 0) {
			return {};
		}

		const Eigen::Index rows = val.rows();
		const Eigen::Index cols = val.cols();
		std::vector<std::vector<char>> result(rows, std::vector<char>(cols));

		for (Eigen::Index i = 0; i < rows; ++i) {
			for (Eigen::Index j = 0; j < cols; ++j) {
				result[i][j] = denormalize(val(i, j));
			}
		}

		return result;
	}

	void save(const std::string& filename) const {
		std::ofstream file(filename, std::ios::trunc); // Используйте trunc для перезаписи
		if (!file) throw std::runtime_error("Cannot open file for writing");

		// Сохраняем только актуальные параметры
		file << this->Input_size << "\n" << this->Hidden_size << "\n";

		// Сохраняем веса и смещения
		save_matrix(file, this->W_F_H);
		save_matrix(file, this->W_I_H);
		save_matrix(file, this->W_C_H);
		save_matrix(file, this->W_O_H);

		save_matrix(file, this->W_F_I);
		save_matrix(file, this->W_I_I);
		save_matrix(file, this->W_C_I);
		save_matrix(file, this->W_O_I);

		save_matrix(file, this->B_F);
		save_matrix(file, this->B_I);
		save_matrix(file, this->B_C);
		save_matrix(file, this->B_O);
	}

	void load(const std::string& filename) {
		std::ifstream file(filename);
		if (!file) throw std::runtime_error("Cannot open file for reading");

		file >> this->Input_size >> this->Hidden_size;

		load_matrix(file, this->W_F_H);
		load_matrix(file, this->W_I_H);
		load_matrix(file, this->W_C_H);
		load_matrix(file, this->W_O_H);

		load_matrix(file, this->W_F_I);
		load_matrix(file, this->W_I_I);
		load_matrix(file, this->W_C_I);
		load_matrix(file, this->W_O_I);

		load_matrix(file, this->B_F);
		load_matrix(file, this->B_I);
		load_matrix(file, this->B_C);
		load_matrix(file, this->B_O);
	}

	void save_matrix(std::ofstream& file, const MatrixXld& m) const {
		file << m.rows() << " " << m.cols() << "\n";
		for (Eigen::Index i = 0; i < m.rows(); ++i) {
			for (Eigen::Index j = 0; j < m.cols(); ++j) {
				file << m(i, j) << " ";
			}
			file << "\n";
		}
	}

	void load_matrix(std::ifstream& file, MatrixXld& m) {
		Eigen::Index rows, cols;
		file >> rows >> cols;
		m = MatrixXld(rows, cols);
		for (Eigen::Index i = 0; i < rows; ++i) {
			for (Eigen::Index j = 0; j < cols; ++j) {
				file >> m(i, j);
			}
		}
	}

protected:
	/*struct LSTMGradients {
		// Градиенты для Forget Gate
		MatrixXld dW_fg_hs;  // по весам hidden-state
		MatrixXld dW_fg_is;  // по весам input
		MatrixXld db_fg;     // по смещению

		// Градиенты для Input Gate
		MatrixXld dW_ig_hs;
		MatrixXld dW_ig_is;
		MatrixXld db_ig;

		// Градиенты для Cell State
		MatrixXld dW_ct_hs;
		MatrixXld dW_ct_is;
		MatrixXld db_ct;

		// Градиенты для Output Gate
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
		W_x << this->W_F_I, this->W_I_I, this->W_C_I, this->W_O_I;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->W_F_H, this->W_I_H, this->W_C_H, this->W_O_H;

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
	BiLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_) {
		if (Hidden_size_ <= 0) {
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}
		this->Common_Input_size = Number_states;
		this->Common_Hidden_size = Hidden_size_;

		this->Forward = SimpleLSTM(this->Common_Input_size, this->Common_Hidden_size);
		this->Backward = SimpleLSTM(this->Common_Input_size, this->Common_Hidden_size);
		// Инициализация весов
		//SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

		// Инициализация смещений (1xHidden_size)
		//SetRandomDisplacements(-1.5L, 1.5L);

		// Инициализация состояний
		//Input_states = MatrixXld(0, this->Input_size); // Пустая матрица
		//Cell_states = MatrixXld::Zero(1, Hidden_size_);
		//Hidden_states = MatrixXld::Zero(1, Hidden_size_);
	}

	BiLSTM() = default;

	~BiLSTM() {
		this->Save("BiLSTM_state.txt");
	}

	void All_state_Сalculation() {
		this->Forward.All_state_Сalculation();
		this->Backward.All_state_Сalculation();

		this->Common_Hidden_states.clear();
		this->Common_Hidden_states.resize(this->Common_Input_states.size());

		for (size_t i = 0; i < this->Common_Input_states.size(); ++i) {
			const MatrixXld& Hf = this->Forward.Hidden_states[i];
			const MatrixXld& Hb = this->Backward.Hidden_states[i];
			size_t T = Hf.rows();

			MatrixXld concat(T, 2 * this->Common_Hidden_size);
			for (size_t t = 0; t < T; ++t) {
				concat.row(t) << Hf.row(t), Hb.row(T - 1 - t); // склеивание forward и backward
			}
			this->Common_Hidden_states[i] = concat;
		}
	}

	void SetInput_states(const std::vector<MatrixXld>& inputs) {
		this->Common_Input_states = inputs;
		this->Forward.SetInput_states(inputs);

		std::vector<MatrixXld> reversed_inputs(inputs.size());
		for (size_t i = 0; i < inputs.size(); ++i) {
			MatrixXld reversed = inputs[i].colwise().reverse().eval(); // обратный порядок временных шагов
			reversed_inputs[i] = reversed;
		}
		this->Backward.SetInput_states(reversed_inputs);
	}

	std::vector<RowVectorXld> GetFinalHidden_ForwardBackward() const {
		std::vector<RowVectorXld> out;
		for (const auto& H : this->Common_Hidden_states) {
			if (H.rows() > 0) {
				out.push_back(H.row(H.rows() - 1)); // последний шаг
			}
		}
		return out;
	}

	void Save(const std::string& filename) {
		auto addtofilename = [](const std::string& filename, const std::string& whatadd) {std::string ffilename;  for (const char a : filename) { if (a != '.') { ffilename += a; } else { ffilename += (whatadd + '.'); } } return ffilename; };
		this->Forward.save(addtofilename(filename, "_Forward"));
		this->Backward.save(addtofilename(filename, "_Backward"));
	}

	void Load(const std::string& filename) {
		this->Forward.load(filename + "_Forward");
		this->Backward.load(filename + "_Backward");
	}
protected:
	Eigen::Index Common_Input_size;
	Eigen::Index Common_Hidden_size;
	std::vector<MatrixXld> Common_Input_states;
	std::vector<MatrixXld> Common_Hidden_states;
private:
	SimpleLSTM Forward;
	SimpleLSTM Backward;
};