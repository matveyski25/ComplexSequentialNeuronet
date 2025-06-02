#include "HeaderLib_ComplexSequentialNeuronet.h"
 
using MatrixXld = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXld = Eigen::Matrix<long double, 1, Eigen::Dynamic>; // Вектор-строка
using VectorXld = Eigen::Matrix<long double, Eigen::Dynamic, 1>;    // Вектор-столбец

class SimpleLSTM {
	friend class BiLSTM;
public:
	
	SimpleLSTM(Eigen::Index Number_states = 1, Eigen::Index Hidden_size_ = 10){
		if (Hidden_size_ <= 0){
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
	//SimpleLSTM() = delete;
	~SimpleLSTM() {
		save("LSTM_state.txt");
	}

	void SetInput_states(const std::vector<MatrixXld>& Input_states_) {
		for(const auto & b : Input_states_){
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

	void SetDisplacements(const MatrixXld& displacements_FG, const MatrixXld& displacements_IG, const MatrixXld& displacements_CT, const MatrixXld& displacements_OG){
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
		auto xavier_init = [](size_t rows, size_t cols, long double c = 2.0L){
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

		Output_weights = ActivationFunctions::matrix_random(Hidden_size, 1, -0.1L, 0.1L);
	}

	void SetRandomDisplacements(long double a = -0.5L, long double b = 0.5L) {
		this->B_F = MatrixXld::Constant(1, Hidden_size, 1.0L);
		this->B_I = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->B_C = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->B_O = ActivationFunctions::matrix_random(1, Hidden_size, a, b);

		Output_bias = MatrixXld::Zero(1, 1);
	}

	void All_state_Сalculation() {
		// Подготовка весов вне цикла
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F_I, this->W_I_I, this->W_C_I, this->W_O_I;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->W_F_H, this->W_I_H, this->W_C_H, this->W_O_H;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		// Подготовка хранилищ
		this->Hidden_states.resize(this->Input_states.size());
		this->Cell_states.resize(this->Input_states.size());

		for (size_t nstep = 0; nstep < this->Input_states.size(); ++nstep) {
			size_t sequence_length = this->Input_states[nstep].rows();

			this->Hidden_states[nstep] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			this->Cell_states[nstep] = MatrixXld::Zero(sequence_length, this->Hidden_size);

			for (size_t timestep = 0; timestep < sequence_length; ++timestep) {
				RowVectorXld x_t = this->Input_states[nstep].row(timestep);
				RowVectorXld h_t_l = RowVectorXld::Zero(this->Hidden_size);
				RowVectorXld c_t_l = RowVectorXld::Zero(this->Hidden_size);

				if (timestep > 0) {
					h_t_l = this->Hidden_states[nstep].row(timestep - 1);
					c_t_l = this->Cell_states[nstep].row(timestep - 1);
				}

				RowVectorXld Z_t = x_t * W_x + h_t_l * W_h;
				Z_t += b;

				RowVectorXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(this->Hidden_size));
				RowVectorXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(this->Hidden_size, this->Hidden_size));
				RowVectorXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * this->Hidden_size, this->Hidden_size));
				RowVectorXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(this->Hidden_size));

				RowVectorXld new_c_t = f_t.array() * c_t_l.array() + i_t.array() * c_t_bar.array();
				RowVectorXld new_h_t = o_t.array() * ActivationFunctions::Tanh(new_c_t).array();

				this->Hidden_states[nstep].row(timestep) = new_h_t;
				this->Cell_states[nstep].row(timestep) = new_c_t;
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
	std::vector<MatrixXld> Hidden_states;
	std::vector<MatrixXld> Cell_states;

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

	MatrixXld Output_weights; // (Hidden_size x 1)
	MatrixXld Output_bias;    // (1 x 1)

	void n_state_Сalculation(size_t timestep, size_t nstep) {
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
			size_t T = timestep + 1;
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
	}

};

class SimpleLSTM_ForTrain : public SimpleLSTM{
	friend class BiLSTM_ForTrain;
public:
	SimpleLSTM_ForTrain(size_t Batch_size_ = 32, Eigen::Index Number_states = 1, Eigen::Index Hidden_size_ = 10) {
		if (Hidden_size_ <= 0) {
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}
		if (Batch_size_ <= 0) {
			throw std::invalid_argument("Размеры батчей должны быть больше 0");
		}

		this->Batch_size = Batch_size_;
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
	SimpleLSTM_ForTrain() = default;
	~SimpleLSTM_ForTrain() {
		save("LSTM_state_ForTrain.txt");
	}

	void SetInput_states(const std::vector<MatrixXld>& Input_states_) {
		for (const auto& b : Input_states_) {
			if (b.rows() == 0 || b.cols() != Input_size) {
				throw std::invalid_argument("Invalid input matrix dimensions");
			}
		}

		// Очищаем историю состояний
		FG_states.clear();
		IG_states.clear();
		CT_states.clear();
		OG_states.clear();

		Input_states = Input_states_;
		//size_t steps = Input_states.rows();

		// Корректная инициализация размеров
		//Cell_states = MatrixXld::Zero(steps + 1, Hidden_size);
		//Hidden_states = MatrixXld::Zero(steps + 1, Hidden_size);
	}

	void Train(const MatrixXld& inputs, const MatrixXld& targets, size_t epochs = 10000, long double learning_rate = 0.01, bool Enable_debugging_messages = false, bool decrease_learning_rate = false, int patience_ = 6000) {

	}

protected:
	size_t Batch_size;

	std::vector<MatrixXld> FG_states;
	std::vector<MatrixXld> IG_states;
	std::vector<MatrixXld> CT_states;
	std::vector<MatrixXld> OG_states;

	void Batch_n_state_Сalculation(size_t timestep, size_t Batchstep) {
		// Инициализация входов, скрытых и ячеечных состояний
		MatrixXld x_t(this->Batch_size, this->Input_size);
		MatrixXld h_t_l = MatrixXld::Zero(this->Batch_size, this->Hidden_size);
		MatrixXld c_t_l = MatrixXld::Zero(this->Batch_size, this->Hidden_size);

		// Заполнение входных данных на шаге t
		for (size_t i = 0; i < this->Batch_size; ++i) {
			size_t idx = Batchstep * this->Batch_size + i;
			if (idx < this->Input_states.size()) {
				x_t.row(i) = this->Input_states[idx].row(timestep);
			}
		}

		// Если timestep > 0, заполняем предыдущие состояния
		if (timestep > 0) {
			for (size_t i = 0; i < this->Batch_size; ++i) {
				size_t idx = Batchstep * this->Batch_size + i;
				if (idx < this->Hidden_states.size()) {
					h_t_l.row(i) = this->Hidden_states[idx].row(timestep - 1);
					c_t_l.row(i) = this->Cell_states[idx].row(timestep - 1);
				}
			}
		}

		// Объединённые веса и смещения
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F_I, this->W_I_I, this->W_C_I, this->W_O_I;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->W_F_H, this->W_I_H, this->W_C_H, this->W_O_H;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		// Расчет Z и разделение на ворота
		MatrixXld Z_t = x_t * W_x + h_t_l * W_h;
		Z_t.rowwise() += b;

		MatrixXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(this->Hidden_size));
		MatrixXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(this->Hidden_size, this->Hidden_size));
		MatrixXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * this->Hidden_size, this->Hidden_size));
		MatrixXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(this->Hidden_size));

		MatrixXld new_c_t = f_t.array() * c_t_l.array() + i_t.array() * c_t_bar.array();
		MatrixXld new_h_t = o_t.array() * ActivationFunctions::Tanh(new_c_t).array();

		// Убедимся, что состояние достаточно расширено
		size_t total_sequences = this->Input_states.size();
		this->Hidden_states.resize(total_sequences);
		this->Cell_states.resize(total_sequences);
		this->FG_states.resize(total_sequences);
		this->IG_states.resize(total_sequences);
		this->CT_states.resize(total_sequences);
		this->OG_states.resize(total_sequences);

		for (size_t i = 0; i < total_sequences; ++i) {
			size_t T = timestep + 1;
			if (this->Hidden_states[i].rows() < T) this->Hidden_states[i].conservativeResize(T, this->Hidden_size);
			if (this->Cell_states[i].rows() < T) this->Cell_states[i].conservativeResize(T, this->Hidden_size);
			if (this->FG_states[i].rows() < T) this->FG_states[i].conservativeResize(T, this->Hidden_size);
			if (this->IG_states[i].rows() < T) this->IG_states[i].conservativeResize(T, this->Hidden_size);
			if (this->CT_states[i].rows() < T) this->CT_states[i].conservativeResize(T, this->Hidden_size);
			if (this->OG_states[i].rows() < T) this->OG_states[i].conservativeResize(T, this->Hidden_size);
		}

		// Сохранение состояний
		for (size_t i = 0; i < this->Batch_size; ++i) {
			size_t idx = Batchstep * this->Batch_size + i;
			if (idx < total_sequences) {
				this->Hidden_states[idx].row(timestep) = new_h_t.row(i);
				this->Cell_states[idx].row(timestep) = new_c_t.row(i);
				this->FG_states[idx].row(timestep) = f_t.row(i);
				this->IG_states[idx].row(timestep) = i_t.row(i);
				this->CT_states[idx].row(timestep) = c_t_bar.row(i);
				this->OG_states[idx].row(timestep) = o_t.row(i);
			}
		}
	}

	void Batch_All_state_Сalculation() {
		size_t total_sequences = this->Input_states.size();
		if (total_sequences == 0) return;

		size_t sequence_length = this->Input_states[0].rows(); // правильная длина последовательности

		// Подготовка весов и смещений
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F_I, this->W_I_I, this->W_C_I, this->W_O_I;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->W_F_H, this->W_I_H, this->W_C_H, this->W_O_H;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		// Подготовка хранилищ состояний
		this->Hidden_states.resize(total_sequences);
		this->Cell_states.resize(total_sequences);
		this->FG_states.resize(total_sequences);
		this->IG_states.resize(total_sequences);
		this->CT_states.resize(total_sequences);
		this->OG_states.resize(total_sequences);

		for (size_t i = 0; i < total_sequences; ++i) {
			this->Hidden_states[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			this->Cell_states[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			this->FG_states[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			this->IG_states[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			this->CT_states[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			this->OG_states[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
		}

		size_t num_batches = (total_sequences + this->Batch_size - 1) / this->Batch_size;

		// Основной цикл: по батчам и шагам
		for (size_t Batchstep = 0; Batchstep < num_batches; ++Batchstep) {
			for (size_t timestep = 0; timestep < sequence_length; ++timestep) {
				MatrixXld x_t(this->Batch_size, this->Input_size);
				MatrixXld h_t_l = MatrixXld::Zero(this->Batch_size, this->Hidden_size);
				MatrixXld c_t_l = MatrixXld::Zero(this->Batch_size, this->Hidden_size);

				// Подготовка входов
				for (size_t i = 0; i < this->Batch_size; ++i) {
					size_t idx = Batchstep * this->Batch_size + i;
					if (idx < total_sequences) {
						x_t.row(i) = this->Input_states[idx].row(timestep);
						if (timestep > 0) {
							h_t_l.row(i) = this->Hidden_states[idx].row(timestep - 1);
							c_t_l.row(i) = this->Cell_states[idx].row(timestep - 1);
						}
					}
				}

				// Прямой проход через слой
				MatrixXld Z_t = x_t * W_x + h_t_l * W_h;
				Z_t.rowwise() += b;

				MatrixXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(this->Hidden_size));
				MatrixXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(this->Hidden_size, this->Hidden_size));
				MatrixXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * this->Hidden_size, this->Hidden_size));
				MatrixXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(this->Hidden_size));

				MatrixXld new_c_t = f_t.array() * c_t_l.array() + i_t.array() * c_t_bar.array();
				MatrixXld new_h_t = o_t.array() * ActivationFunctions::Tanh(new_c_t).array();

				// Запись состояний
				for (size_t i = 0; i < this->Batch_size; ++i) {
					size_t idx = Batchstep * this->Batch_size + i;
					if (idx < total_sequences) {
						this->Hidden_states[idx].row(timestep) = new_h_t.row(i);
						this->Cell_states[idx].row(timestep) = new_c_t.row(i);
						this->FG_states[idx].row(timestep) = f_t.row(i);
						this->IG_states[idx].row(timestep) = i_t.row(i);
						this->CT_states[idx].row(timestep) = c_t_bar.row(i);
						this->OG_states[idx].row(timestep) = o_t.row(i);
					}
				}
			}
		}
	}
};

class BiLSTM {
	
public:
	BiLSTM(Eigen::Index Number_states = 1, Eigen::Index Hidden_size_ = 10) {
		if (Hidden_size_ <= 0) {
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}
		this->Common_Input_size = Number_states;
		this->Common_Hidden_size = Hidden_size_;

		// Инициализация весов
		//SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

		// Инициализация смещений (1xHidden_size)
		//SetRandomDisplacements(-1.5L, 1.5L);

		// Инициализация состояний
		//Input_states = MatrixXld(0, this->Input_size); // Пустая матрица
		//Cell_states = MatrixXld::Zero(1, Hidden_size_);
		//Hidden_states = MatrixXld::Zero(1, Hidden_size_);
	}

	//BiLSTM() = default;

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
		this->Forward.save(filename);
		this->Backward.save(filename);
	}

	void Load(const std::string& filename) {
		this->Forward.load(filename);
		this->Backward.load(filename);
	}
protected:
	SimpleLSTM Forward;
	SimpleLSTM Backward;
	Eigen::Index Common_Input_size;
	Eigen::Index Common_Hidden_size;
	std::vector<MatrixXld> Common_Input_states;
	std::vector<MatrixXld> Common_Hidden_states;

};

class BiLSTM_ForTrain : public BiLSTM {
public:
	BiLSTM_ForTrain(size_t Batch_size_ = 32, Eigen::Index Number_states = 1, Eigen::Index Hidden_size_ = 10) {
		if (Hidden_size_ <= 0) {
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}
		if (Batch_size_ <= 0) {
			throw std::invalid_argument("Размеры батчей должны быть больше 0");
		}

		this->Common_Batch_size = Batch_size_;
		this->Common_Input_size = Number_states;
		this->Common_Hidden_size = Hidden_size_;

		// Инициализация весов
		//SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

		// Инициализация смещений (1xHidden_size)
		//SetRandomDisplacements(-1.5L, 1.5L);

		// Инициализация состояний
		//Input_states = MatrixXld(0, this->Input_size); // Пустая матрица
		//Cell_states = MatrixXld::Zero(1, Hidden_size_);
		//Hidden_states = MatrixXld::Zero(1, Hidden_size_);
	}

	//BiLSTM_ForTrain() = default;

	~BiLSTM_ForTrain() {
		this->Save("BiLSTM_state_ForTrain.txt");
	}

	void Batch_All_state_Сalculation() {
		//this->Forward.Batch_All_state_Сalculation();
		//this->Backward.Batch_All_state_Сalculation();
	}


	void Train() {

	}

protected:
	size_t Common_Batch_size;
};

class Attention {
public:
	virtual ~Attention() = default;

	// Абстрактный метод: вычисляет контекст по шагу
	virtual RowVectorXld ComputeContext(const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden) = 0;

	// Очистка накопленных значений
	virtual void ClearCache() {
		all_attention_weights_.clear();
		all_scores_.clear();
	}

	// Получение attention-весов по всем временным шагам
	const std::vector<VectorXld>& GetAllAttentionWeights() const { return all_attention_weights_; }

	// Получение сырых score-векторов (до softmax)
	const std::vector<VectorXld>& GetAllScores() const { return all_scores_; }

protected:
	// Вспомогательные буферы для накопления истории attention по всем шагам
	std::vector<VectorXld> all_attention_weights_;  // α_t для всех t
	std::vector<VectorXld> all_scores_;             // e_{t,i} для всех t
};

class BahdanauAttention : public Attention {
public:
	BahdanauAttention(Eigen::Index encoder_hidden_size, Eigen::Index decoder_hidden_size, Eigen::Index attention_size)
		: encoder_hidden_size_(encoder_hidden_size),
		decoder_hidden_size_(decoder_hidden_size),
		attention_size_(attention_size)
	{
		// Инициализация весов (Xavier)
		W_encoder_ = ActivationFunctions::matrix_random(attention_size_, encoder_hidden_size_, -1.0L, 1.0L); // [A x 2H]
		W_decoder_ = ActivationFunctions::matrix_random(attention_size_, decoder_hidden_size_, -1.0L, 1.0L); // [A x H_dec]
		attention_vector_ = ActivationFunctions::matrix_random(attention_size_, 1, -1.0L, 1.0L);              // [A x 1]
	}

	// Вычисляет контекстный вектор и сохраняет внутренние веса
	RowVectorXld ComputeContext(const MatrixXld& encoder_outputs,
		const RowVectorXld& decoder_prev_hidden) override
	{
		const size_t time_steps_enc = encoder_outputs.rows();   // Tx
		const size_t hidden_size_enc = encoder_outputs.cols();  // 2H

		VectorXld scores(time_steps_enc);  // e_{t,i}

		for (size_t i = 0; i < time_steps_enc; ++i) {
			RowVectorXld h_i = encoder_outputs.row(i);  // [1 x 2H]

			RowVectorXld combined_input =
				(W_encoder_ * h_i.transpose() + W_decoder_ * decoder_prev_hidden.transpose()).transpose(); // [1 x A]

			scores(i) = (combined_input.array().tanh().matrix() * attention_vector_).value();  // scalar
		}

		// Softmax для e → α
		VectorXld attention_weights = scores.array().exp();
		long double sum_exp = attention_weights.sum() + 1e-8L;
		attention_weights /= sum_exp;

		// Сохраняем историю на каждом шаге
		this->all_attention_weights_.push_back(attention_weights);
		this->all_scores_.push_back(scores);

		// Контекст: sum_i α_i * h_i
		RowVectorXld context = RowVectorXld::Zero(hidden_size_enc);
		for (size_t i = 0; i < time_steps_enc; ++i) {
			context += attention_weights(i) * encoder_outputs.row(i);
		}

		return context;
	}

private:
	Eigen::Index encoder_hidden_size_;    // 2H
	Eigen::Index decoder_hidden_size_;    // H_dec
	Eigen::Index attention_size_;         // A

	MatrixXld W_encoder_;       // [A x 2H]
	MatrixXld W_decoder_;       // [A x H_dec]
	MatrixXld attention_vector_; // [A x 1]
};

// ==== БАЗОВЫЙ ЭНКОДЕР ====
class EncoderForSeq2Seq : public BiLSTM {
public:
	EncoderForSeq2Seq(Eigen::Index input_size, Eigen::Index hidden_size)
		: BiLSTM(input_size, hidden_size) {
	}

	void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
		SetInput_states(input_sequence_batch);
		All_state_Сalculation();
	}

	const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
		return this->Common_Hidden_states;
	}
	virtual ~EncoderForSeq2Seq() = default;
};


// ==== ТРЕНИРОВОЧНЫЙ ЭНКОДЕР ====
class EncoderForSeq2Seq_ForTrain : public BiLSTM_ForTrain {
public:
	EncoderForSeq2Seq_ForTrain(size_t batch_size, Eigen::Index input_size, Eigen::Index hidden_size)
		: BiLSTM_ForTrain(batch_size, input_size, hidden_size) {
	}

	void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
		this->SetInput_states(input_sequence_batch);
		this->Batch_All_state_Сalculation();
	}

	const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
		return this->Common_Hidden_states;
	}

};


class DecoderForSeq2SeqWithAttention : public SimpleLSTM {
public:
	DecoderForSeq2SeqWithAttention(std::shared_ptr<Attention> attention_module)
		: attention_(std::move(attention_module)) {
	}

	virtual ~DecoderForSeq2SeqWithAttention() = default;

	void SetEncoderOutputs(const std::vector<MatrixXld>& encoder_outputs) {
		this->encoder_outputs_ = encoder_outputs;
	}

	void All_state_Calculation() {
		if (Input_states.empty() || encoder_outputs_.empty()) return;

		Output_state.clear();
		context_vectors.clear();
		attention_->ClearCache();

		size_t batch_size = Input_states.size();
		size_t T_dec = Input_states[0].rows();
		size_t T_enc = encoder_outputs_[0].rows();
		size_t encoder_hidden_dim = encoder_outputs_[0].cols();

		// Инициализация выходов и контекстов
		Output_state.resize(batch_size);
		context_vectors.resize(batch_size);

		Hidden_states.resize(batch_size);
		Cell_states.resize(batch_size);
		for (size_t i = 0; i < batch_size; ++i) {
			Hidden_states[i] = MatrixXld::Zero(T_dec, Hidden_size);
			Cell_states[i] = MatrixXld::Zero(T_dec, Hidden_size);
			Output_state[i] = MatrixXld::Zero(T_dec, Hidden_size);
			context_vectors[i] = MatrixXld::Zero(T_dec, encoder_hidden_dim);
		}

		for (size_t n = 0; n < batch_size; ++n) {
			RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
			RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

			for (size_t t = 0; t < T_dec; ++t) {
				RowVectorXld y_prev = Input_states[n].row(t); // y_{t-1}
				const MatrixXld& H_enc = encoder_outputs_[n]; // [T_enc x 2H]

				RowVectorXld context = attention_->ComputeContext(H_enc, h_prev); // [1 x 2H]

				// Сохраняем
				context_vectors[n].row(t) = context;

				// Конкатенируем y_prev и context
				RowVectorXld decoder_input(y_prev.cols() + context.cols());
				decoder_input << y_prev, context;

				// Расчёт веса
				MatrixXld W_x(Input_size, 4 * Hidden_size);
				W_x << W_F_I, W_I_I, W_C_I, W_O_I;

				MatrixXld W_h(Hidden_size, 4 * Hidden_size);
				W_h << W_F_H, W_I_H, W_C_H, W_O_H;

				RowVectorXld b(4 * Hidden_size);
				b << B_F, B_I, B_C, B_O;

				// Подаем decoder_input вместо y_t напрямую
				RowVectorXld Z_t = decoder_input * W_x + h_prev * W_h;
				Z_t += b;

				RowVectorXld f_t = ActivationFunctions::Sigmoid(Z_t.leftCols(Hidden_size));
				RowVectorXld i_t = ActivationFunctions::Sigmoid(Z_t.middleCols(Hidden_size, Hidden_size));
				RowVectorXld c_t_bar = ActivationFunctions::Tanh(Z_t.middleCols(2 * Hidden_size, Hidden_size));
				RowVectorXld o_t = ActivationFunctions::Sigmoid(Z_t.rightCols(Hidden_size));

				RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_t_bar.array();
				RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();

				Hidden_states[n].row(t) = h_t;
				Cell_states[n].row(t) = c_t;
				Output_state[n].row(t) = h_t;

				// обновляем h_prev и c_prev
				h_prev = h_t;
				c_prev = c_t;
			}
		}
	}

	const std::vector<MatrixXld>& GetContextVectors() const { return context_vectors; }
	const std::vector<MatrixXld>& GetOutputStates() const { return Output_state; }

private:
	std::shared_ptr<Attention> attention_;
	std::vector<MatrixXld> encoder_outputs_; // [B][T_enc x 2H]
	std::vector<MatrixXld> context_vectors;  // [B][T_dec x 2H]
	std::vector<MatrixXld> Output_state;     // [B][T_dec x Hidden_size]
};

// ==== ТРЕНИРОВОЧНЫЙ ДЕКОДЕР ====
class DecoderForSeq2SeqWithAttention_ForTrain : public DecoderForSeq2SeqWithAttention {
public:
	DecoderForSeq2SeqWithAttention_ForTrain(std::shared_ptr<Attention> attention_module)
		: DecoderForSeq2SeqWithAttention(attention_module) {
	}

	void SetTargets(const std::vector<MatrixXld>& targets) {
		target_outputs_ = targets;
	}

	const std::vector<MatrixXld>& GetTargets() const {
		return target_outputs_;
	}

private:
	std::vector<MatrixXld> target_outputs_;  // [B][T_dec x Output_dim]
	MatrixXld W_output; // [output_size x hidden_size + context_size]
	RowVectorXld b_output; // [1 x output_size]
	std::vector<MatrixXld> logits; // [B][T x output_size]
};


// ==== БАЗОВАЯ Seq2Seq + Attention ====
class Seq2SeqWithAttention {
public:
	template<typename EncoderT, typename DecoderT>
	Seq2SeqWithAttention(
		std::unique_ptr<EncoderT> encoder,
		std::unique_ptr<DecoderT> decoder)
		: encoder_(std::move(encoder)), decoder_(std::move(decoder)) {
	}

	void Inference(const std::vector<MatrixXld>& input_sequence_batch,
		const std::vector<MatrixXld>& decoder_inputs)
	{
		encoder_->Encode(input_sequence_batch);
		decoder_->SetEncoderOutputs(encoder_->GetEncodedHiddenStates());
		decoder_->SetInput_states(decoder_inputs);
		decoder_->All_state_Calculation();
	}

	const std::vector<MatrixXld>& GetDecoderOutputs() const {
		return decoder_->GetOutputStates();
	}

protected:
	std::unique_ptr<EncoderForSeq2Seq> encoder_;
	std::unique_ptr<DecoderForSeq2SeqWithAttention> decoder_;
};


// ==== ТРЕНИРОВОЧНАЯ Seq2Seq + Attention ====
class Seq2SeqWithAttention_ForTrain : public Seq2SeqWithAttention {
public:
	Seq2SeqWithAttention_ForTrain( std::unique_ptr<EncoderForSeq2Seq_ForTrain> encoder_train, std::unique_ptr<DecoderForSeq2SeqWithAttention_ForTrain> decoder_train)
		: Seq2SeqWithAttention(std::move(encoder_train), std::move(decoder_train)) {

	}

	void TrainStep(const std::vector<MatrixXld>& input_batch,
		const std::vector<MatrixXld>& decoder_inputs,
		const std::vector<MatrixXld>& decoder_targets)
	{
		auto* encoder_train = dynamic_cast<EncoderForSeq2Seq_ForTrain*>(encoder_.get());
		auto* decoder_train = dynamic_cast<DecoderForSeq2SeqWithAttention_ForTrain*>(decoder_.get());

		encoder_train->Encode(input_batch);
		decoder_train->SetEncoderOutputs(encoder_train->GetEncodedHiddenStates());
		decoder_train->SetInput_states(decoder_inputs);
		decoder_train->SetTargets(decoder_targets);
		decoder_train->All_state_Calculation();

		// TODO: реализовать backpropagation
	}
};


int main() {
	setlocale(LC_ALL, "Russian");

	return 0;
}