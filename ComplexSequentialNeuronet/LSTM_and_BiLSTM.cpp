#include "HeaderLSTM_and_BiLSTM.h"

SimpleLSTM::SimpleLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_) {
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

SimpleLSTM::~SimpleLSTM() {
	save("LSTM_state.txt");
}

void SimpleLSTM::SetInput_states(const std::vector<MatrixXld>& Input_states_) {
	for (const auto& b : Input_states_) {
		if (b.rows() == 0 || b.cols() != Input_size) {
			throw std::invalid_argument("Invalid input matrix dimensions");
		}
	}

	this->Input_states = Input_states_;
	//size_t steps = Input_states.rows();

	// Корректная инициализация размеров
	//Cell_states = MatrixXld::Zero(steps + 1, Hidden_size);
	//Hidden_states = MatrixXld::Zero(steps + 1, Hidden_size);
}

void SimpleLSTM::SetWeights(const MatrixXld& weights_I_F, const MatrixXld& weights_I_I, const MatrixXld& weights_I_C, const MatrixXld& weights_I_O, const MatrixXld& weights_H_F, const MatrixXld& weights_H_I, const MatrixXld& weights_H_C, const MatrixXld& weights_H_O)
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

	this->W_F = weights_I_F;
	this->W_I = weights_I_I;
	this->W_C = weights_I_C;
	this->W_O = weights_I_O;

	this->U_F = weights_H_F;
	this->U_I = weights_H_I;
	this->U_C = weights_H_C;
	this->U_O = weights_H_O;
}

void SimpleLSTM::SetDisplacements(const MatrixXld& displacements_FG, const MatrixXld& displacements_IG, const MatrixXld& displacements_CT, const MatrixXld& displacements_OG) {
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

void SimpleLSTM::SetRandomWeights(long double a, long double b) {
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
	this->U_F = init_h(this->Hidden_size, this->Hidden_size);
	this->U_I = init_h(this->Hidden_size, this->Hidden_size);
	this->U_C = init_h(this->Hidden_size, this->Hidden_size);
	this->U_O = init_h(this->Hidden_size, this->Hidden_size);

	this->W_F = init_i(this->Input_size, this->Hidden_size);
	this->W_I = init_i(this->Input_size, this->Hidden_size);
	this->W_C = init_i(this->Input_size, this->Hidden_size);
	this->W_O = init_i(this->Input_size, this->Hidden_size);

	//Output_weights = ActivationFunctions::matrix_random(Hidden_size, 1, -0.1L, 0.1L);
}

void SimpleLSTM::SetRandomDisplacements(long double a, long double b) {
	this->B_F = MatrixXld::Constant(1, Hidden_size, 1.0L);
	this->B_I = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
	this->B_C = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
	this->B_O = ActivationFunctions::matrix_random(1, Hidden_size, a, b);

	//Output_bias = MatrixXld::Zero(1, 1);
}

void SimpleLSTM::All_state_Сalculation() {
	// Подготовка весов вне цикла
	MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
	W_x << this->W_F, this->W_I, this->W_C, this->W_O;

	MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
	W_h << this->U_F, this->U_I, this->U_C, this->U_O;

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

std::vector<RowVectorXld> SimpleLSTM::GetLastOutputs() const {
	std::vector<RowVectorXld> outputs;
	for (const auto& state : this->Hidden_states) {
		if (state.rows() > 0) {
			outputs.push_back(state.row(state.rows() - 1));
		}
	}
	return outputs;
}

void SimpleLSTM::save(const std::string& filename) const {
	std::ofstream file(filename, std::ios::trunc); // Используйте trunc для перезаписи
	if (!file) throw std::runtime_error("Cannot open file for writing");

	// Сохраняем только актуальные параметры
	file << this->Input_size << "\n" << this->Hidden_size << "\n";

	// Сохраняем веса и смещения
	save_matrix(file, this->U_F);
	save_matrix(file, this->U_I);
	save_matrix(file, this->U_C);
	save_matrix(file, this->U_O);

	save_matrix(file, this->W_F);
	save_matrix(file, this->W_I);
	save_matrix(file, this->W_C);
	save_matrix(file, this->W_O);

	save_matrix(file, this->B_F);
	save_matrix(file, this->B_I);
	save_matrix(file, this->B_C);
	save_matrix(file, this->B_O);
}

void SimpleLSTM::load(const std::string& filename) {
	std::ifstream file(filename);
	if (!file) throw std::runtime_error("Cannot open file for reading");

	file >> this->Input_size >> this->Hidden_size;

	load_matrix(file, this->U_F);
	load_matrix(file, this->U_I);
	load_matrix(file, this->U_C);
	load_matrix(file, this->U_O);

	load_matrix(file, this->W_F);
	load_matrix(file, this->W_I);
	load_matrix(file, this->W_C);
	load_matrix(file, this->W_O);

	load_matrix(file, this->B_F);
	load_matrix(file, this->B_I);
	load_matrix(file, this->B_C);
	load_matrix(file, this->B_O);
}

void SimpleLSTM::save_matrix(std::ofstream& file, const MatrixXld& m) const {
	file << m.rows() << " " << m.cols() << "\n";
	for (Eigen::Index i = 0; i < m.rows(); ++i) {
		for (Eigen::Index j = 0; j < m.cols(); ++j) {
			file << m(i, j) << " ";
		}
		file << "\n";
	}
}

void SimpleLSTM::load_matrix(std::ifstream& file, MatrixXld& m) {
	Eigen::Index rows, cols;
	file >> rows >> cols;
	m = MatrixXld(rows, cols);
	for (Eigen::Index i = 0; i < rows; ++i) {
		for (Eigen::Index j = 0; j < cols; ++j) {
			file >> m(i, j);
		}
	}
}



BiLSTM::BiLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_) {
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

BiLSTM::~BiLSTM() {
	this->Save("BiLSTM_state.txt");
}

void BiLSTM::All_state_Сalculation() {
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

void BiLSTM::SetInput_states(const std::vector<MatrixXld>& inputs) {
	this->Common_Input_states = inputs;
	this->Forward.SetInput_states(inputs);

	std::vector<MatrixXld> reversed_inputs(inputs.size());
	for (size_t i = 0; i < inputs.size(); ++i) {
		MatrixXld reversed = inputs[i].colwise().reverse().eval(); // обратный порядок временных шагов
		reversed_inputs[i] = reversed;
	}
	this->Backward.SetInput_states(reversed_inputs);
}

std::vector<RowVectorXld> BiLSTM::GetFinalHidden_ForwardBackward() const {
	std::vector<RowVectorXld> out;
	for (const auto& H : this->Common_Hidden_states) {
		if (H.rows() > 0) {
			out.push_back(H.row(H.rows() - 1)); // последний шаг
		}
	}
	return out;
}

void BiLSTM::Save(const std::string& filename) {
	auto addtofilename = [](const std::string& filename, const std::string& whatadd) {std::string ffilename;  for (const char a : filename) { if (a != '.') { ffilename += a; } else { ffilename += (whatadd + '.'); } } return ffilename; };
	this->Forward.save(addtofilename(filename, "_Forward"));
	this->Backward.save(addtofilename(filename, "_Backward"));
}

void BiLSTM::Load(const std::string& filename) {
	this->Forward.load(filename + "_Forward");
	this->Backward.load(filename + "_Backward");
}



SimpleLSTM_ForTrain::SimpleLSTM_ForTrain(size_t Batch_size_, Eigen::Index Number_states, Eigen::Index Hidden_size_) {
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

SimpleLSTM_ForTrain::~SimpleLSTM_ForTrain() {
	save("LSTM_state_ForTrain.txt");
}

void SimpleLSTM_ForTrain::SetInput_states(const std::vector<MatrixXld>& Input_states_) {
	for (const auto& b : Input_states_) {
		if (b.rows() == 0 || b.cols() != Input_size) {
			throw std::invalid_argument("Invalid input matrix dimensions");
		}
	}

	// Очищаем историю состояний
	this->statesForgrads.f.clear();
	this->statesForgrads.i.clear();
	this->statesForgrads.ccond.clear();
	this->statesForgrads.o.clear();
	this->statesForgrads.c.clear();
	this->statesForgrads.h.clear();

	Input_states = Input_states_;
	//size_t steps = Input_states.rows();

	// Корректная инициализация размеров
	//Cell_states = MatrixXld::Zero(steps + 1, Hidden_size);
	//Hidden_states = MatrixXld::Zero(steps + 1, Hidden_size);
}

void SimpleLSTM_ForTrain::save(const std::string& filename) const {
	std::ofstream file(filename, std::ios::trunc); // Используйте trunc для перезаписи
	if (!file) throw std::runtime_error("Cannot open file for writing");

	// Сохраняем только актуальные параметры
	file << this->Input_size << "\n" << this->Hidden_size << "\n" << this->Batch_size << "\n";

	// Сохраняем веса и смещения
	save_matrix(file, this->U_F);
	save_matrix(file, this->U_I);
	save_matrix(file, this->U_C);
	save_matrix(file, this->U_O);

	save_matrix(file, this->W_F);
	save_matrix(file, this->W_I);
	save_matrix(file, this->W_C);
	save_matrix(file, this->W_O);

	save_matrix(file, this->B_F);
	save_matrix(file, this->B_I);
	save_matrix(file, this->B_C);
	save_matrix(file, this->B_O);
}

void SimpleLSTM_ForTrain::load(const std::string& filename) {
	std::ifstream file(filename);
	if (!file) throw std::runtime_error("Cannot open file for reading");

	file >> this->Input_size >> this->Hidden_size >> this->Batch_size;

	load_matrix(file, this->U_F);
	load_matrix(file, this->U_I);
	load_matrix(file, this->U_C);
	load_matrix(file, this->U_O);

	load_matrix(file, this->W_F);
	load_matrix(file, this->W_I);
	load_matrix(file, this->W_C);
	load_matrix(file, this->W_O);

	load_matrix(file, this->B_F);
	load_matrix(file, this->B_I);
	load_matrix(file, this->B_C);
	load_matrix(file, this->B_O);
}

void SimpleLSTM_ForTrain::Batch_All_state_Сalculation() {
		size_t total_sequences = this->Input_states.size();
		if (total_sequences == 0) return;

		size_t sequence_length = this->Input_states[0].rows(); // правильная длина последовательности

		// Подготовка весов и смещений
		MatrixXld W_x(this->Input_size, 4 * this->Hidden_size);
		W_x << this->W_F, this->W_I, this->W_C, this->W_O;

		MatrixXld W_h(this->Hidden_size, 4 * this->Hidden_size);
		W_h << this->U_F, this->U_I, this->U_C, this->U_O;

		RowVectorXld b(4 * this->Hidden_size);
		b << this->B_F, this->B_I, this->B_C, this->B_O;

		// Подготовка хранилищ состояний
		statesForgrads.f.resize(total_sequences);
		statesForgrads.i.resize(total_sequences);
		statesForgrads.ccond.resize(total_sequences);
		statesForgrads.o.resize(total_sequences);
		statesForgrads.c.resize(total_sequences);
		statesForgrads.h.resize(total_sequences);

		for (size_t i = 0; i < total_sequences; ++i) {
			statesForgrads.f[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			statesForgrads.i[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			statesForgrads.ccond[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			statesForgrads.o[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			statesForgrads.c[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
			statesForgrads.h[i] = MatrixXld::Zero(sequence_length, this->Hidden_size);
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
							c_t_l.row(i) = this->statesForgrads.c[idx].row(timestep - 1);
							h_t_l.row(i) = this->statesForgrads.h[idx].row(timestep - 1);
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
						statesForgrads.f[idx].row(timestep) = f_t.row(i);
						statesForgrads.i[idx].row(timestep) = i_t.row(i);
						statesForgrads.ccond[idx].row(timestep) = c_t_bar.row(i);
						statesForgrads.o[idx].row(timestep) = o_t.row(i);
						statesForgrads.c[idx].row(timestep) = new_c_t.row(i);
						statesForgrads.h[idx].row(timestep) = new_h_t.row(i);

					}
				}
			}
		}
	}


void BiLSTM_ForTrain::SetInput_states(const std::vector<MatrixXld>& inputs) {
	this->Common_Input_states = inputs;
	this->Forward.SetInput_states(inputs);

	std::vector<MatrixXld> reversed_inputs(inputs.size());
	for (size_t i = 0; i < inputs.size(); ++i) {
		MatrixXld reversed = inputs[i].colwise().reverse().eval(); // обратный порядок временных шагов
		reversed_inputs[i] = reversed;
	}
	this->Backward.SetInput_states(reversed_inputs);
}

std::vector<RowVectorXld> BiLSTM_ForTrain::GetFinalHidden_ForwardBackward() const {
	std::vector<RowVectorXld> out;
	for (const auto& H : this->Common_Hidden_states) {
		if (H.rows() > 0) {
			out.push_back(H.row(H.rows() - 1)); // последний шаг
		}
	}
	return out;
}

void BiLSTM_ForTrain::Save(const std::string& filename) {
	auto addtofilename = [](const std::string& filename, const std::string& whatadd) {std::string ffilename;  for (const char a : filename) { if (a != '.') { ffilename += a; } else { ffilename += (whatadd + '.'); } } return ffilename; };
	this->Forward.save(addtofilename(filename, "_Forward"));
	this->Backward.save(addtofilename(filename, "_Backward"));
}

void BiLSTM_ForTrain::Load(const std::string& filename) {
	this->Forward.load(filename + "_Forward");
	this->Backward.load(filename + "_Backward");
}

BiLSTM_ForTrain::BiLSTM_ForTrain(size_t Batch_size_, Eigen::Index Number_states, Eigen::Index Hidden_size_)
	: BiLSTM(Number_states, Hidden_size_),
	Forward(Batch_size_, Number_states, Hidden_size_),
	Backward(Batch_size_, Number_states, Hidden_size_),
	Common_Batch_size(Batch_size_) {

	if (Hidden_size_ <= 0) {
		throw std::invalid_argument("Размеры слоев должны быть больше 0");
	}
	if (Batch_size_ <= 0) {
		throw std::invalid_argument("Размеры батчей должны быть больше 0");
	}

	// Инициализация весов
	//SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

	// Инициализация смещений (1xHidden_size)
	//SetRandomDisplacements(-1.5L, 1.5L);

	// Инициализация состояний
	//Input_states = MatrixXld(0, this->Input_size); // Пустая матрица
	//Cell_states = MatrixXld::Zero(1, Hidden_size_);
	//Hidden_states = MatrixXld::Zero(1, Hidden_size_);
}

BiLSTM_ForTrain::~BiLSTM_ForTrain() {
	this->Save("BiLSTM_state_ForTrain.txt");
}

void BiLSTM_ForTrain::Batch_All_state_Сalculation() {
	this->Forward.Batch_All_state_Сalculation();
	this->Backward.Batch_All_state_Сalculation();
	Common_Hidden_states.clear();
	Common_Hidden_states.resize(Forward.statesForgrads.h.size());
	for (size_t i = 0; i < Forward.statesForgrads.h.size(); ++i) {
		const auto& Hf = Forward.statesForgrads.h[i];  // [T × H]
		const auto& Hb = Backward.statesForgrads.h[i]; // [T × H], но обратный порядок
		size_t T = Hf.rows();
		MatrixXld concat(T, 2 * this->Common_Hidden_size);
		for (size_t t = 0; t < T; ++t) {
			concat.row(t) << Hf.row(t), Hb.row(T - 1 - t);
		}
		Common_Hidden_states[i] = concat;
	}
}



// Очистка накопленных значений
void Attention::ClearCache() {
	all_attention_weights_.clear();
	all_scores_.clear();
	all_tanh_outputs_.clear();
}

// Получение attention-весов по всем временным шагам
const std::vector<VectorXld>& Attention::GetAllAttentionWeights() const { return all_attention_weights_; }

// Получение сырых score-векторов (до softmax)
const std::vector<VectorXld>& Attention::GetAllScores() const { return all_scores_; }



BahdanauAttention::BahdanauAttention(Eigen::Index encoder_hidden_size, Eigen::Index decoder_hidden_size, Eigen::Index attention_size)
	: encoder_hidden_size_(encoder_hidden_size),
	decoder_hidden_size_(decoder_hidden_size),
	attention_size_(attention_size)
{
	// Инициализация весов (Xavier)
	SetRandomWeights(-1, 1);
}

// Вычисляет контекстный вектор и сохраняет внутренние веса
RowVectorXld BahdanauAttention::ComputeContext(const MatrixXld& encoder_outputs,
	const RowVectorXld& decoder_prev_hidden) {
	const size_t time_steps_enc = encoder_outputs.rows();   // длина входной последовательности
	const size_t hidden_size_enc = encoder_outputs.cols();  // размерность h_i (обычно 2H)
	const size_t A = this->attention_size_;                 // размер attention-пространства

	VectorXld scores(time_steps_enc);         // e_{ti}
	std::vector<RowVectorXld> u_t;            // вектор для хранения u_{ti} на текущем шаге t

	for (size_t i = 0; i < time_steps_enc; ++i) {
		RowVectorXld h_i = encoder_outputs.row(i); // [1 x 2H]

		// [A x 1] = W_encoder * h_i^T + W_decoder * s_{t-1}^T
		RowVectorXld combined_input =
			(W_encoder_ * h_i.transpose() + W_decoder_ * decoder_prev_hidden.transpose()).transpose(); // [1 x A]

		RowVectorXld u_ti = combined_input.array().tanh().matrix();  // [1 x A]

		scores(i) = (u_ti * attention_vector_).value();  // e_{ti} = v^T u_{ti}

		u_t.push_back(u_ti);  // сохраняем u_{ti} для текущего i
	}

	// Softmax по e_{ti} → α_{ti}
	VectorXld attention_weights = ActivationFunctions::Softmax(scores);

	// Сохраняем веса и логиты
	this->all_attention_weights_.push_back(attention_weights);
	this->all_scores_.push_back(scores);
	this->all_tanh_outputs_.push_back(u_t);  // сохраняем все u_{ti} для текущего t

	// Вычисляем контекст: c_t = ∑ α_{ti} * h_i
	RowVectorXld context = RowVectorXld::Zero(hidden_size_enc);
	for (size_t i = 0; i < time_steps_enc; ++i) {
		context += attention_weights(i) * encoder_outputs.row(i);
	}

	return context;
}

void BahdanauAttention::SetRandomWeights(long double a, long double b) {
	this->W_encoder_ = ActivationFunctions::matrix_random(this->attention_size_, this->encoder_hidden_size_, a, b);
	this->W_decoder_ = ActivationFunctions::matrix_random(this->attention_size_, this->decoder_hidden_size_, a, b);
	this->attention_vector_ = ActivationFunctions::matrix_random(this->attention_size_, 1, a, b);
}