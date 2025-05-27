#include "HeaderLib_ComplexSequentialNeuronet.h"
 
using MatrixXld = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXld = Eigen::Matrix<long double, 1, Eigen::Dynamic>; // Вектор-строка
using VectorXld = Eigen::Matrix<long double, Eigen::Dynamic, 1>;    // Вектор-столбец

class SimpleLSTM {
public:
	
	SimpleLSTM(Eigen::Index Batch, size_t Number_states = 1, size_t Hidden_size_ = 10){
		if (Hidden_size_ == 0){
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}

		this->Input_size = Number_states;
		this->Hidden_size = Hidden_size_;
		// Инициализация весов
		SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

		// Инициализация смещений (1xHidden_size)
		SetRandomDisplacements(-1.5L, 1.5L);

		// Инициализация состояний
		Input_states = MatrixXld(0, this->Input_size); // Пустая матрица
		Cell_states = MatrixXld::Zero(1, Hidden_size_);
		Hidden_states = MatrixXld::Zero(1, Hidden_size_);
	}

	SimpleLSTM() = default;

	~SimpleLSTM() {
		save("LSTM_state.txt");
	}

	void SetInput_states(const MatrixXld& Input_states_) {
		if (Input_states_.rows() == 0 || Input_states_.cols() != Input_size) {
			throw std::invalid_argument("Invalid input matrix dimensions");
		}

		// Очищаем историю состояний
		FG_states.clear();
		IG_states.clear();
		CT_states.clear();
		OG_states.clear();

		Input_states = Input_states_;
		size_t steps = Input_states.rows();

		// Корректная инициализация размеров
		Cell_states = MatrixXld::Zero(steps + 1, Hidden_size);
		Hidden_states = MatrixXld::Zero(steps + 1, Hidden_size);
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

		Weights_for_FG_IS = weights_I_F;
		Weights_for_IG_IS = weights_I_I;
		Weights_for_CT_IS = weights_I_C;
		Weights_for_OG_IS = weights_I_O;

		Weights_for_FG_HS = weights_H_F;
		Weights_for_IG_HS = weights_H_I;
		Weights_for_CT_HS = weights_H_C;
		Weights_for_OG_HS = weights_H_O;
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

		Displacements_for_FG = displacements_FG;
		Displacements_for_IG = displacements_IG;
		Displacements_for_CT = displacements_CT;
		Displacements_for_OG = displacements_OG;
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
		this->Weights_for_FG_HS = init_h(Hidden_size, Hidden_size);
		this->Weights_for_IG_HS = init_h(Hidden_size, Hidden_size);
		this->Weights_for_CT_HS = init_h(Hidden_size, Hidden_size);
		this->Weights_for_OG_HS = init_h(Hidden_size, Hidden_size);
 								 
		this->Weights_for_FG_IS = init_i(Hidden_size, Input_size);
		this->Weights_for_IG_IS = init_i(Hidden_size, Input_size);
		this->Weights_for_CT_IS = init_i(Hidden_size, Input_size);
		this->Weights_for_OG_IS = init_i(Hidden_size, Input_size);

		Output_weights = ActivationFunctions::matrix_random(Hidden_size, 1, -0.1L, 0.1L);
	}

	void SetRandomDisplacements(long double a = -0.5L, long double b = 0.5L) {
		this->Displacements_for_FG = MatrixXld::Constant(1, Hidden_size, 1.0L);
		this->Displacements_for_IG = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->Displacements_for_CT = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->Displacements_for_OG = ActivationFunctions::matrix_random(1, Hidden_size, a, b);

		Output_bias = MatrixXld::Zero(1, 1);
	}

	bool CalculationAll_states(const MatrixXld& targets, size_t max_steps, long double precision = 0.1) {
		for (size_t t = 0; t < max_steps; ++t) {
			n_state_Сalculation(t);
			MatrixXld predictions = GetOutput_states();
			MatrixXld error = predictions - targets;

			// Проверка ошибки
			bool all_match = true;
			for (Eigen::Index i = 0; i < error.rows(); ++i) {
				for (Eigen::Index j = 0; j < error.cols(); ++j) {
					auto err = std::abs(error(i, j));
					//std::cout << targets(1, 0) << std::endl;
					//std::cout << predictions(1, 0) << std::endl ;
					if (err > precision) {
						all_match = false;
						break;
					}
				}
				if (!all_match) break;
			}
			if (all_match) return true; // Остановка при достижении точности
		}
		return false;
	}

	void CalculationAll_states(long double limit, long double precision = 0.0001) {
		size_t i = 0;
		while (true) {
			n_state_Сalculation(i);
			auto a = GetOutput_states();
			if (std::abs(a(a.rows() - 1, 0) - limit) <= precision) {
				break;
			}
		}
	}

	void CalculationAll_states() {
		for (Eigen::Index i = 0; i < this->Input_states.rows(); ++i) {
			n_state_Сalculation(i);
		}
	}

	MatrixXld GetOutput_states() const {
		return ActivationFunctions::Tanh(Hidden_states * Output_weights + Output_bias);
	}

	std::vector<MatrixXld> GetWeightsAndDisplacement() {
		return {
			this->Weights_for_FG_HS, this->Weights_for_FG_IS, this->Displacements_for_FG,
			this->Weights_for_IG_HS, this->Weights_for_IG_IS, this->Displacements_for_IG,
			this->Weights_for_CT_HS, this->Weights_for_CT_IS, this->Displacements_for_CT,
			this->Weights_for_OG_HS, this->Weights_for_OG_IS, this->Displacements_for_OG
		};
	}

	void Train(const MatrixXld& inputs, const MatrixXld& targets, size_t epochs = 10000, long double learning_rate = 0.01, bool Enable_debugging_messages = false, bool decrease_learning_rate = false, int patience_ = 6000) {
		if (inputs.rows() != targets.rows())
			throw std::invalid_argument("Input and target row mismatch");
		SetInput_states(inputs);
		long double best_error = std::numeric_limits<long double>::max();
		int patience = patience_;
		int wait = 0;
		for (size_t epoch = 0; epoch < epochs; ++epoch) {
			// Очищаем состояния гейтов перед каждой эпохой
			FG_states.clear();
			IG_states.clear();
			CT_states.clear();
			OG_states.clear();

			Cell_states.setZero();
			Hidden_states.setZero();
			// Прямой проход
			for (Eigen::Index t = 0; t < inputs.rows(); ++t) {
				n_state_Сalculation(t);
			}
			// Обратный проход и обновление весов
			MatrixXld predictions = ActivationFunctions::Sigmoid(GetOutput_states());
			MatrixXld error = (predictions - targets).cwiseQuotient(
				predictions.unaryExpr([](long double x) { return x * (1 - x) + 1e-8L; })
			);

			long double current_error = error.norm();

			if (current_error < best_error) {
				best_error = current_error;
				wait = 0;
			}
			else {
				if (++wait >= patience) {
					std::cout << "Early stopping at epoch " << epoch << std::endl;
					break;
				}
			}
			auto has_nan = [](const MatrixXld& m) {
				return !((m.array() == m.array()).all()); // NaN != NaN
				};

			if (has_nan(predictions) || has_nan(error)) {
				std::cerr << "NaN detected at epoch " << epoch << std::endl;
				break;
			}
			if (!Hidden_states.allFinite()) {
				std::cerr << "Hidden_states contains NaN or Inf!" << std::endl;
			}
			if (decrease_learning_rate) {
				learning_rate = 0.001L * std::pow(0.95L, epoch / 100.0L); // Используйте 100.0L для плавающего деления
			}
			if (epoch % 100 == 0 && Enable_debugging_messages == true) {
				MatrixXld current_pred = GetOutput_states();
				auto denorm = SimpleLSTM::denormalize(current_pred);
				std::cout << "Intermediate prediction (numeric):\n" << current_pred << std::endl;
				std::cout << "Denormalized: ";
				for (Eigen::Index i = 0; i < current_pred.rows(); ++i) {
					for (Eigen::Index j = 0; j < current_pred.cols(); ++j) {
						std::cout << static_cast<int>(denorm[i][j]) << " : " << denorm[i][j]; // Вывод ASCII кодов
					}
				}
				std::cout << std::endl;
				std::cout << "Epoch: " << epoch
					<< " Error: " << current_error
					<< " LR: " << learning_rate << "\n";
				LSTMGradients grads = Backward(error);
				std::cout << "Gradient norms: "
					<< grads.dW_fg_hs.norm() << " "
					<< grads.dW_ig_hs.norm() << " "
					<< grads.dW_og_hs.norm() << std::endl;
			}

			LSTMGradients grads = Backward(error);
			for (auto* mat : grads.GetAll()) {
				if (!mat->allFinite()) {
					std::cerr << "NaN detected in gradient before UpdateWeights at epoch " << epoch << std::endl;
					return;
				}
			}
			UpdateWeights(grads, learning_rate, epoch, targets);
		}
	}

	void vector_Train(std::vector<MatrixXld> vec_, size_t epochs, long double learning_rate, long double precision = 0.001) {
		for(size_t epoch_ = 0; epoch_ < vec_.size() / 2; ++epoch_){
			auto inputs = vec_[2*epoch_];
			auto targets = vec_[2 * epoch_ + 1];
			SetInput_states(inputs);
			for (size_t epoch = 0; epoch < epochs; ++epoch) {
				if (CalculationAll_states(targets, targets.rows(), precision)) {
					break;
				}
				MatrixXld predictions = GetOutput_states();
				MatrixXld error = predictions - targets;
				//std::cout << targets(1, 0) << std::endl;
				//std::cout << predictions(1, 0) << std::endl;
				// Отладочный вывод
				if (epoch % 100 == 0) {
					for (Eigen::Index i = 0; i < predictions.rows(); i++) {
						for (Eigen::Index j = 0; j < predictions.cols(); j++) {
							std::cout << "Epoch " << epoch
								<< ", Prediction: " << predictions(i, j)
								<< ", Target: " << targets(i, j)
								<< ", Error: " << error(i, j) << std::endl;
						}
					}
				}

				LSTMGradients grads = Backward(error);
				UpdateWeights(grads, learning_rate);
			}
		}
	}

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

		// Сохраняем текущие состояния (без истории)
		save_matrix(file, this->Input_states);
		save_matrix(file, this->Hidden_states);
		save_matrix(file, this->Cell_states);

		// Сохраняем веса и смещения
		save_matrix(file, this->Weights_for_FG_HS);
		save_matrix(file, this->Weights_for_IG_HS);
		save_matrix(file, this->Weights_for_CT_HS);
		save_matrix(file, this->Weights_for_OG_HS);
		save_matrix(file, this->Weights_for_FG_IS);
		save_matrix(file, this->Weights_for_IG_IS);
		save_matrix(file, this->Weights_for_CT_IS);
		save_matrix(file, this->Weights_for_OG_IS);
		save_matrix(file, this->Displacements_for_FG);
		save_matrix(file, this->Displacements_for_IG);
		save_matrix(file, this->Displacements_for_CT);
		save_matrix(file, this->Displacements_for_OG);

		// Пустые векторы состояний
		save_vector(file, this->FG_states);
		save_vector(file, this->IG_states);
		save_vector(file, this->CT_states);
		save_vector(file, this->OG_states);
	}

	void load(const std::string& filename) {
		std::ifstream file(filename);
		if (!file) throw std::runtime_error("Cannot open file for reading");

		file >> this->Input_size >> this->Hidden_size;

		// Сохраните все матрицы и векторы:
		load_matrix(file, this->Input_states);
		load_matrix(file, this->Hidden_states);
		load_matrix(file, this->Cell_states);

		load_matrix(file, this->Weights_for_FG_HS);
		load_matrix(file, this->Weights_for_IG_HS);
		load_matrix(file, this->Weights_for_CT_HS);
		load_matrix(file, this->Weights_for_OG_HS);

		load_matrix(file, this->Weights_for_FG_IS);
		load_matrix(file, this->Weights_for_IG_IS);
		load_matrix(file, this->Weights_for_CT_IS);
		load_matrix(file, this->Weights_for_OG_IS);

		load_matrix(file, this->Displacements_for_FG);
		load_matrix(file, this->Displacements_for_IG);
		load_matrix(file, this->Displacements_for_CT);
		load_matrix(file, this->Displacements_for_OG);

		// Сохраняем векторы состояний
		load_vector(file, this->FG_states);
		load_vector(file, this->IG_states);
		load_vector(file, this->CT_states);
		load_vector(file, this->OG_states);
	}

//private:

protected:
	struct LSTMGradients {
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
	};

	size_t Input_size;
	size_t Hidden_size;
	MatrixXld Input_states;
	MatrixXld Hidden_states;
	MatrixXld Cell_states;

	MatrixXld Weights_for_FG_HS;  // Forget gate hidden state weights
	MatrixXld Weights_for_IG_HS;  // Input gate hidden state weights
	MatrixXld Weights_for_CT_HS;  // Cell state hidden state weights
	MatrixXld Weights_for_OG_HS;  // Output gate hidden state weights

	MatrixXld Weights_for_FG_IS;  // Forget gate input weights
	MatrixXld Weights_for_IG_IS;  // Input gate input weights
	MatrixXld Weights_for_CT_IS;  // Cell state input weights
	MatrixXld Weights_for_OG_IS;  // Output gate input weights

	MatrixXld Displacements_for_FG;  // Матрица 1xHidden_size
	MatrixXld Displacements_for_IG;  // Матрица 1xHidden_size
	MatrixXld Displacements_for_CT;  // Матрица 1xHidden_size
	MatrixXld Displacements_for_OG;  // Матрица 1xHidden_size

	std::vector<MatrixXld> FG_states;
	std::vector<MatrixXld> IG_states;
	std::vector<MatrixXld> CT_states;
	std::vector<MatrixXld> OG_states;

	MatrixXld Output_weights; // (Hidden_size x 1)
	MatrixXld Output_bias;    // (1 x 1)


	std::vector<MatrixXld> StepСalculation(const MatrixXld& Hidden_State, const MatrixXld& Last_State, const MatrixXld& Input_State){
		// Проверка размерностей
		if (Hidden_State.cols() != static_cast<Eigen::Index>(Hidden_size) ||
			Hidden_State.rows() != 1 ||
			Last_State.cols() != static_cast<Eigen::Index>(Hidden_size) ||
			Last_State.rows() != 1 ||
			Input_State.cols() != static_cast<Eigen::Index>(Input_size) ||
			Input_State.rows() != 1)
		{
			throw std::invalid_argument("Invalid state dimensions in StepCalculation");
		}


		// Forget Gate (1xHidden_size)
		MatrixXld forget_gate = ActivationFunctions::Sigmoid(
			(Weights_for_FG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_FG_IS * Input_State.transpose()).transpose() +
			Displacements_for_FG
		);

		// Input Gate (1xHidden_size)
		MatrixXld input_gate = ActivationFunctions::Sigmoid(
			(Weights_for_IG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_IG_IS * Input_State.transpose()).transpose() +
			Displacements_for_IG
		);

		// Cell State Candidate (1xHidden_size)
		MatrixXld ct_candidate = ActivationFunctions::Tanh(
			(Weights_for_CT_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_CT_IS * Input_State.transpose()).transpose() +
			Displacements_for_CT
		);

		// New Cell State (1xHidden_size)
		MatrixXld new_cell_state = forget_gate.cwiseProduct(Last_State) + input_gate.cwiseProduct(ct_candidate);

		// Output Gate (1xHidden_size)
		MatrixXld output_gate = ActivationFunctions::Sigmoid(
			(Weights_for_OG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_OG_IS * Input_State.transpose()).transpose() +
			Displacements_for_OG
		);
		
		// New Hidden State (1xHidden_size)
		MatrixXld new_hidden_state = output_gate.cwiseProduct(ActivationFunctions::Tanh(new_cell_state));

		new_hidden_state = new_hidden_state.unaryExpr([](long double x) {
			return std::max(-5.0L, std::min(5.0L, x)); // Расширяем диапазон
			});

		if (!new_cell_state.allFinite() || !new_hidden_state.allFinite()) {
			throw std::runtime_error("NaN/Inf in states c or h");
		}
		return { new_cell_state, new_hidden_state, forget_gate, input_gate, ct_candidate, output_gate };
		new_cell_state = new_cell_state.unaryExpr([](long double x) {
			return std::max(-5.0L, std::min(5.0L, x));
			});
		new_hidden_state = new_hidden_state.unaryExpr([](long double x) {
			return std::max(-1.0L, std::min(1.0L, x));
			});
		return { new_cell_state, new_hidden_state, forget_gate, input_gate, ct_candidate, output_gate };
	}

	void n_state_Сalculation(size_t timestep) {
		if (timestep >= static_cast<size_t>(Input_states.rows())) {
			throw std::out_of_range("Invalid timestep");
		}

		MatrixXld input = Input_states.row(timestep);
		MatrixXld prev_hidden = Hidden_states.row(timestep);
		MatrixXld prev_cell = Cell_states.row(timestep);

		auto results = StepСalculation(prev_hidden, prev_cell, input);

		// Обновляем состояния напрямую в предварительно выделенной матрице
		Cell_states.row(timestep + 1) = results[0];
		Hidden_states.row(timestep + 1) = results[1];

		FG_states.push_back(results[2]);
		IG_states.push_back(results[3]);
		CT_states.push_back(results[4]);
		OG_states.push_back(results[5]);
	}

	LSTMGradients Backward(const MatrixXld& error) {
    LSTMGradients grads;

    MatrixXld dC_next = MatrixXld::Zero(1, Hidden_size);
    MatrixXld dh_next = MatrixXld::Zero(1, Hidden_size);

	Eigen::Index T = Input_states.rows(); // Используем количество входных шагов

    auto clamp = [](MatrixXld& m, long double low = -0.95L, long double high = 0.95L) {
        m = m.unaryExpr([&](long double x) {
            return std::max(low, std::min(high, x));
        });
    };

    for (Eigen::Index t = T - 2; t >= 0; --t) {
		if (FG_states.size() != T - 1 || IG_states.size() != T - 1 || CT_states.size() != T - 1 || OG_states.size() != T - 1) {
			throw std::runtime_error("Vector states size mismatch in Backward");
		}

        const MatrixXld& hs_t = Hidden_states.row(t);
        const MatrixXld& x_t = Input_states.row(t);
        const MatrixXld& f_t = FG_states[t];
        const MatrixXld& i_t = IG_states[t];
        const MatrixXld& TC_t = CT_states[t];
        const MatrixXld& o_t = OG_states[t];
		MatrixXld C_prev;
		if (t > 0)
			C_prev = Cell_states.row(t - 1);
		else
			C_prev = MatrixXld::Zero(1, Hidden_size);
        const MatrixXld& C_t = Cell_states.row(t);

        MatrixXld f = f_t;
        MatrixXld i = i_t;
        MatrixXld TC = TC_t;
        MatrixXld o = o_t;

        // Укреплённый clamp
        clamp(TC);
        clamp(i, 0.05L, 0.95L);
        clamp(f, 0.05L, 0.95L);
        clamp(o, 0.05L, 0.95L);

        // Стабилизированный tanh(C_t)
        MatrixXld tanh_Ct = ActivationFunctions::Tanh(C_t);
        tanh_Ct = tanh_Ct.unaryExpr([](long double x) {
            return std::max(-0.999999L, std::min(0.999999L, x));
        });

        MatrixXld dh = error.row(t) + dh_next;

        // Clip dh
        dh = dh.unaryExpr([](long double x) {
            return std::max(std::min(x, 5.0L), -5.0L);
        });

        MatrixXld dC = dh.cwiseProduct(o.cwiseProduct((1.0L - tanh_Ct.array().square()).matrix())) + dC_next;

        // Clip dC
        dC = dC.unaryExpr([](long double x) {
            return std::max(std::min(x, 5.0L), -5.0L);
        });

		MatrixXld f_term = f.array() * (1.0L - f.array());
		MatrixXld df = dC.cwiseProduct(C_prev).cwiseProduct(f_term.matrix());

		MatrixXld i_term = i.array() * (1.0L - i.array());
		MatrixXld di = dC.cwiseProduct(TC).cwiseProduct(i_term.matrix());

		MatrixXld TC_term = (1.0L - TC.array().square());
		MatrixXld dTC = dC.cwiseProduct(i).cwiseProduct(TC_term.matrix());

		MatrixXld o_term = o.array() * (1.0L - o.array());
		MatrixXld do_gate = dh.cwiseProduct(tanh_Ct).cwiseProduct(o_term.matrix());

        // Debug: check for NaNs
        if (!df.allFinite() || !di.allFinite() || !do_gate.allFinite()) {
            std::cerr << "NaN detected in gradient at timestep backprop. Zeroing gradients.\n";
            for (auto* g : grads.GetAll()) {
                *g = MatrixXld::Zero(g->rows(), g->cols());
            }
            break;
        }

        // Обновление градиентов
        grads.dW_fg_hs += hs_t.transpose() * df;
        grads.dW_fg_is += x_t.transpose() * df;
        grads.db_fg += df;

        grads.dW_ig_hs += hs_t.transpose() * di;
        grads.dW_ig_is += x_t.transpose() * di;
        grads.db_ig += di;

        grads.dW_ct_hs += hs_t.transpose() * dTC;
        grads.dW_ct_is += x_t.transpose() * dTC;
        grads.db_ct += dTC;

        grads.dW_og_hs += hs_t.transpose() * do_gate;
        grads.dW_og_is += x_t.transpose() * do_gate;
        grads.db_og += do_gate;

        // Градиенты для следующего шага назад
        dh_next = df * Weights_for_FG_HS.transpose()
                + di * Weights_for_IG_HS.transpose()
                + dTC * Weights_for_CT_HS.transpose()
                + do_gate * Weights_for_OG_HS.transpose();

        dC_next = dC.cwiseProduct(f);
    }

    // Итоговый клиппинг
    for (auto* g : grads.GetAll()) {
        *g = g->unaryExpr([](long double x) {
            return std::max(std::min(x, 0.5L), -0.5L);
        });
    }

    return grads;
}

	void UpdateWeights(LSTMGradients& gradients, long double learning_rate) {
		auto l2_reg = [](MatrixXld& weights, long double lambda) {
			weights -= lambda * weights;
			};
		auto clip_gradients = [](MatrixXld& grad, long double max_norm = 1.0L) {
			long double norm = grad.norm();
			if (std::isnan(norm)) {
				grad = MatrixXld::Zero(grad.rows(), grad.cols());
			}
			else if (norm > max_norm) {
				grad *= (max_norm / norm);
			}
		};

		clip_gradients(gradients.dW_fg_hs, 0.5L);
		clip_gradients(gradients.dW_fg_is, 0.5L);
		clip_gradients(gradients.db_fg, 0.5L);

		clip_gradients(gradients.dW_ig_hs, 0.5L);
		clip_gradients(gradients.dW_ig_is, 0.5L);
		clip_gradients(gradients.db_ig, 0.5L);

		clip_gradients(gradients.dW_ct_hs, 0.5L);
		clip_gradients(gradients.dW_ct_is, 0.5L);
		clip_gradients(gradients.db_ct, 0.5L);

		clip_gradients(gradients.dW_og_hs, 0.5L);
		clip_gradients(gradients.dW_og_is, 0.5L);
		clip_gradients(gradients.db_og, 0.5L);

		auto check_nan = [](const MatrixXld& m, const std::string& name) {
			for (Eigen::Index i = 0; i < m.rows(); ++i) {
				for (Eigen::Index j = 0; j < m.cols(); ++j) {
					if (std::isnan(m(i, j)) || std::isinf(m(i, j))) {
						throw std::runtime_error(name + " содержит NaN/Inf");
					}
				}
			}
			};
		
		check_nan(gradients.dW_fg_hs, "dW_fg_hs");
		check_nan(gradients.dW_fg_is, "dW_fg_is");
		check_nan(gradients.db_fg, "db_fg");

		check_nan(gradients.dW_ig_hs, "dW_ig_hs");
		check_nan(gradients.dW_ig_is, "dW_ig_is");
		check_nan(gradients.db_ig, "db_ig");

		check_nan(gradients.dW_ct_hs, "dW_ct_hs");
		check_nan(gradients.dW_ct_is, "dW_ct_is");
		check_nan(gradients.db_ct, "db_ct");

		check_nan(gradients.dW_og_hs, "dW_og_hs");
		check_nan(gradients.dW_og_is, "dW_og_is");
		check_nan(gradients.db_og, "db_og");

		auto safe_update = [learning_rate](MatrixXld& weights, const MatrixXld& grad) {
			for (Eigen::Index i = 0; i < grad.rows(); ++i) {
				for (Eigen::Index j = 0; j < grad.cols(); ++j) {
					if (std::isnan(grad(i, j))) {
						weights(i, j) -= 0;  // Игнорируем NaN
					}
					else {
						weights(i, j) -= learning_rate * grad(i, j);
					}
				}
			}
			};

		safe_update(Weights_for_FG_HS, gradients.dW_fg_hs);
		safe_update(Weights_for_FG_IS, gradients.dW_fg_is);
		safe_update(Displacements_for_FG, gradients.db_fg);

		safe_update(Weights_for_IG_HS, gradients.dW_ig_hs);
		safe_update(Weights_for_IG_IS, gradients.dW_ig_is);
		safe_update(Displacements_for_IG, gradients.db_ig);

		safe_update(Weights_for_CT_HS, gradients.dW_ct_hs);
		safe_update(Weights_for_CT_IS, gradients.dW_ct_is);
		safe_update(Displacements_for_CT, gradients.db_ct);

		safe_update(Weights_for_OG_HS, gradients.dW_og_hs);
		safe_update(Weights_for_OG_IS, gradients.dW_og_is);
		safe_update(Displacements_for_OG, gradients.db_og);

		/*Weights_for_FG_HS -= gradients.dW_fg_hs * learning_rate;
		Weights_for_FG_IS -= gradients.dW_fg_is * learning_rate;
		Displacements_for_FG -= gradients.db_fg * learning_rate;

		Weights_for_IG_HS -= gradients.dW_ig_hs * learning_rate;
		Weights_for_IG_IS -= gradients.dW_ig_is * learning_rate;
		Displacements_for_IG -= gradients.db_ig * learning_rate;

		Weights_for_CT_HS -= gradients.dW_ct_hs * learning_rate;
		Weights_for_CT_IS -= gradients.dW_ct_is * learning_rate;
		Displacements_for_CT -= gradients.db_ct * learning_rate;

		Weights_for_OG_HS -= gradients.dW_og_hs * learning_rate;
		Weights_for_OG_IS -= gradients.dW_og_is * learning_rate;
		Displacements_for_OG -= gradients.db_og * learning_rate;*/
	}

	void UpdateWeights(LSTMGradients& gradients, long double learning_rate, size_t epoch, const MatrixXld& targets) {
		const long double beta1 = 0.95L;  // Было 0.9
		const long double beta2 = 0.9999L; // Было 0.999
		const long double epsilon = 1e-8L; // Было 1e-6
		const long double lambda = 0.001L; // L2 регуляризация

		std::vector<MatrixXld*> params = {
			&Weights_for_FG_HS, &Weights_for_FG_IS, &Displacements_for_FG,
			&Weights_for_IG_HS, &Weights_for_IG_IS, &Displacements_for_IG,
			&Weights_for_CT_HS, &Weights_for_CT_IS, &Displacements_for_CT,
			&Weights_for_OG_HS, &Weights_for_OG_IS, &Displacements_for_OG
		};

		std::vector<MatrixXld*> grads = gradients.GetAll();

		// Инициализация моментов
		static std::vector<MatrixXld> m, v;
		if (m.empty()) {
			for (const auto* g : grads) {
				m.emplace_back(MatrixXld::Zero(g->rows(), g->cols()));
				v.emplace_back(MatrixXld::Zero(g->rows(), g->cols()));
			}
		}

		for (size_t i = 0; i < grads.size(); ++i) {
			auto& grad = *grads[i];
			auto& weight = *params[i];

			// Градиентный клиппинг
			long double norm = grad.norm();
			if (std::isnan(norm)) grad.setZero();
			else if (norm > 0.5L) grad *= (0.5L / norm);

			// Обновление моментов
			m[i] = beta1 * m[i] + (1.0L - beta1) * grad;
			v[i] = beta2 * v[i] + (1.0L - beta2) * grad.cwiseProduct(grad);

			// Коррекция смещений
			long double b1_corr = 1.0L - std::pow(beta1, epoch + 1);
			long double b2_corr = 1.0L - std::pow(beta2, epoch + 1);
			b1_corr = std::max(b1_corr, 1e-8L);
			b2_corr = std::max(b2_corr, 1e-8L);
			//if (b2_corr < 1e-5L) b2_corr = 1e-5L;

			MatrixXld m_hat = m[i] / b1_corr;
			MatrixXld v_hat = v[i] / b2_corr;

			// Безопасное деление
			MatrixXld denom = (v_hat.array().sqrt() + epsilon).matrix();
			denom = denom.unaryExpr([](long double x) {
				return std::max(1e-6L, std::min(1e+2L, x));
				});

			MatrixXld update = m_hat.array() / denom.array();
			update = update.unaryExpr([](long double x) {
				return std::max(-10.0L, std::min(10.0L, x));
				});

			if (!update.allFinite()) {
				std::cerr << "NaN or Inf in update at param " << i << " epoch " << epoch << "\n";
				continue;
			}

			// Обновление веса
			weight -= learning_rate * update.matrix();

			// L2-регуляризация
			weight *= (1.0L - lambda * learning_rate);
		}

		MatrixXld predictions = this->GetOutput_states();
		MatrixXld output_error = predictions - targets;
		MatrixXld grad_out_w = Hidden_states.transpose() * output_error / Hidden_states.rows();
		MatrixXld grad_out_b = output_error.colwise().mean();

		// 2. Отсечение градиентов для стабильности
		auto clip_gradients = [](MatrixXld& grad, long double max_norm = 1.0L) {
			long double norm = grad.norm();
			if (norm > max_norm) {
				grad *= (max_norm / norm);
			}
			};
		clip_gradients(grad_out_w);  // Применяем к весам
		clip_gradients(grad_out_b);  // Применяем к смещениям

		// 3. Обновляем веса и смещения выходного слоя
		Output_weights -= learning_rate * grad_out_w;
		Output_bias -= learning_rate * grad_out_b;

		// 4. L2-регуляризация (после обновления!)
		const long double output_lambda = 0.001L;
		Output_weights *= (1.0L - output_lambda * learning_rate);
		Output_bias *= (1.0L - output_lambda * learning_rate);
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

	void save_vector(std::ofstream& file, const std::vector<MatrixXld>& vec) const {
		file << vec.size() << "\n";
		for (const auto& m : vec) {
			save_matrix(file, m);
		}
	}

	void load_vector(std::ifstream& file, std::vector<MatrixXld>& vec) {
		size_t size;
		file >> size;
		vec.resize(size);
		for (auto& m : vec) {
			load_matrix(file, m);
		}
	}

};


int main() {
	setlocale(LC_ALL, "Russian");
	try {
		SimpleLSTM test(1, 32);
		MatrixXld input(2, 1);
		input << SimpleLSTM::normalize('М'),
			SimpleLSTM::normalize('Д');

		MatrixXld target(2, 1);
		target << SimpleLSTM::normalize('М'),
			SimpleLSTM::normalize(' ');
		//std::cout << static_cast<int>(' '); // = 32
		// Уменьшаем количество эпох для теста
		test.CalculationAll_states();
		auto a = test.GetOutput_states();
		test.Train(input, target,
			10000,    // Увеличиваем количество эпох
			0.001L,     
			true,
			true,
			500       // Увеличиваем терпение для ранней остановки
		);
		test.SetInput_states(input);
		test.CalculationAll_states();

		MatrixXld predictions = test.GetOutput_states();
		auto denorm = SimpleLSTM::denormalize(predictions);
		std::cout << "\nFinal Result: ";
		for (auto& row : denorm) {
			for (auto c : row) std::cout << c;
		}
		std::cout << "Normalized space: "
			<< SimpleLSTM::normalize(' ') << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
	return 0;
}