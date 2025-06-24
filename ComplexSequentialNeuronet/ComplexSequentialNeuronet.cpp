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

	//MatrixXld Output_weights; // (Hidden_size x 1)
	//MatrixXld Output_bias;    // (1 x 1)

	private:
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

// ==== БАЗОВАЯ Seq2Seq + Attention ====
class Seq2SeqWithAttention {
public:
	class BahdanauAttention : public Attention {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
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

	protected:
		Eigen::Index encoder_hidden_size_;    // 2H
		Eigen::Index decoder_hidden_size_;    // H_dec
		Eigen::Index attention_size_;         // A

		MatrixXld W_encoder_;       // [A x 2H]
		MatrixXld W_decoder_;       // [A x H_dec]
		MatrixXld attention_vector_; // [A x 1]
	};
	class Encoder : public BiLSTM {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		Encoder(Eigen::Index input_size, Eigen::Index hidden_size)
			: BiLSTM(input_size, hidden_size) {
		}
		Encoder() : BiLSTM() {};

		void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
			SetInput_states(input_sequence_batch);
			All_state_Сalculation();
		}

		const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
			return this->Common_Hidden_states;
		}
		virtual ~Encoder() = default;
	};
	class Decoder : public SimpleLSTM {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		Decoder(std::shared_ptr</*Attention*/BahdanauAttention> attention_module,
			Eigen::Index hidden_size_encoder, Eigen::Index Hidden_size_, Eigen::Index embedding_dim_, 
			RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_)
			: SimpleLSTM(embedding_dim_ + 2 * hidden_size_encoder/*= H_emb + 2H_enc*/, Hidden_size_), attention_(std::move(attention_module))
		{
			this->output_size = embedding_dim_;
			//размер контекста = 2 * Hidden_size_encoder = Number_states - embedding_dim
			size_t context_size = 2 * hidden_size_encoder;
			W_output = ActivationFunctions::matrix_random(output_size, Hidden_size_ + context_size);
			b_output = RowVectorXld::Zero(output_size);

			this->layernorm_gamma = RowVectorXld::Ones(Input_size);
			this->layernorm_beta = RowVectorXld::Zero(Input_size);
			// теперь SimpleLSTM::Input_size = Number_states, Hidden_size = Hidden_size_

			this->start_token = start_token_;   // эмбеддинг стартового токена (1 символ)
			this->end_token = end_token_;     // матрица эмбеддингов финишного токена (несколько символов)
			this->max_steps = max_steps_;    // ограничение на число шагов генерации
		}

		void SetEncoderOutputs(const std::vector<MatrixXld>& encoder_outputs) {
			this->encoder_outputs = encoder_outputs;
		}

		void Decode(const std::vector<MatrixXld>& encoder_outputs) {
			this->SetEncoderOutputs(encoder_outputs);
			this->All_state_Calculation();
		}

		const std::vector<MatrixXld>& GetOutputStates() const { return Output_state; }

	protected:
		bool IsEndToken(const RowVectorXld& vec) const {
			for (int i = 0; i < end_token.rows(); ++i) {
				if ((vec - end_token.row(i)).norm() < 1e-6L) return true;
			}
			return false;
		}
		void All_state_Calculation() {
			if (this->encoder_outputs.empty()) return;

			auto apply_layernorm = [this](const RowVectorXld& x) -> RowVectorXld {
				long double mean = x.mean();
				long double variance = (x.array() - mean).square().mean();
				return ((x.array() - mean) / std::sqrt(variance + 1e-5L)).matrix().array() * layernorm_gamma.array() + layernorm_beta.array();
				};

			auto l2_normalize = [](const RowVectorXld& x) -> RowVectorXld {
				long double norm = std::sqrt(x.squaredNorm() + 1e-8L);
				return x / norm;
				};

			// --- Lambda: Масштабирование по max(abs)
			auto normalize_scale = [](RowVectorXld& vec) {
				long double maxval = vec.cwiseAbs().maxCoeff();
				if (maxval > 0.0L) vec /= maxval;
				};

			// Очистка
			Output_state.clear();
			context_vectors.clear();
			U_state.clear();
			attention_->ClearCache();

			size_t batch_size = encoder_outputs.size();
			Output_state.resize(batch_size);
			context_vectors.resize(batch_size);
			U_state.resize(batch_size);

			// Общие веса
			MatrixXld W_x(Input_size, 4 * Hidden_size);
			W_x << W_F_I, W_I_I, W_C_I, W_O_I;

			MatrixXld W_h(Hidden_size, 4 * Hidden_size);
			W_h << W_F_H, W_I_H, W_C_H, W_O_H;

			RowVectorXld b(4 * Hidden_size);
			b << B_F, B_I, B_C, B_O;

			for (size_t n = 0; n < batch_size; ++n) {
				const auto& enc_out = encoder_outputs[n];
				std::vector<RowVectorXld> y_sequence;
				std::vector<RowVectorXld> context_sequence;
				std::vector<RowVectorXld> u_sequence;

				RowVectorXld y_prev = start_token;
				RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
				RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

				for (size_t t = 0; t < max_steps; ++t) {
					RowVectorXld context = attention_->ComputeContext(enc_out, h_prev);
					context_sequence.push_back(context);

					RowVectorXld decoder_input(Input_size);
					decoder_input << y_prev, context;
					decoder_input = l2_normalize(decoder_input);

					RowVectorXld Z = decoder_input * W_x + h_prev * W_h + b;

					RowVectorXld f_t = ActivationFunctions::Sigmoid(Z.leftCols(Hidden_size));
					RowVectorXld i_t = ActivationFunctions::Sigmoid(Z.middleCols(Hidden_size, Hidden_size));
					RowVectorXld c_bar = ActivationFunctions::Tanh(Z.middleCols(2 * Hidden_size, Hidden_size));
					RowVectorXld o_t = ActivationFunctions::Sigmoid(Z.rightCols(Hidden_size));

					RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_bar.array();
					RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();

					RowVectorXld proj_input(Hidden_size + context.size());
					proj_input << h_t, context;
					proj_input = apply_layernorm(proj_input);

					RowVectorXld y_t = proj_input * W_output.transpose() + b_output;
					u_sequence.push_back(proj_input);
					y_sequence.push_back(y_t);

					if (IsEndToken(y_t)) {
						size_t end_len = static_cast<size_t>(end_token.rows());
						if (y_sequence.size() >= end_len - 1) {
							y_sequence.resize(y_sequence.size() - (end_len - 1));
							context_sequence.resize(context_sequence.size() - (end_len - 1));
							u_sequence.resize(u_sequence.size() - (end_len - 1));
						}
						break;
					}

					y_prev = y_t;
					h_prev = h_t;
					c_prev = c_t;
				}

				// Преобразуем в матрицы
				Eigen::Index T = static_cast<Eigen::Index>(y_sequence.size());
				Eigen::Index D = static_cast<Eigen::Index>(y_sequence[0].cols());

				Output_state[n] = MatrixXld(T, D);
				U_state[n] = MatrixXld(T, u_sequence[0].cols());
				context_vectors[n] = MatrixXld(T, context_sequence[0].cols());

				for (Eigen::Index t = 0; t < T; ++t) {
					Output_state[n].row(t) = y_sequence[t];
					U_state[n].row(t) = u_sequence[t];
					context_vectors[n].row(t) = context_sequence[t];
				}
			}
		}

		RowVectorXld start_token;   // эмбеддинг стартового токена (1 символ)
		MatrixXld end_token;     // матрица эмбеддингов финишного токена (несколько символов)
		size_t max_steps;    // ограничение на число шагов генерации

		std::shared_ptr</*Attention*/BahdanauAttention> attention_;

		std::vector<MatrixXld> encoder_outputs;
		std::vector<MatrixXld> context_vectors;

		std::vector<MatrixXld> U_state;

		std::vector<MatrixXld> Output_state;
		// --- Обновляемый выходной слой ---
		MatrixXld W_output;      // [output_size x (hidden_size + context_size)]
		RowVectorXld b_output;   // [1 x output_size]

		size_t output_size;
		//size_t embedding_dim;

		RowVectorXld layernorm_gamma; // [1 x Input_size]
		RowVectorXld layernorm_beta;  // [1 x Input_size]
	};

	template<typename EncoderT, typename DecoderT>
	Seq2SeqWithAttention(
		std::unique_ptr<EncoderT> encoder = std::make_unique<Encoder>,
		std::unique_ptr<DecoderT> decoder = std::make_unique<Decoder>)
		: encoder_(std::move(encoder)), decoder_(std::move(decoder)) {
	}

	Seq2SeqWithAttention(
		Eigen::Index Input_size_, Eigen::Index Encoder_Hidden_size_, Eigen::Index Decoder_Hidden_size_,
		Eigen::Index Output_size, RowVectorXld start_token_, MatrixXld end_token_, size_t max_steps_, 
		std::unique_ptr<Attention> attention_ = std::make_unique<BahdanauAttention>())
		: 
		encoder_(std::make_unique<Encoder>(Input_size_, Encoder_Hidden_size_)), 
		decoder_(std::make_unique<Decoder>(attention_, Encoder_Hidden_size_, Decoder_Hidden_size_, Output_size, start_token_, end_token_, max_steps_)) {
	}

	void SetInput_states(const std::vector<MatrixXld>& _inputs) {
		this->Input_States = _inputs;
	}

	void Inference()
	{
		if(this->Input_States.empty()){ throw std::invalid_argument("Вход пустой"); }
		encoder_->Encode(this->Input_States);
		decoder_->Decode(encoder_->GetEncodedHiddenStates());
	}

	void Inference(const std::vector<MatrixXld>& input_sequence_batch)
	{
		SetInput_states(input_sequence_batch);
		encoder_->Encode(this->Input_States);
		decoder_->Decode(encoder_->GetEncodedHiddenStates());
	}

	const std::vector<MatrixXld>& GetDecoderOutputs() const {
		return decoder_->GetOutputStates();
	}

	void Save(std::string packname) {
		std::filesystem::create_directories(packname);
		encoder_->Save(packname + "/" + "Encoder");
		decoder_->save(packname + "/" + "Decoder");
	}

	void Load(std::string packname) {
		encoder_->Load(packname + "/" + "Encoder");
		decoder_->load(packname + "/" + "Decoder");
	}
protected:
	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
	std::vector<MatrixXld> Input_States;

};


// ==== ТРЕНИРОВОЧНАЯ Seq2Seq + Attention ====
class Seq2SeqWithAttention_ForTrain : public Seq2SeqWithAttention {
public:
	class SimpleLSTM_ForTrain;
	class BiLSTM_ForTrain;

	class SimpleLSTM_ForTrain : public SimpleLSTM {
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

		void save(const std::string& filename) const {
			std::ofstream file(filename, std::ios::trunc); // Используйте trunc для перезаписи
			if (!file) throw std::runtime_error("Cannot open file for writing");

			// Сохраняем только актуальные параметры
			file << this->Input_size << "\n" << this->Hidden_size << "\n" << this->Batch_size << "\n";

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

			file >> this->Input_size >> this->Hidden_size >> this->Batch_size;

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

	class BiLSTM_ForTrain : public BiLSTM {
	public:
		BiLSTM_ForTrain(size_t Batch_size_ = 32, Eigen::Index Number_states = 1, Eigen::Index Hidden_size_ = 10)
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

		//BiLSTM_ForTrain() = default;

		~BiLSTM_ForTrain() {
			this->Save("BiLSTM_state_ForTrain.txt");
		}

		void Batch_All_state_Сalculation() {
			this->Forward.Batch_All_state_Сalculation();
			this->Backward.Batch_All_state_Сalculation();
			Common_Hidden_states.clear();
			Common_Hidden_states.resize(Forward.Hidden_states.size());
			for (size_t i = 0; i < Forward.Hidden_states.size(); ++i) {
				const auto& Hf = Forward.Hidden_states[i];  // [T × H]
				const auto& Hb = Backward.Hidden_states[i]; // [T × H], но обратный порядок
				size_t T = Hf.rows();
				MatrixXld concat(T, 2 * this->Common_Hidden_size);
				for (size_t t = 0; t < T; ++t) {
					concat.row(t) << Hf.row(t), Hb.row(T - 1 - t);
				}
				Common_Hidden_states[i] = concat;
			}
		}
	protected:
		SimpleLSTM_ForTrain Forward;
		SimpleLSTM_ForTrain Backward;
		size_t Common_Batch_size;
	};

	class Encoder : public Seq2SeqWithAttention::Encoder, public BiLSTM_ForTrain {
	public:
		friend class Seq2SeqWithAttention_ForTrain;
		using BiLSTM_ForTrain::SetInput_states;
		using BiLSTM_ForTrain::Common_Hidden_states;

		Encoder(size_t batch_size, Eigen::Index input_size, Eigen::Index hidden_size)
			: BiLSTM_ForTrain(batch_size, input_size, hidden_size) {
		}
		Encoder() = default;

		void Encode(const std::vector<MatrixXld>& input_sequence_batch) {
			this->SetInput_states(input_sequence_batch);
			this->Batch_All_state_Сalculation();
		}

		const std::vector<MatrixXld>& GetEncodedHiddenStates() const {
			return this->Common_Hidden_states;
		}

	};

	class Decoder : public Seq2SeqWithAttention::Decoder, public SimpleLSTM_ForTrain{
		friend class Seq2SeqWithAttention_ForTrain;

		using SimpleLSTM_ForTrain::Input_states;
		using SimpleLSTM_ForTrain::Input_size;
		using SimpleLSTM_ForTrain::Hidden_size;
		using SimpleLSTM_ForTrain::Hidden_states;
		using SimpleLSTM_ForTrain::Cell_states;

		using SimpleLSTM_ForTrain::W_F_H;  // Forget gate hidden state weights
		using SimpleLSTM_ForTrain::W_I_H;  // Input gate hidden state weights
		using SimpleLSTM_ForTrain::W_C_H;  // Cell state hidden state weights
		using SimpleLSTM_ForTrain::W_O_H;  // Output gate hidden state weights

		using SimpleLSTM_ForTrain::W_F_I;  // Forget gate input weights
		using SimpleLSTM_ForTrain::W_I_I;  // Input gate input weights
		using SimpleLSTM_ForTrain::W_C_I;  // Cell state input weights
		using SimpleLSTM_ForTrain::W_O_I;  // Output gate input weights

		using SimpleLSTM_ForTrain::B_F;  // Матрица 1xHidden_size
		using SimpleLSTM_ForTrain::B_I;  // Матрица 1xHidden_size
		using SimpleLSTM_ForTrain::B_C;  // Матрица 1xHidden_size
		using SimpleLSTM_ForTrain::B_O;  // Матрица 1xHidden_size

	public:
		/*Decoder(std::shared_ptr</*Attention*//*BahdanauAttention> attention_module,
			size_t Batch_size,
			Eigen::Index Number_states,   // = H_emb + 2H_enc
			Eigen::Index Hidden_size_, Eigen::Index embedding_dim_)
			: Seq2SeqWithAttention::Decoder(std::move(attention_module), Number_states, Hidden_size_, embedding_dim_),
			SimpleLSTM_ForTrain(Batch_size, Number_states, Hidden_size_)
		{
			// теперь SimpleLSTM::Input_size = Number_states, Hidden_size = Hidden_size_
		}*/

		// --- Настройка целей для обучения ---
		void SetTargets(const std::vector<MatrixXld>& targets) {
			target_outputs_ = targets;
		}

		const std::vector<MatrixXld>& GetTargets() const {
			return target_outputs_;
		}

		// --- Результаты прямого прохода ---
		std::vector<MatrixXld> logits;  // [B][T_dec x output_size]

		const std::vector<MatrixXld>& GetLogits() const { return logits; }
		
		void Batch_All_state_Сalculation() {
			if (Input_states.empty() || encoder_outputs.empty()) return;

			// Очистка
			logits.clear();
			context_vectors.clear();
			U_state.clear();
			Hidden_states.clear();
			Cell_states.clear();
			attention_->ClearCache();

			const size_t batch_size = Input_states.size();
			const size_t T_dec = Input_states[0].rows();
			const size_t enc_dim = encoder_outputs[0].cols();
			const size_t H = Hidden_size;
			const size_t E = output_size;
			const size_t C = enc_dim;
			const size_t D = H + C;

			// Подготовка выходных контейнеров
			logits.resize(batch_size, MatrixXld::Zero(T_dec, E));
			context_vectors.resize(batch_size, MatrixXld::Zero(T_dec, C));
			U_state.resize(batch_size, MatrixXld::Zero(T_dec, D));
			Hidden_states.resize(batch_size, MatrixXld::Zero(T_dec, H));
			Cell_states.resize(batch_size, MatrixXld::Zero(T_dec, H));

			// Собираем объединённые веса вне цикла
			MatrixXld W_x(Input_size, 4 * H);
			W_x << W_F_I, W_I_I, W_C_I, W_O_I;
			MatrixXld W_h(H, 4 * H);
			W_h << W_F_H, W_I_H, W_C_H, W_O_H;
			RowVectorXld b(4 * H);
			b << B_F, B_I, B_C, B_O;

			// По батчам
			for (size_t n = 0; n < batch_size; ++n) {
				RowVectorXld h_prev = RowVectorXld::Zero(H);
				RowVectorXld c_prev = RowVectorXld::Zero(H);

				for (size_t t = 0; t < T_dec; ++t) {
					// 1) Attention
					const MatrixXld& H_enc = encoder_outputs[n];
					RowVectorXld context = attention_->ComputeContext(H_enc, h_prev);
					context_vectors[n].row(t) = context;

					// 2) Собираем вход декодера: y_prev||context
					RowVectorXld y_prev = Input_states[n].row(t);
					// L2-нормализация
					long double norm = std::sqrt(y_prev.squaredNorm() + 1e-8L);
					y_prev /= norm;
					RowVectorXld dec_in(D);
					dec_in << y_prev, context;

					// 3) LSTM-шаг
					RowVectorXld Z = dec_in * W_x + h_prev * W_h + b;
					auto f_t = ActivationFunctions::Sigmoid(Z.leftCols(H));
					auto i_t = ActivationFunctions::Sigmoid(Z.middleCols(H, H));
					auto c_bar = ActivationFunctions::Tanh(Z.middleCols(2 * H, H));
					auto o_t = ActivationFunctions::Sigmoid(Z.rightCols(H));
					RowVectorXld c_t = f_t.array() * c_prev.array() + i_t.array() * c_bar.array();
					RowVectorXld h_t = o_t.array() * ActivationFunctions::Tanh(c_t).array();
					Hidden_states[n].row(t) = h_t;
					Cell_states[n].row(t) = c_t;
					h_prev = h_t;
					c_prev = c_t;

					// 4) Проекции + LayerNorm
					RowVectorXld proj(D);
					proj << h_t, context;
					// LayerNorm
					long double mean = proj.mean();
					long double var = (proj.array() - mean).square().mean();
					proj = ((proj.array() - mean) / std::sqrt(var + 1e-5L)).matrix().array().cwiseProduct(layernorm_gamma.array())
						+ layernorm_beta.array();
					U_state[n].row(t) = proj;

					// 5) Логиты
					logits[n].row(t) = proj * W_output.transpose() + b_output;
				}
			}
		}


	private:
		std::vector<MatrixXld> target_outputs_;  // [B][T_dec x Output_dim]
	};

	Seq2SeqWithAttention_ForTrain( std::unique_ptr<Encoder> encoder_train = std::make_unique<Encoder>(), std::unique_ptr<Decoder> decoder_train = std::make_unique<Decoder>())
		: Seq2SeqWithAttention(std::move(encoder_train), std::move(decoder_train)) {
	}
	struct grads_Seq2SeqWithAttention {
		MatrixXld dW_out; MatrixXld dB_out;

		MatrixXld dW_gamma_layernorm; MatrixXld dB_beta_layernorm;

		MatrixXld dV_a_attention, dW_e_attention, dW_d_attention;

		MatrixXld dW_f_dec, dU_f_dec; MatrixXld dB_f_dec;
		MatrixXld dW_i_dec, dU_i_dec; MatrixXld dB_i_dec;
		MatrixXld dW_c_dec, dU_c_dec; MatrixXld dB_c_dec;
		MatrixXld dW_o_dec, dU_o_dec; MatrixXld dB_o_dec;

		MatrixXld dW_f_forw_enc, dU_f_forw_enc; MatrixXld dB_f_forw_enc;
		MatrixXld dW_i_forw_enc, dU_i_forw_enc; MatrixXld dB_i_forw_enc;
		MatrixXld dW_c_forw_enc, dU_c_forw_enc; MatrixXld dB_c_forw_enc;
		MatrixXld dW_o_forw_enc, dU_o_forw_enc; MatrixXld dB_o_forw_enc;

		MatrixXld dW_f_back_enc, dU_f_back_enc; MatrixXld dB_f_back_enc;
		MatrixXld dW_i_back_enc, dU_i_back_enc; MatrixXld dB_i_back_enc;
		MatrixXld dW_c_back_enc, dU_c_back_enc; MatrixXld dB_c_back_enc;
		MatrixXld dW_o_back_enc, dU_o_back_enc; MatrixXld dB_o_back_enc;

		MatrixXld && get_dB_out(const MatrixXld & Y, const MatrixXld& Y_true) {
			return std::move(Y_true - Y);
		}
		MatrixXld&& get_dW_out(const MatrixXld & dB_out, const MatrixXld & U_state) {
			return std::move(dB_out.transpose() * U_state);
		}
		MatrixXld&& get_dW_out(const MatrixXld& dB_out, const MatrixXld& U_state) {
			return std::move(dB_out.transpose() * U_state);
		}
	};
	grads_Seq2SeqWithAttention Backward() {
		grads_Seq2SeqWithAttention grads;
		{
			Eigen::Index E = this->decoder_->output_size;
			Eigen::Index H = this->decoder_->Hidden_size;
			Eigen::Index C = this->decoder_->Input_size - E;
			Eigen::Index D = H + C;
			Eigen::Index A = this->decoder_->attention_->attention_size_;
			Eigen::Index X = E + C;
			Eigen::Index HE = this->encoder_->Common_Hidden_size;
			Eigen::Index EE = this->encoder_->Common_Input_size;
			grads.dW_out.conservativeResize(E, D), grads.dB_out.conservativeResize(1, E);
		
			grads.dW_gamma_layernorm.conservativeResize(1, D), grads.dB_beta_layernorm.conservativeResize(1,  D);
		
			grads.dV_a_attention.conservativeResize(A, 1), grads.dW_e_attention.conservativeResize(A, C), grads.dW_d_attention.conservativeResize(A, H);
		
			grads.dW_f_dec.conservativeResize(H, X), grads.dU_f_dec.conservativeResize(H, H), grads.dB_f_dec.conservativeResize(1, H),
			grads.dW_i_dec.conservativeResize(H, X), grads.dU_i_dec.conservativeResize(H, H), grads.dB_i_dec.conservativeResize(1, H),
			grads.dW_c_dec.conservativeResize(H, X), grads.dU_c_dec.conservativeResize(H, H), grads.dB_c_dec.conservativeResize(1, H),
			grads.dW_o_dec.conservativeResize(H, X), grads.dU_o_dec.conservativeResize(H, H), grads.dB_o_dec.conservativeResize(1, H);
		
			grads.dW_f_forw_enc.conservativeResize(HE, EE), grads.dU_f_forw_enc.conservativeResize(HE, HE), grads.dB_f_forw_enc.conservativeResize(1, HE),
			grads.dW_i_forw_enc.conservativeResize(HE, EE), grads.dU_i_forw_enc.conservativeResize(HE, HE), grads.dB_i_forw_enc.conservativeResize(1, HE),
			grads.dW_c_forw_enc.conservativeResize(HE, EE), grads.dU_c_forw_enc.conservativeResize(HE, HE), grads.dB_c_forw_enc.conservativeResize(1, HE),
			grads.dW_o_forw_enc.conservativeResize(HE, EE), grads.dU_o_forw_enc.conservativeResize(HE, HE), grads.dB_o_forw_enc.conservativeResize(1, HE);
									
			grads.dW_f_back_enc.conservativeResize(HE, EE), grads.dU_f_back_enc.conservativeResize(HE, HE), grads.dB_f_back_enc.conservativeResize(1, HE),
			grads.dW_i_back_enc.conservativeResize(HE, EE), grads.dU_i_back_enc.conservativeResize(HE, HE), grads.dB_i_back_enc.conservativeResize(1, HE),
			grads.dW_c_back_enc.conservativeResize(HE, EE), grads.dU_c_back_enc.conservativeResize(HE, HE), grads.dB_c_back_enc.conservativeResize(1, HE),
			grads.dW_o_back_enc.conservativeResize(HE, EE), grads.dU_o_back_enc.conservativeResize(HE, HE), grads.dB_o_back_enc.conservativeResize(1, HE);
		}
		Eigen::Index T = this->GetDecoderOutputs()[0].cols();

	}
};


int main() {
	setlocale(LC_ALL, "Russian");

	return 0;
}