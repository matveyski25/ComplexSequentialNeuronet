#include "HeaderLib_ComplexSequentialNeuronet.h"
 
using MatrixXld = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXld = Eigen::Matrix<long double, 1, Eigen::Dynamic>; // Вектор-строка
using VectorXld = Eigen::Matrix<long double, Eigen::Dynamic, 1>;    // Вектор-столбец

class SimpleLSTM {
	friend class BiLSTM;
public:
	
	SimpleLSTM(Eigen::Index Number_states, Eigen::Index Hidden_size_){
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

	SimpleLSTM() = default;

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
		all_tanh_outputs_.clear();

	}

	// Получение attention-весов по всем временным шагам
	const std::vector<VectorXld>& GetAllAttentionWeights() const { return all_attention_weights_; }

	// Получение сырых score-векторов (до softmax)
	const std::vector<VectorXld>& GetAllScores() const { return all_scores_; }

protected:
	// Вспомогательные буферы для накопления истории attention по всем шагам
	std::vector<VectorXld> all_attention_weights_;  // α_t для всех t
	std::vector<VectorXld> all_scores_;             // e_{t,i} для всех t
	std::vector<std::vector<RowVectorXld>> all_tanh_outputs_;  // u_{ti} для всех t, i

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
			VectorXld attention_weights = scores.array().exp();
			long double sum_exp = attention_weights.sum() + 1e-8L;
			attention_weights /= sum_exp;

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
		Decoder(std::unique_ptr</*Attention*/BahdanauAttention> attention_module,
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
		Decoder() = default;
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

		RowVectorXld start_token;   // эмбеддинг стартового токена (1 символ)
		MatrixXld end_token;     // матрица эмбеддингов финишного токена (несколько символов)
		size_t max_steps;    // ограничение на число шагов генерации

		std::shared_ptr</*Attention*/BahdanauAttention> attention_;

		std::vector<MatrixXld> encoder_outputs;
		//std::vector<MatrixXld> context_vectors;

		//std::vector<MatrixXld> U_state;

		std::vector<MatrixXld> Output_state;
		// --- Обновляемый выходной слой ---
		MatrixXld W_output;      // [output_size x (hidden_size + context_size)]
		RowVectorXld b_output;   // [1 x output_size]

		size_t output_size;
		//size_t embedding_dim;

		RowVectorXld layernorm_gamma; // [1 x Input_size]
		RowVectorXld layernorm_beta;  // [1 x Input_size]

	private:
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
			//context_vectors.clear();
			//U_state.clear();
			attention_->ClearCache();

			size_t batch_size = encoder_outputs.size();
			Output_state.resize(batch_size);
			//context_vectors.resize(batch_size);
			//U_state.resize(batch_size);

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
				//std::vector<RowVectorXld> context_sequence;
				//std::vector<RowVectorXld> u_sequence;

				RowVectorXld y_prev = start_token;
				RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
				RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

				for (size_t t = 0; t < max_steps; ++t) {
					RowVectorXld context = attention_->ComputeContext(enc_out, h_prev);
					//context_sequence.push_back(context);

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
					//u_sequence.push_back(proj_input);
					y_sequence.push_back(y_t);

					if (IsEndToken(y_t)) {
						size_t end_len = static_cast<size_t>(end_token.rows());
						if (y_sequence.size() >= end_len - 1) {
							y_sequence.resize(y_sequence.size() - (end_len - 1));
							//context_sequence.resize(context_sequence.size() - (end_len - 1));
							//u_sequence.resize(u_sequence.size() - (end_len - 1));
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
				//U_state[n] = MatrixXld(T, u_sequence[0].cols());
				//context_vectors[n] = MatrixXld(T, context_sequence[0].cols());

				for (Eigen::Index t = 0; t < T; ++t) {
					Output_state[n].row(t) = y_sequence[t];
					//this->states[n].row(t) = u_sequence[t];
					//context_vectors[n].row(t) = context_sequence[t];
				}
			}
		}
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
	std::vector<MatrixXld> Input_States;
private:
	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;
};


// ==== ТРЕНИРОВОЧНАЯ Seq2Seq + Attention ====
class Seq2SeqWithAttention_ForTrain : public Seq2SeqWithAttention {
public:
	class SimpleLSTM_ForTrain;
	class BiLSTM_ForTrain;

	class SimpleLSTM_ForTrain : public SimpleLSTM {
		friend class BiLSTM_ForTrain;
	public:
		SimpleLSTM_ForTrain(size_t Batch_size_, Eigen::Index Number_states, Eigen::Index Hidden_size_) {
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
private:

	struct states_forgrads {
		std::vector<MatrixXld> f, i, ccond, o, c, h;
	};

	states_forgrads statesForgrads;

	};

	class BiLSTM_ForTrain : public BiLSTM {
		friend class Seq2SeqWithAttention_ForTrain;
	public:
		BiLSTM_ForTrain(size_t Batch_size_, Eigen::Index Number_states, Eigen::Index Hidden_size_)
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

		BiLSTM_ForTrain() = default;

		~BiLSTM_ForTrain() {
			this->Save("BiLSTM_state_ForTrain.txt");
		}

		void Batch_All_state_Сalculation() {
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
		using BiLSTM_ForTrain::Common_Input_states;
		using BiLSTM_ForTrain::Common_Hidden_size;
		using BiLSTM_ForTrain::Common_Input_size;

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

		std::vector<Eigen::Index> Get_LenghtsInputStates() {
			std::vector<Eigen::Index> N;
			for (auto& inp : this->Common_Input_states) {
				N.push_back(inp.cols());
			}
		}
	};

	class Decoder : public Seq2SeqWithAttention::Decoder, public SimpleLSTM_ForTrain{
		struct states_forgrads {
			std::vector<MatrixXld> f, i, o, ccond, c, h, context, z, x, p, p_;
		};
		states_forgrads StatesForgrads;

		friend class Seq2SeqWithAttention_ForTrain;

		using SimpleLSTM_ForTrain::Input_states;
		using SimpleLSTM_ForTrain::Input_size;
		using SimpleLSTM_ForTrain::Hidden_size;

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
		
		/*void Batch_All_state_Сalculation(
			const std::vector<MatrixXld>& encoder_outputs,
			const std::vector<MatrixXld>& teacher_inputs,        // [B][T_dec x emb_dim]
			const std::vector<std::vector<bool>>& loss_mask,     // [B][T_dec]
			long double teacher_forcing_ratio)
		{
		}*/

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
			this->StatesForgrads.context.clear();
			this->StatesForgrads.x.clear();
			this->StatesForgrads.p.clear();
			this->StatesForgrads.p_.clear();
			this->StatesForgrads.z.clear();
			this->StatesForgrads.f.clear();
			this->StatesForgrads.i.clear();
			this->StatesForgrads.o.clear();
			this->StatesForgrads.ccond.clear(); 
			this->StatesForgrads.c.clear();
			this->StatesForgrads.h.clear();
			attention_->ClearCache();

			size_t batch_size = encoder_outputs.size();
			Output_state.resize(batch_size);
			this->StatesForgrads.context.resize(batch_size);
			this->StatesForgrads.x.resize(batch_size);
			this->StatesForgrads.p.resize(batch_size);
			this->StatesForgrads.p_.resize(batch_size);
			this->StatesForgrads.z.resize(batch_size);
			this->StatesForgrads.f.resize(batch_size);
			this->StatesForgrads.i.resize(batch_size);
			this->StatesForgrads.o.resize(batch_size);
			this->StatesForgrads.ccond.resize(batch_size);
			this->StatesForgrads.c.resize(batch_size);
			this->StatesForgrads.h.resize(batch_size);

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

				RowVectorXld y_prev = start_token;
				RowVectorXld h_prev = RowVectorXld::Zero(Hidden_size);
				RowVectorXld c_prev = RowVectorXld::Zero(Hidden_size);

				for (size_t t = 0; t < max_steps; ++t) {
					RowVectorXld context = attention_->ComputeContext(enc_out, h_prev);

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

					RowVectorXld proj_input_(Hidden_size + context.size());
					proj_input_ << h_t, context;
					auto proj_input = apply_layernorm(proj_input_);

					RowVectorXld y_t = proj_input * W_output.transpose() + b_output;
					y_sequence.push_back(y_t);

					this->StatesForgrads.f[n].row(t) = f_t;
					this->StatesForgrads.i[n].row(t) = i_t;
					this->StatesForgrads.ccond[n].row(t) = c_bar;
					this->StatesForgrads.o[n].row(t) = o_t;
					this->StatesForgrads.c[n].row(t) = c_t;
					this->StatesForgrads.h[n].row(t) = h_t;

					this->StatesForgrads.context[n].row(t) = context;
					this->StatesForgrads.x[n].row(t) = decoder_input;
					this->StatesForgrads.p[n].row(t) = proj_input_;
					this->StatesForgrads.p_[n].row(t) = proj_input;
					this->StatesForgrads.z[n].row(t) = Z;

					if (IsEndToken(y_t)) {
						/*size_t end_len = static_cast<size_t>(end_token.rows());
						if (y_sequence.size() >= end_len - 1) {
							y_sequence.resize(y_sequence.size() - (end_len - 1));
							context_sequence.resize(context_sequence.size() - (end_len - 1));
							u_sequence.resize(u_sequence.size() - (end_len - 1));
						}*/
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
				
				for (Eigen::Index t = 0; t < T; ++t) {
					Output_state[n].row(t) = y_sequence[t];
				}
			}
		}
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
	};

	struct states_forgrads {
		MatrixXld f_enc_forw, i_enc_forw, ccond_enc_forw, o_enc_forw, c_enc_forw;
		MatrixXld f_enc_back, i_enc_back, ccond_enc_back, o_enc_back, c_enc_back;
	};

	grads_Seq2SeqWithAttention Backward(size_t Number_InputState, MatrixXld Y_True) {
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

		Eigen::Index T = std::min(this->GetDecoderOutputs()[Number_InputState].cols(), Y_True.cols());
		Eigen::Index N = this->encoder_->Get_LenghtsInputStates()[Number_InputState];

		RowVectorXld _dC_t = RowVectorXld::Zero(this->decoder_->Hidden_size);
		for (Eigen::Index t = T; t >= 0; t--) {
			RowVectorXld dY_t = Y_True.row(t) - this->GetDecoderOutputs()[Number_InputState].row(t); //Y_true_t - Y_t
			MatrixXld DW_out_t = dY_t * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).transpose();
			RowVectorXld dp__t = this->decoder_->W_output.transpose() * dY_t;
			RowVectorXld DB_out_t = std::move(dY_t);

			RowVectorXld dS_t = dp__t.leftCols(this->decoder_->Hidden_size);
			RowVectorXld dContext_t = dp__t.middleCols(this->decoder_->Hidden_size, dp__t.cols() - this->decoder_->Hidden_size);

			RowVectorXld DGamma_t = dp__t.array() * this->decoder_->StatesForgrads.p_[Number_InputState].row(t).array();
			RowVectorXld DBeta_t = dp__t;

			RowVectorXld F_t = this->decoder_->StatesForgrads.f[Number_InputState].row(t);
			RowVectorXld I_t = this->decoder_->StatesForgrads.i[Number_InputState].row(t);
			RowVectorXld Ccond_t = this->decoder_->StatesForgrads.ccond[Number_InputState].row(t);
			RowVectorXld O_t = this->decoder_->StatesForgrads.o[Number_InputState].row(t);
			RowVectorXld C_t = this->decoder_->StatesForgrads.c[Number_InputState].row(t);
			RowVectorXld C_t_l;
			if (t == 0) {
				C_t_l = RowVectorXld::Zero(this->decoder_->StatesForgrads.c[Number_InputState].row(t).cols());
			}
			else {
				C_t_l = this->decoder_->StatesForgrads.c[Number_InputState].row(t - 1);
			}
			
			RowVectorXld dO_t = dS_t.array() * ActivationFunctions::Tanh(C_t).array() * O_t.array() * (MatrixXld::Constant((O_t).size(), 1) - O_t).array();
			RowVectorXld dC_t = dS_t.array() * O_t.array() * 
				(MatrixXld::Constant((C_t * C_t).size(), 1) - ActivationFunctions::Tanh(C_t) * ActivationFunctions::Tanh(C_t)).array() + 
				_dC_t.array() * F_t.array();
			RowVectorXld dCcond_t = dC_t.array() * I_t.array() * (MatrixXld::Constant((Ccond_t * Ccond_t).size(), 1) - Ccond_t * Ccond_t).array();
			RowVectorXld dI_t = dC_t.array() * I_t.array() * Ccond_t.array() * (MatrixXld::Constant((I_t).size(), 1) - I_t).array();
			RowVectorXld dF_t = dC_t.array() * C_t_l.array() * F_t.array() * (MatrixXld::Constant((F_t).size(), 1) - F_t).array();

			RowVectorXld dGates_t(4 * this->decoder_->Hidden_size);
			dGates_t  << dF_t, dI_t, dCcond_t, dO_t;

			MatrixXld DW_dec_t = this->decoder_->StatesForgrads.x[Number_InputState].row(t).transpose() * dGates_t;
			MatrixXld DU_dec_t;
			if (t == 0) {
				DU_dec_t = MatrixXld::Zero(this->decoder_->StatesForgrads.h[Number_InputState].row(t).cols(), 4 * this->decoder_->Hidden_size);
			}
			else {
				DU_dec_t = this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1).transpose() * dGates_t;
			}
			VectorXld DB_dec_t = dGates_t;

			std::vector<MatrixXld> _dH_back;
			RowVectorXld Enc_Forw_dC_t = RowVectorXld::Zero(this->encoder_->Common_Hidden_size);
			for (Eigen::Index j = N - 1; j >= 0; --j) {
				RowVectorXld h_j = this->encoder_->Common_Hidden_states[Number_InputState].row(j);
				RowVectorXld s_t_1 = (t > 0) ? this->decoder_->StatesForgrads.h[Number_InputState].row(t - 1)
					: RowVectorXld::Zero(this->decoder_->Hidden_size);

				long double alpha_j = this->decoder_->attention_->all_attention_weights_[t](j);
				long double dAlpha_j = dContext_t.dot(h_j);

				long double dE_tj = 0.0;
				for (int k = 0; k < N; ++k) {
					long double alpha_k = this->decoder_->attention_->all_attention_weights_[t](k);
					RowVectorXld h_k = this->encoder_->Common_Hidden_states[Number_InputState].row(k);
					long double dAlpha_k = dContext_t.dot(h_k);

					dE_tj += dAlpha_k * alpha_k * ((j == k) - alpha_j);  // ∂α_k / ∂e_j
				}

				RowVectorXld u_tj = this->decoder_->attention_->all_tanh_outputs_[t][j];
				RowVectorXld dU_tj = dE_tj * this->decoder_->attention_->attention_vector_.transpose();  // [1 x A]
				RowVectorXld dPreact_tj = dU_tj.array() * (1.0 - u_tj.array().square());

				MatrixXld DW_att_enc_tj = dPreact_tj.transpose() * h_j;   // [A x 1] * [1 x H_enc]
				MatrixXld DW_att_dec_tj = dPreact_tj.transpose() * s_t_1; // [A x 1] * [1 x H_dec]
				MatrixXld DV_att_tj = u_tj.transpose() * dE_tj;       // [A x 1]

				MatrixXld dH_j = dContext_t * alpha_j + this->decoder_->attention_->W_decoder_.transpose() * dU_tj;
				RowVectorXld dH_forw_j = dH_j.leftCols(this->encoder_->Common_Hidden_size);
				RowVectorXld dH_back_j = dH_j.rightCols(this->encoder_->Common_Hidden_size);

				_dH_back.push_back(dH_back_j);

				
				RowVectorXld Enc_Forw_F_j = this->encoder_->Forward->StatesForgrads.f[Number_InputState].row(j);
				RowVectorXld Enc_Forw_I_j = this->encoder_->Forward->StatesForgrads.i[Number_InputState].row(j);
				RowVectorXld Enc_Forw_Ccond_j = this->encoder_->Forward->StatesForgrads.ccond[Number_InputState].row(j);
				RowVectorXld Enc_Forw_O_j = this->encoder_->Forward->StatesForgrads.o[Number_InputState].row(j);
				RowVectorXld Enc_Forw_C_j = this->encoder_->Forward->StatesForgrads.c[Number_InputState].row(j);
				RowVectorXld Enc_Forw_C_j_l;
				if (j == 0) {
					Enc_Forw_C_j_l = RowVectorXld::Zero(this->encoder_->Forward->StatesForgrads.c[Number_InputState].row(j).cols());
				}
				else {
					Enc_Forw_C_j_l = this->encoder_->Forward->StatesForgrads.c[Number_InputState].row(j - 1);
				}

				RowVectorXld dEnc_Forw_O_j = dH_forw_j.array() * ActivationFunctions::Tanh(Enc_Forw_C_j).array() * Enc_Forw_O_j.array() * (MatrixXld::Constant((Enc_Forw_O_j).size(), 1) - Enc_Forw_O_j).array();
				RowVectorXld dEnc_Forw_C_j = dH_forw_j.array() * Enc_Forw_O_j.array() *
					(MatrixXld::Constant((Enc_Forw_C_j * Enc_Forw_C_j).size(), 1) - ActivationFunctions::Tanh(Enc_Forw_C_j) * ActivationFunctions::Tanh(Enc_Forw_C_j)).array() +
					Enc_Forw__dC_j.array() * Enc_Forw_F_j.array();
				RowVectorXld dEnc_Forw_Ccond_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * (MatrixXld::Constant((Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).size(), 1) - Enc_Forw_Ccond_j * Enc_Forw_Ccond_j).array();
				RowVectorXld dEnc_Forw_I_j = dEnc_Forw_C_j.array() * Enc_Forw_I_j.array() * Enc_Forw_Ccond_j.array() * (MatrixXld::Constant((Enc_Forw_I_j).size(), 1) - Enc_Forw_I_j).array();
				RowVectorXld dEnc_Forw_F_j = dEnc_Forw_C_j.array() * Enc_Forw_C_j_l.array() * Enc_Forw_F_j.array() * (MatrixXld::Constant((Enc_Forw_F_j).size(), 1) - Enc_Forw_F_j).array();

				RowVectorXld dEnc_Forw_Gates_j(4 * this->encoder->Common_Hidden_size);
				dEnc_Forw_Gates_j << dEnc_Forw_F_j, dEnc_Forw_I_j, dEnc_Forw_Ccond_j, dEnc_Forw_O_j;

				MatrixXld DW_Enc_Forw_j = this->encoder_->Common_Input_states.row(t).transpose() * dEnc_Forw_Gates_j;
				MatrixXld DU_Enc_Forw_j;
				if (t == 0) {
					DU_Enc_Forw_j = MatrixXld::Zero(this->encoder_->Forward->StatesForgrads.h[Number_InputState].row(j).cols(), 4 * this->encoder->Common_Hidden_size);
				}
				else {
					DU_Enc_Forw_j = this->encoder_->Forward->StatesForgrads.h[Number_InputState].row(j - 1).transpose() * dEnc_Forw_Gates_j;
				}
				VectorXld DB_Enc_Forw_j = dEnc_Forw_Gates_j;
			}

		}
	}
	std::unique_ptr<Encoder> encoder_;
	std::unique_ptr<Decoder> decoder_;

	std::vector<MatrixXld> Target_outputs;  // [B][T_dec x Output_dim]

};


int main() {
	setlocale(LC_ALL, "Russian");

	return 0;
}