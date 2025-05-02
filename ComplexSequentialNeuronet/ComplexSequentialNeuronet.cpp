#include "HeaderLib_ComplexSequentialNeuronet.h"

class Matrix {
private:
	size_t rows;
	size_t cols;
	std::vector<std::vector<long double>> data;

	class RowProxy {
		std::vector<long double>& row_ref;
		size_t max_col;
	public:
		RowProxy(std::vector<long double>& row, size_t max_cols)
			: row_ref(row), max_col(max_cols) {
		}

		long double& operator[](size_t col) {
			if (col >= max_col)
				throw std::out_of_range("Column index out of range");
			return row_ref[col];
		}
	};

	// Прокси-класс для строки (константный доступ)
	class ConstRowProxy {
		const std::vector<long double>& row_ref;
		size_t max_col;
	public:
		ConstRowProxy(const std::vector<long double>& row, size_t max_cols)
			: row_ref(row), max_col(max_cols) {
		}

		const long double& operator[](size_t col) const {
			if (col >= max_col)
				throw std::out_of_range("Column index out of range");
			return row_ref[col];
		}
	};

public:
	// Конструкторы
	Matrix() : rows(0), cols(0) {}

	Matrix(size_t rows_, size_t cols_){
		this->rows = rows_; 
		this->cols = cols_;
		data.resize(rows, std::vector<long double>(cols, 0.0));
	}

	Matrix(const std::vector<std::vector<long double>>& input) {
		if (input.empty()) {
			rows = 0;
			cols = 0;
			return;
		}

		rows = input.size();
		cols = input[0].size();
		for (const auto& row : input) {
			if (row.size() != cols) {
				throw std::invalid_argument("All rows must have the same length");
			}
		}
		data = input;
	}
	Matrix(const std::vector<long double>& input_) {
		auto input = std::vector<std::vector<long double>>(1, input_);
		if (input.empty()) {
			rows = 0;
			cols = 0;
			return;
		}

		rows = input.size();
		cols = input[0].size();
		for (const auto& row : input) {
			if (row.size() != cols) {
				throw std::invalid_argument("All rows must have the same length");
			}
		}
		data = input;
	}
	Matrix(std::vector<long double>&& input_) {
		auto input = std::vector<std::vector<long double>>(1, input_);
		if (input.empty()) {
			rows = 0;
			cols = 0;
			return;
		}

		rows = input.size();
		cols = input[0].size();
		for (const auto& row : input) {
			if (row.size() != cols) {
				throw std::invalid_argument("All rows must have the same length");
			}
		}
		data = input;
	}


	RowProxy operator[](size_t row) {
		if (row >= rows)
			throw std::out_of_range("Row index out of range");
		return RowProxy(data[row], cols);
	}

	const ConstRowProxy operator[](size_t row) const {
		if (row >= rows)
			throw std::out_of_range("Row index out of range");
		return ConstRowProxy(data[row], cols);
	}

	static Matrix from_scalar(long double val, size_t rows, size_t cols) {
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result(i, j) = val;
			}
		}
		return result;
	}

	// Методы доступа
	size_t getRows() const { return rows; }
	size_t getCols() const { return cols; }

	// Операторы доступа к элементам
	long double& operator()(size_t row, size_t col) {
		if (row >= rows || col >= cols)
		{
			throw std::out_of_range("Matrix indices out of range");
		}
		return data[row][col];
	}

	const long double& operator()(size_t row, size_t col) const {
		if (row >= rows || col >= cols)
		{
			throw std::out_of_range("Matrix indices out of range");
		}
		return data[row][col];
	}

	const Matrix& operator()(size_t row) const {
		if (row >= rows)
		{
			throw std::out_of_range("Matrix indices out of range");
		}
		return { data[row] };
	}
	Matrix& operator()( size_t row) {
		if (row >= rows)
		{
			throw std::out_of_range("Matrix indices out of range");
		}
		auto result = Matrix(data[row]);
		return result;
	}

	// Арифметические операции
	Matrix operator+(const Matrix& other) const {
		if (rows != other.rows || cols != other.cols)
		{
			throw std::invalid_argument("Matrix dimensions must agree");
		}

		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result(i, j) = data[i][j] + other(i, j);
			}
		}
		return result;
	}

	Matrix operator-() const {
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result(i, j) = -data[i][j];
			}
		}
		return result;
	}

	// Оператор вычитания матриц
	Matrix operator-(const Matrix& other) const {
		if (rows != other.rows || cols != other.cols) {
			throw std::invalid_argument("Matrix dimensions must agree");
		}
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result(i, j) = data[i][j] - other(i, j);
			}
		}
		return result;
	}

	// Оператор вычитания с присваиванием
	Matrix& operator-=(const Matrix& other) {
		*this = *this - other;
		return *this;
	}

	Matrix operator*(const Matrix& other) const {
		if (cols != other.rows)
		{
			throw std::invalid_argument("Matrix dimensions must agree");
		}

		Matrix result(rows, other.cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < other.cols; ++j) {
				for (size_t k = 0; k < cols; ++k) {
					result(i, j) += data[i][k] * other(k, j);
				}
			}
		}
		return result;
	}


	Matrix operator*(long double scalar) const {
		Matrix result(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result(i, j) = data[i][j] * scalar;
			}
		}
		return result;
	}

	friend Matrix operator*(double scalar, const Matrix& matrix) {
		return matrix * scalar;
	}

	Matrix operator%(const Matrix & other) {
		if (this->cols != other.cols || this->rows != other.rows) {
			throw std::invalid_argument("Matrix dimensions must agree");
		}
		Matrix result(this->rows, this->cols);
		for (size_t i = 0; i < result.rows; i++) {
			for (size_t j = 0; j < result.cols; j++) {
				result.data[i][j] = this->data[i][j] * other.data[i][j];
			}
		}
		return result;
	}
	size_t size() {
		return this->data.size();
	}
	bool empty() {
		if (size() == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}
	// Составные присваивания
	Matrix& operator+=(const Matrix& other) {
		*this = *this + other;
		return *this;
	}

	Matrix& operator*=(const Matrix& other) {
		*this = *this * other;
		return *this;
	}

	Matrix& operator*=(double scalar) {
		*this = *this * scalar;
		return *this;
	}

	// Транспонирование
	Matrix transpose() const {
		Matrix result(cols, rows);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result(j, i) = data[i][j];
			}
		}
		return result;
	}
	Matrix get_row(size_t row) const {
		//if (row >= rows) throw std::out_of_range("Row index out of range");
		Matrix result(1, cols);
		for (size_t j = 0; j < cols; ++j) {
			result(0, j) = data[row][j];
		}
		return result;
	}

	// Метод для установки строки из другой матрицы
	void set_row(size_t row, const Matrix& source) {
		if (row >= rows) throw std::out_of_range("Row index out of range");
		if (source.getCols() != cols) throw std::invalid_argument("Column count mismatch");
		for (size_t j = 0; j < cols; ++j) {
			data[row][j] = source(0, j);
		}
	}

	// Вывод матрицы
	friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
		for (size_t i = 0; i < matrix.rows; ++i) {
			for (size_t j = 0; j < matrix.cols; ++j) {
				os << matrix(i, j) << "\t";
			}
			os << "\n";
		}
		return os;
	}
	void pushback(std::vector<long double> vec_) {
		data.push_back(vec_);
		this->rows++;
	}
	/*void operator=(const Matrix other) {
		this->data = other.data;
		this->cols = other.cols;
		this->rows = other.rows;
	}*/
	/*void operator=(const std::vector<std::vector<long double>> other) {
		this->data = other;
		this->cols = data[0].size();
		this->rows = other.size();
	}*/
};

namespace ActivationFunctions {
	bool StepFunction(long double value, long double step = 0.0) {
		if (value >= step) {
			return 1;
		}
		else {
			return 0;
		}
	}
	Matrix StepFunction(const Matrix& matx) {
		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				result[i][j] = matx[i][j] >= 0.0L ? 1.0L : 0.0L;
			}
		}
		return result;
	}
	long double Sigmoid(long double value) {
		return 1.0L / (1.0L + std::exp(-value));
	}
	Matrix Sigmoid(const Matrix& matx) {
		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				result[i][j] = 1.0L / (1.0L + std::exp(-matx[i][j]));
			}
		}
		return result;
	}
	long double Tanh(long double value) {
		return std::tanhl(value);
	}
	Matrix Tanh(const Matrix& matx) {
		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				result[i][j] = std::tanhl(matx[i][j]);
			}
		}
		return result;
	}
	long double ReLU(long double value) {
		return std::fmaxl(0, value);
	}
	Matrix ReLU(const Matrix& matx) {
		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				result[i][j] = std::fmaxl(0.0L, matx[i][j]);
			}
		}
		return result;
	}
	long double LeakyReLU(long double value, long double a = 0.001) {
		if (value >= 0) {
			return value;
		}
		else {
			return (a * value);
		}
	}
	Matrix LeakyReLU(const Matrix& matx, const Matrix& a) {
		if (matx.getRows() != a.getRows() || matx.getCols() != a.getCols()) {
			throw std::invalid_argument("Matrix dimensions must match");
		}

		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				result[i][j] = (matx[i][j] >= 0.0L)
					? matx[i][j]
					: a[i][j] * matx[i][j];
			}
		}
		return result;
	}
	Matrix LeakyReLU(const Matrix& matx, long double a = 0.001L) {
		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				result[i][j] = (matx[i][j] >= 0.0L)
					? matx[i][j]
					: a * matx[i][j];
			}
		}
		return result;
	}
	long double Swish(long double value, long double b = 1.0) {
		long double x = value * b;
		// Ограничение для предотвращения переполнения exp(x)
		x = std::max(x, -700.0L); 
		x = std::min(x, 700.0L); 
		return value * (1.0L / (1.0L + std::exp(-x)));
	}
	Matrix Swish(const Matrix& matx, const Matrix& b) {
		if (matx.getRows() != b.getRows() || matx.getCols() != b.getCols()) {
			throw std::invalid_argument("Matrix dimensions must match");
		}

		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				long double x = matx[i][j] * b[i][j];
				x = std::max(x, -700.0L);
				x = std::min(x, 700.0L);
				result[i][j] = matx[i][j] / (1.0L + std::exp(-x));
			}
		}
		return result;
	}
	Matrix Swish(const Matrix& matx, long double b = 1.0L) {
		Matrix result(matx.getRows(), matx.getCols());
		for (size_t i = 0; i < matx.getRows(); ++i) {
			for (size_t j = 0; j < matx.getCols(); ++j) {
				long double x = matx[i][j] * b;
				x = std::max(x, -700.0L);
				x = std::min(x, 700.0L);
				result[i][j] = matx[i][j] / (1.0L + std::exp(-x));
			}
		}
		return result;
	}
	std::vector<long double> Softmax(const std::vector<long double>& values) {
		if (values.empty()) {
			throw std::invalid_argument("Input vector is empty");
		}
		long double max_val = *std::max_element(values.begin(), values.end());
		long double sum = 0.0;
		std::vector<long double> result;
		result.reserve(values.size());
		for (auto v : values) {
			long double exp_val = std::exp(v - max_val);
			sum += exp_val;
			result.push_back(exp_val);
		}
		if (sum == 0) {
			// Вернуть равномерное распределение или бросить исключение
			throw std::runtime_error("Все значения в Softmax нулевые");
		}
		for (auto& val : result) {
			val /= sum;
		}
		return result;
	}
	long double random(long double a = std::numeric_limits<long double>::lowest(), long double b = std::numeric_limits<long double>::max()) {
		if (a > b) {
			std::swap(a, b);
		}
		static std::mt19937_64 gen(std::random_device{}());
		std::uniform_real_distribution<long double> dis(a, b);
		return dis(gen);
	}
	Matrix matrix_random(size_t rows, size_t cols, long double a = 0.0L, long double b = 1.0L) {
		Matrix result(rows, cols);
		static std::mt19937_64 gen(std::random_device{}());
		std::uniform_real_distribution<long double> dist(a, b);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result[i][j] = dist(gen);
			}
		}
		return result;
	}
	Matrix matrix_random( Matrix matrix, long double a = 0.0L, long double b = 1.0L) {
		size_t rows = matrix.getRows();
		size_t cols = matrix.getCols();
		Matrix result(rows, cols);
		static std::mt19937_64 gen(std::random_device{}());
		std::uniform_real_distribution<long double> dist(a, b);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				result[i][j] = dist(gen);
			}
		}
		return result;
	}
}

class SimpleSNT {
public:
	
	SimpleSNT(size_t Number_states = 1, size_t lenght_states = 1, size_t Hidden_size = 10){
		this->Input_size = lenght_states;
		this->Hidden_size = Hidden_size;
		// Инициализация весов
		// Инициализация весов (Hidden_size x Hidden_size)
		Weights_for_FG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size);
		Weights_for_IG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size);
		Weights_for_CT_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size);
		Weights_for_OG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size);

		// Инициализация весов (Hidden_size x Input_size)
		Weights_for_FG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size);
		Weights_for_IG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size);
		Weights_for_CT_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size);
		Weights_for_OG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size);

		// Инициализация смещений (1xHidden_size)
		Displacement_for_FG = ActivationFunctions::matrix_random(1, Hidden_size);
		Displacement_for_IG = ActivationFunctions::matrix_random(1, Hidden_size);
		Displacement_for_CT = ActivationFunctions::matrix_random(1, Hidden_size);
		Displacement_for_OG = ActivationFunctions::matrix_random(1, Hidden_size);

		// Инициализация состояний
		Input_states = Matrix(Number_states, Input_size);
		Output_states = Matrix(Number_states, Hidden_size);
		Hidden_states = Matrix(Number_states, Hidden_size);
	}

	void SetInput_states(const Matrix& Input_states_) {
		if (Input_states_.getRows() == 0 || Input_states_.getCols() != Input_size) {
			throw std::invalid_argument("Invalid input matrix dimensions");
		}

		Input_states = Input_states_;
		Output_states = Matrix(Input_states.getRows(), Hidden_size);  // Изменено с Input_size на Hidden_size
		Hidden_states = Matrix(Input_states.getRows(), Hidden_size);
	}

	void SetWeights(const Matrix& weights_I_F, const Matrix& weights_I_I, const Matrix& weights_I_C, const Matrix& weights_I_O, const Matrix& weights_H_F, const Matrix& weights_H_I, const Matrix& weights_H_C, const Matrix& weights_H_O)
	{
		// Проверка размеров весов
		auto check_weights = [&](const Matrix& W, size_t rows, size_t cols) {
			return W.getRows() == rows && W.getCols() == cols;
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

	void SetDisplacements(const Matrix& displacement_FG, const Matrix& displacement_IG, const Matrix& displacement_CT, const Matrix& displacement_OG){
		auto check_displacement = [&](const Matrix& m) {
			return m.getRows() == 1 && m.getCols() == Hidden_size;
			};

		if (!check_displacement(displacement_FG) ||
			!check_displacement(displacement_IG) ||
			!check_displacement(displacement_CT) ||
			!check_displacement(displacement_OG))
		{
			throw std::invalid_argument("Displacements must be 1xHidden_size matrices");
		}

		Displacement_for_FG = displacement_FG;
		Displacement_for_IG = displacement_IG;
		Displacement_for_CT = displacement_CT;
		Displacement_for_OG = displacement_OG;
	}

	void SetRandomWeights(long double a = -0.5L, long double b = 0.5L) {
		Weights_for_FG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		Weights_for_IG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		Weights_for_CT_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		Weights_for_OG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);

		Weights_for_FG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
		Weights_for_IG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
		Weights_for_CT_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
		Weights_for_OG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
	}

	bool CalculationAll_states( const Matrix& limiter, size_t max_steps = 1000, long double precision = 1e-5){
		if (limiter.getCols() != Hidden_size) {
			throw std::invalid_argument("Limiter columns mismatch hidden size");
		}

		for (size_t t = 0; t < max_steps; ++t) {
			n_state_Сalculation(t);

			// Проверяем все шаги в limiter
			bool all_match = true;
			for (size_t i = 0; i < limiter.getRows(); ++i) {
				Matrix output_row = Output_states.get_row(t - limiter.getRows() + i + 1);
				for (size_t j = 0; j < Hidden_size; ++j) {
					if (std::abs(output_row(0, j) - limiter(i, j)) > precision) {
						all_match = false;
						break;
					}
				}
				if (!all_match) break;
			}

			if (all_match) return 0;
		}
		return 1;
	}

	Matrix GetOutput_states() const {
		return Output_states;
	}

	std::vector<Matrix> GetWeightsAndDisplacement() {
		return {
			this->Weights_for_FG_HS, this->Weights_for_FG_IS, this->Displacement_for_FG,
			this->Weights_for_IG_HS, this->Weights_for_IG_IS, this->Displacement_for_IG,
			this->Weights_for_CT_HS, this->Weights_for_CT_IS, this->Displacement_for_CT,
			this->Weights_for_OG_HS, this->Weights_for_OG_IS, this->Displacement_for_OG
		};
	}

	void Train(const Matrix& inputs, const Matrix& targets, size_t epochs, long double learning_rate, const Matrix& limiter, size_t max_steps = 1000, long double precision = 1e-5) {
		SetInput_states(inputs);
		for (size_t epoch = 0; epoch < epochs; ++epoch) {
			if (CalculationAll_states(limiter, max_steps, precision) == 0) {
				return;
			}
			Matrix predictions = GetOutput_states();
			Matrix error = predictions - targets;
			LSTMGradients grads = Backward(error);
			UpdateWeights(grads, learning_rate);
		}
	}
private:
	struct LSTMGradients {
		// Градиенты для Forget Gate
		Matrix dW_fg_hs;  // по весам hidden-state
		Matrix dW_fg_is;  // по весам input
		Matrix db_fg;     // по смещению

		// Градиенты для Input Gate
		Matrix dW_ig_hs;
		Matrix dW_ig_is;
		Matrix db_ig;

		// Градиенты для Cell State
		Matrix dW_ct_hs;
		Matrix dW_ct_is;
		Matrix db_ct;

		// Градиенты для Output Gate
		Matrix dW_og_hs;
		Matrix dW_og_is;
		Matrix db_og;
	};

	size_t Input_size;
	size_t Hidden_size;
	Matrix Input_states;
	Matrix Hidden_states;
	Matrix Output_states;

	Matrix Weights_for_FG_HS;  // Forget gate hidden state weights
	Matrix Weights_for_IG_HS;  // Input gate hidden state weights
	Matrix Weights_for_CT_HS;  // Cell state hidden state weights
	Matrix Weights_for_OG_HS;  // Output gate hidden state weights

	Matrix Weights_for_FG_IS;  // Forget gate input weights
	Matrix Weights_for_IG_IS;  // Input gate input weights
	Matrix Weights_for_CT_IS;  // Cell state input weights
	Matrix Weights_for_OG_IS;  // Output gate input weights

	Matrix Displacement_for_FG;  // Матрица 1xHidden_size
	Matrix Displacement_for_IG;  // Матрица 1xHidden_size
	Matrix Displacement_for_CT;  // Матрица 1xHidden_size
	Matrix Displacement_for_OG;  // Матрица 1xHidden_size

	std::vector<Matrix> FG_states;
	std::vector<Matrix> IG_states;
	std::vector<Matrix> CT_states;
	std::vector<Matrix> OG_states;

	std::vector<Matrix> StepСalculation(const Matrix& Hidden_State,
		const Matrix& Last_State,
		const Matrix& Input_State)
	{
		// Проверка размерностей
		if (Hidden_State.getCols() != Hidden_size || Hidden_State.getRows() != 1 ||
			Last_State.getCols() != Hidden_size || Last_State.getRows() != 1 ||
			Input_State.getCols() != Input_size || Input_State.getRows() != 1)
		{
			throw std::invalid_argument("Invalid state dimensions in StepCalculation");
		}

		// Forget Gate (1xHidden_size)
		Matrix forget_gate = ActivationFunctions::Sigmoid(
			(Weights_for_FG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_FG_IS * Input_State.transpose()).transpose() +
			Displacement_for_FG
		);

		// Input Gate (1xHidden_size)
		Matrix input_gate = ActivationFunctions::Sigmoid(
			(Weights_for_IG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_IG_IS * Input_State.transpose()).transpose() +
			Displacement_for_IG
		);

		// Cell State Candidate (1xHidden_size)
		Matrix ct_candidate = ActivationFunctions::Tanh(
			(Weights_for_CT_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_CT_IS * Input_State.transpose()).transpose() +
			Displacement_for_CT
		);

		// New Cell State (1xHidden_size)
		Matrix new_cell_state = forget_gate % Last_State + input_gate % ct_candidate;

		// Output Gate (1xHidden_size)
		Matrix output_gate = ActivationFunctions::Sigmoid(
			(Weights_for_OG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_OG_IS * Input_State.transpose()).transpose() +
			Displacement_for_OG
		);

		// New Hidden State (1xHidden_size)
		Matrix new_hidden_state = output_gate % ActivationFunctions::Tanh(new_cell_state);

		return { new_cell_state, new_hidden_state, forget_gate, input_gate, ct_candidate, output_gate};
	}

	void n_state_Сalculation(size_t timestep) {
		// Увеличиваем размеры матриц состояний при необходимости
		while (timestep >= Input_states.getRows()) {
			Input_states.pushback(std::vector<long double>(Input_size, 0.0L));
		}
		while (timestep >= Hidden_states.getRows()) {
			Hidden_states.pushback(std::vector<long double>(Hidden_size, 0.0L));
		}
		while (timestep >= Output_states.getRows()) {
			Output_states.pushback(std::vector<long double>(Hidden_size, 0.0L));
		}

		// Получаем текущий вход
		Matrix input = Input_states.get_row(timestep);

		// Получаем предыдущие состояния
		Matrix hidden = (timestep == 0)
			? Matrix(1, Hidden_size)
			: Hidden_states.get_row(timestep - 1);

		Matrix cell_state = (timestep == 0)
			? Matrix(1, Hidden_size)
			: Output_states.get_row(timestep - 1);

		// Вычисляем новые состояния
		auto results = StepСalculation(hidden, cell_state, input);

		// Сохраняем состояния
		Output_states.set_row(timestep, results[0]);  // Cell state
		Hidden_states.set_row(timestep, results[1]);  // Hidden state

		// Обновляем временные параметры гейтов
		if (timestep >= FG_states.size()) {
			FG_states.resize(timestep + 1);
		}
		FG_states[timestep] = results[2];

		if (timestep >= IG_states.size()) {
			IG_states.resize(timestep + 1);
		}
		IG_states[timestep] = results[3];

		if (timestep >= CT_states.size()) {
			CT_states.resize(timestep + 1);
		}
		CT_states[timestep] = results[4];

		if (timestep >= OG_states.size()) {
			OG_states.resize(timestep + 1);
		}
		OG_states[timestep] = results[5];
	}

	LSTMGradients Backward(const Matrix& error /*error = выходы(Output_states) - ожидаемые значения(просто Matrix)*/) {
		LSTMGradients grads;
		size_t T = Input_states.getRows();  // Количество временных шагов

		// Инициализация градиентов нулями
		grads.dW_fg_hs = Matrix(Hidden_size, Hidden_size);
		grads.dW_fg_is = Matrix(Hidden_size, Input_size);
		grads.db_fg = Matrix(1, Hidden_size);

		grads.dW_ig_hs = Matrix(Hidden_size, Hidden_size);
		grads.dW_ig_is = Matrix(Hidden_size, Input_size);
		grads.db_ig = Matrix(1, Hidden_size);

		grads.dW_ct_hs = Matrix(Hidden_size, Hidden_size);
		grads.dW_ct_is = Matrix(Hidden_size, Input_size);
		grads.db_ct = Matrix(1, Hidden_size);

		grads.dW_og_hs = Matrix(Hidden_size, Hidden_size);
		grads.dW_og_is = Matrix(Hidden_size, Input_size);
		grads.db_og = Matrix(1, Hidden_size);

		Matrix dh_next = Matrix(1, Hidden_size);  // Градиент по h из будущего
		Matrix dC_next = Matrix(1, Hidden_size);  // Градиент по C из будущего

		for (int t = T - 1; t >= 0; t--) {
			// Получаем сохранённые значения для шага t
			Matrix f_t = FG_states[t];
			Matrix i_t = IG_states[t];
			Matrix TC_t = CT_states[t];
			Matrix o_t = OG_states[t];
			Matrix C_t = Output_states.get_row(t);
			Matrix C_prev = (t == 0) ? Matrix(1, Hidden_size) : Output_states.get_row(t - 1);
			Matrix x_t = Input_states.get_row(t);
			Matrix h_prev = (t == 0) ? Matrix(1, Hidden_size) : Hidden_states.get_row(t - 1);

			// Градиент по h_t (из текущего шага + из будущего)
			Matrix dh = error.get_row(t) + dh_next;

			// Градиент по C_t
			Matrix tanh_Ct = ActivationFunctions::Tanh(C_t);
			Matrix onesdC = Matrix::from_scalar(1.0L, 1, Hidden_size);
			Matrix dC = (dh % o_t % (onesdC - tanh_Ct % tanh_Ct)) + dC_next;

			// Градиенты для гейтов
			Matrix onesdf = Matrix::from_scalar(1.0L, 1, Hidden_size);
			Matrix df = dC % C_prev % f_t % (onesdf - f_t);
			Matrix onesdi = Matrix::from_scalar(1.0L, 1, Hidden_size);
			Matrix di = dC % TC_t % i_t % (onesdi - i_t);
			Matrix onesTC = Matrix::from_scalar(1.0L, 1, Hidden_size);
			Matrix dTC = dC % i_t % (onesTC - TC_t % TC_t);
			Matrix onesdo_gate = Matrix::from_scalar(1.0L, 1, Hidden_size);
			Matrix do_gate = dh % tanh_Ct % o_t % (onesdo_gate - o_t);

			// Обновляем градиенты весов
			grads.dW_fg_hs += df.transpose() * h_prev;
			grads.dW_fg_is += df.transpose() * x_t;
			grads.db_fg += df;

			grads.dW_ig_hs += di.transpose() * h_prev;
			grads.dW_ig_is += di.transpose() * x_t;
			grads.db_ig += di;

			grads.dW_ct_hs += dTC.transpose() * h_prev;
			grads.dW_ct_is += dTC.transpose() * x_t;
			grads.db_ct += dTC;

			grads.dW_og_hs += do_gate.transpose() * h_prev;
			grads.dW_og_is += do_gate.transpose() * x_t;
			grads.db_og += do_gate;

			// Обновляем градиенты для предыдущего шага
			dh_next = df * Weights_for_FG_HS + di * Weights_for_IG_HS +
				dTC * Weights_for_CT_HS + do_gate * Weights_for_OG_HS;
			dC_next = dC % f_t;
		}

		return grads;
	}

	void UpdateWeights(const LSTMGradients& gradients, long double learning_rate) {
		Weights_for_FG_HS -= gradients.dW_fg_hs * learning_rate;
		Weights_for_FG_IS -= gradients.dW_fg_is * learning_rate;
		Displacement_for_FG -= gradients.db_fg * learning_rate;

		Weights_for_IG_HS -= gradients.dW_ig_hs * learning_rate;
		Weights_for_IG_IS -= gradients.dW_ig_is * learning_rate;
		Displacement_for_IG -= gradients.db_ig * learning_rate;

		Weights_for_CT_HS -= gradients.dW_ct_hs * learning_rate;
		Weights_for_CT_IS -= gradients.dW_ct_is * learning_rate;
		Displacement_for_CT -= gradients.db_ct * learning_rate;

		Weights_for_OG_HS -= gradients.dW_og_hs * learning_rate;
		Weights_for_OG_IS -= gradients.dW_og_is * learning_rate;
		Displacement_for_OG -= gradients.db_og * learning_rate;
	}

	
};

int main() {
	SimpleSNT lstm(2, 1, 1); //  временных шагов, вход , скрытый слой 

	// Генерируем входные данные размером 10x5
	Matrix inputs(2, 1);
	inputs.set_row(0, { {static_cast<long double>('M')} });
	inputs.set_row(1, { {static_cast<long double>('D')} });

	Matrix targets(2, 1);
	targets.set_row(0, { {(long double)(int)"М"} });
	targets.set_row(1, { {0.0} });
	// Вычисляем состояния
	lstm.CalculationAll_states({{0.0}});
	// Выводим входные данные
	std::cout << "Input states:\n" << inputs << std::endl;
	auto wad = lstm.GetWeightsAndDisplacement();
	std::cout << "WAD states:";
	for (short i = 0; i < wad.size(); i++) {
		std::cout << wad[i] << std::endl;
	}
	// Получаем выходные состояния
	Matrix outputs = lstm.GetOutput_states();
	std::cout << "Output states:\n" << outputs;

	lstm.Train(inputs, targets, 10000, 0.001, {{0.0}}, 2);

	return 0;
}