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
	Matrix(std::initializer_list<std::initializer_list<long double>> il) {
		if (il.size() == 0) {
			rows = 0;
			cols = 0;
			return;
		}

		rows = il.size();
		cols = il.begin()->size();

		// Проверка одинаковой длины строк
		for (const auto& row : il) {
			if (row.size() != cols) {
				throw std::invalid_argument("All rows must have the same length");
			}
		}

		// Заполнение данных матрицы
		data.reserve(rows);
		for (const auto& row_il : il) {
			data.emplace_back(row_il.begin(), row_il.end());
		}
	}

	Matrix sum_rows() const {
		Matrix result(1, cols);
		for (size_t j = 0; j < cols; ++j) {
			long double sum = 0.0L;
			for (size_t i = 0; i < rows; ++i) {
				sum += data[i][j];
			}
			result(0, j) = sum;
		}
		return result;
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
	void pushback(const std::vector<long double>& vec_) {
		if (cols != 0 && vec_.size() != cols)
			throw std::invalid_argument("Row size mismatch");
		data.push_back(vec_);
		rows++;
	}
	void operator=(const Matrix other) {
		this->data = other.data;
		this->cols = other.cols;
		this->rows = other.rows;
	}
	void operator=(const std::vector<std::vector<long double>> other) {
		this->data = other;
		this->cols = data[0].size();
		this->rows = other.size();
	}
	// Сериализация в текстовый файл
	void save_to_text(const std::string& filename) const {
		std::ofstream file(filename, std::ios::trunc); // Перезапись файла
		if (!file) throw std::runtime_error("Cannot open file for writing");

		file << rows << " " << cols << "\n";
		for (const auto& row : data) {
			for (long double val : row) {
				file << val << " ";
			}
			file << "\n";
		}
	}

	// Десериализация из текстового файла
	void load_from_text(const std::string& filename) {
		std::ifstream file(filename);
		if (!file) throw std::runtime_error("Cannot open file for reading");

		file >> rows >> cols;
		data.resize(rows, std::vector<long double>(cols));

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				if (!(file >> data[i][j])) {
					throw std::runtime_error("Error reading matrix data");
				}
			}
		}
	}

	static Matrix zeros(size_t rows, size_t cols) {
		return Matrix(rows, cols);
	}

	// Добавить метод добавления строки
	Matrix append_row(const Matrix& row) const {
		if (row.getRows() != 1 || (getCols() != 0 && row.getCols() != getCols())) {
			throw std::invalid_argument("Invalid row dimensions");
		}
		Matrix new_matrix = *this;
		if (row.getRows() == 1) {
			new_matrix.data.push_back(row.data[0]);
		}
		else {
			throw std::invalid_argument("Row must have exactly 1 row");
		}
		new_matrix.rows++;
		if (new_matrix.cols == 0) new_matrix.cols = row.getCols();
		return new_matrix;
	}
	void clear() {
		data.clear();
		rows = 0;
		cols = 0;
	}
};

/**/


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

class SimpleLSTM {
public:
	
	SimpleLSTM(size_t Number_states = 1, size_t lenght_states = 1, size_t Hidden_size_ = 10){
		if (Hidden_size_ == 0){
			throw std::invalid_argument("Размеры слоев должны быть больше 0");
		}

		this->Input_size = lenght_states;
		this->Hidden_size = Hidden_size_;
		// Инициализация весов
		SetRandomWeights(-0.5L, 0.5L); // Инициализация весов LSTM

		// Инициализация смещений (1xHidden_size)
		SetRandomDisplacements(-1.5L, 1.5L);

		// Инициализация состояний
		Input_states = Matrix(0, this->Input_size); // Пустая матрица
		Cell_states = Matrix::zeros(0, Hidden_size_); // Пустая матрица
		Hidden_states = Matrix::zeros(0, Hidden_size_);
	}

	SimpleLSTM() = default;

	~SimpleLSTM() {
		save("LSTM_state.txt");
	}

	void SetInput_states(const Matrix& Input_states_) {
		if (Input_states_.getRows() == 0 || Input_states_.getCols() != Input_size) {
			throw std::invalid_argument("Invalid input matrix dimensions");
		}

		Cell_states.clear();
		Hidden_states.clear();
		FG_states.clear();
		IG_states.clear();
		CT_states.clear();
		OG_states.clear();
		Input_states = Input_states_;
		// Инициализируем пустые матрицы для состояний
		Cell_states = Matrix(0, Hidden_size);
		Hidden_states = Matrix(0, Hidden_size);
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

	void SetDisplacements(const Matrix& displacements_FG, const Matrix& displacements_IG, const Matrix& displacements_CT, const Matrix& displacements_OG){
		auto check_displacement = [&](const Matrix& m) {
			return m.getRows() == 1 && m.getCols() == Hidden_size;
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

	void SetRandomWeights(long double a = -0.5L, long double b = 0.5L) {
		this->Weights_for_FG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		this->Weights_for_IG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		this->Weights_for_CT_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
		this->Weights_for_OG_HS = ActivationFunctions::matrix_random(Hidden_size, Hidden_size, a, b);
 
		this->Weights_for_FG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
		this->Weights_for_IG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
		this->Weights_for_CT_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
		this->Weights_for_OG_IS = ActivationFunctions::matrix_random(Hidden_size, Input_size, a, b);
	}

	void SetRandomDisplacements(long double a = -1.5L, long double b = 1.5L) {
		this->Displacements_for_FG = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->Displacements_for_IG = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->Displacements_for_CT = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
		this->Displacements_for_OG = ActivationFunctions::matrix_random(1, Hidden_size, a, b);
	}

	bool CalculationAll_states(const Matrix& targets, size_t max_steps, long double precision = 0.1) {
		for (size_t t = 0; t < max_steps; ++t) {
			n_state_Сalculation(t);
			Matrix predictions = GetOutput_states();
			Matrix error = predictions - targets;

			// Проверка ошибки
			bool all_match = true;
			for (size_t i = 0; i < error.getRows(); ++i) {
				for (size_t j = 0; j < error.getCols(); ++j) {
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
			if (std::abs(a(a.getRows() - 1, 0) - limit) <= precision) {
				break;
			}
		}
	}

	Matrix GetOutput_states() const {
		return this->Hidden_states;
	}

	std::vector<Matrix> GetWeightsAndDisplacement() {
		return {
			this->Weights_for_FG_HS, this->Weights_for_FG_IS, this->Displacements_for_FG,
			this->Weights_for_IG_HS, this->Weights_for_IG_IS, this->Displacements_for_IG,
			this->Weights_for_CT_HS, this->Weights_for_CT_IS, this->Displacements_for_CT,
			this->Weights_for_OG_HS, this->Weights_for_OG_IS, this->Displacements_for_OG
		};
	}

	void Train(const Matrix& inputs, const Matrix& targets, size_t epochs, long double learning_rate) {
		for (size_t epoch = 0; epoch < epochs; ++epoch) {
			SetInput_states(inputs);
			// Прямой проход
			for (size_t t = 0; t < inputs.getRows(); ++t) {
				n_state_Сalculation(t);
			}

			// Обратный проход и обновление весов
			Matrix error = this->Hidden_states - targets;
			LSTMGradients grads = Backward(error);
			UpdateWeights(grads, learning_rate);
		}
	}

	void vector_Train(std::vector<Matrix> vec_, size_t epochs, long double learning_rate, long double precision = 0.001) {
		for(size_t epoch_ = 0; epoch_ < vec_.size() / 2; ++epoch_){
			auto inputs = vec_[2*epoch_];
			auto targets = vec_[2 * epoch_ + 1];
			SetInput_states(inputs);
			for (size_t epoch = 0; epoch < epochs; ++epoch) {
				if (CalculationAll_states(targets, targets.getRows(), precision)) {
					break;
				}
				Matrix predictions = GetOutput_states();
				Matrix error = predictions - targets;
				//std::cout << targets(1, 0) << std::endl;
				//std::cout << predictions(1, 0) << std::endl;
				// Отладочный вывод
				if (epoch % 100 == 0) {
					for (size_t i = 0; i < predictions.getRows(); i++) {
						for (size_t j = 0; j < predictions.getCols(); j++) {
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

	static Matrix normalize(const std::vector<std::vector<char>>& c) {
		if (c.empty()) {
			return Matrix(); // Возвращаем пустую матрицу
		}

		const size_t rows = c.size();
		const size_t cols = c[0].size();

		// Проверка согласованности размеров
		for (const auto& row : c) {
			if (row.size() != cols) {
				throw std::invalid_argument("All rows must have the same length");
			}
		}

		Matrix result(rows, cols);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				// Преобразование char -> long double и нормализация
				const long double value = static_cast<long double>(c[i][j]);
				result(i, j) = value / 127.5L - 1.0L;
			}
		}

		return result;
	}

	static long double normalize(char c) {
		return static_cast<long double>(c) / 127.5L - 1.0L;
	}

	static char denormalize(long double val) {
		return static_cast<char>((val + 1.0L) * 127.5L);
	}

	static std::vector<std::vector<char>> denormalize(const Matrix& val) {
		if (val.getRows() == 0 || val.getCols() == 0) {
			return {};
		}

		const size_t rows = val.getRows();
		const size_t cols = val.getCols();
		std::vector<std::vector<char>> result(rows, std::vector<char>(cols));

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				// Выполняем обратное преобразование с контролем диапазона
				const long double normalized_value = val(i, j);
				long double denorm_value = (normalized_value + 1.0L) * 127.5L;

				// Ограничиваем значение в диапазоне [CHAR_MIN, CHAR_MAX]
				denorm_value = std::max<long double>(
					denorm_value,
					static_cast<long double>(std::numeric_limits<char>::min())
				);
				denorm_value = std::min<long double>(
					denorm_value,
					static_cast<long double>(std::numeric_limits<char>::max())
				);

				result[i][j] = static_cast<char>(std::round(denorm_value));
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
	Matrix Cell_states;

	Matrix Weights_for_FG_HS;  // Forget gate hidden state weights
	Matrix Weights_for_IG_HS;  // Input gate hidden state weights
	Matrix Weights_for_CT_HS;  // Cell state hidden state weights
	Matrix Weights_for_OG_HS;  // Output gate hidden state weights

	Matrix Weights_for_FG_IS;  // Forget gate input weights
	Matrix Weights_for_IG_IS;  // Input gate input weights
	Matrix Weights_for_CT_IS;  // Cell state input weights
	Matrix Weights_for_OG_IS;  // Output gate input weights

	Matrix Displacements_for_FG;  // Матрица 1xHidden_size
	Matrix Displacements_for_IG;  // Матрица 1xHidden_size
	Matrix Displacements_for_CT;  // Матрица 1xHidden_size
	Matrix Displacements_for_OG;  // Матрица 1xHidden_size

	std::vector<Matrix> FG_states;
	std::vector<Matrix> IG_states;
	std::vector<Matrix> CT_states;
	std::vector<Matrix> OG_states;

	std::vector<Matrix> StepСalculation(const Matrix& Hidden_State, const Matrix& Last_State, const Matrix& Input_State){
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
			Displacements_for_FG
		);

		// Input Gate (1xHidden_size)
		Matrix input_gate = ActivationFunctions::Sigmoid(
			(Weights_for_IG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_IG_IS * Input_State.transpose()).transpose() +
			Displacements_for_IG
		);

		// Cell State Candidate (1xHidden_size)
		Matrix ct_candidate = ActivationFunctions::Tanh(
			(Weights_for_CT_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_CT_IS * Input_State.transpose()).transpose() +
			Displacements_for_CT
		);

		// New Cell State (1xHidden_size)
		Matrix new_cell_state = forget_gate % Last_State + input_gate % ct_candidate;

		// Output Gate (1xHidden_size)
		Matrix output_gate = ActivationFunctions::Sigmoid(
			(Weights_for_OG_HS * Hidden_State.transpose()).transpose() +
			(Weights_for_OG_IS * Input_State.transpose()).transpose() +
			Displacements_for_OG
		);

		// New Hidden State (1xHidden_size)
		Matrix new_hidden_state = output_gate % ActivationFunctions::Tanh(new_cell_state);

		return { new_cell_state, new_hidden_state, forget_gate, input_gate, ct_candidate, output_gate};
	}

	void n_state_Сalculation(size_t timestep) {
		if (timestep >= Input_states.getRows()) {
			throw std::out_of_range("Invalid timestep");
		}
		Matrix input = Input_states.get_row(timestep);

		// Получаем предыдущие состояния
		Matrix prev_hidden = (timestep == 0) ?
			Matrix::zeros(1, Hidden_size) : Hidden_states.get_row(timestep - 1);

		Matrix prev_cell = (timestep == 0) ?
			Matrix::zeros(1, Hidden_size) : Cell_states.get_row(timestep - 1);

		auto results = StepСalculation(prev_hidden, prev_cell, input);

		// Добавляем новые состояния как новые строки
		if (timestep == 0) {
			Cell_states = results[0];
			Hidden_states = results[1];
		}
		else {
			Cell_states = Cell_states.append_row(results[0]);
			Hidden_states = Hidden_states.append_row(results[1]);
		}

		// Сохраняем гейты
		FG_states.push_back(results[2]);
		IG_states.push_back(results[3]);
		CT_states.push_back(results[4]);
		OG_states.push_back(results[5]);
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

		for (int t = static_cast<int>(T) - 1; t >= 0; --t) {
			if (t >= FG_states.size() || t >= IG_states.size() /*...*/) {
				throw std::runtime_error("Invalid gate states index");
			}
			// Получаем сохранённые значения для шага t
			Matrix f_t = FG_states[t];
			Matrix i_t = IG_states[t];
			Matrix TC_t = CT_states[t];
			Matrix o_t = OG_states[t];
			Matrix C_t = Cell_states.get_row(t);
			Matrix C_prev; 

			if (t == 0) {
				C_prev = Matrix::zeros(1, Hidden_size);
			}
			else {
				if (Cell_states.getRows() <= t - 1) {
					throw std::runtime_error("Invalid cell state index");
				}
				C_prev = Cell_states.get_row(t - 1);
			}

			Matrix h_prev = (t == 0) ? Matrix::zeros(1, Hidden_size) : Hidden_states.get_row(t - 1);

			Matrix x_t = Input_states.get_row(t);

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
		grads.dW_fg_hs *= (1.0 / T);
		grads.dW_fg_is *= (1.0 / T);

		grads.dW_ig_hs *= (1.0 / T);
		grads.dW_ig_is *= (1.0 / T);

		grads.dW_ct_hs *= (1.0 / T);
		grads.dW_ct_is *= (1.0 / T);

		grads.dW_og_hs *= (1.0 / T);
		grads.dW_og_is *= (1.0 / T);

		grads.db_fg = grads.db_fg.sum_rows() * (1.0 / T);
		grads.db_ig = grads.db_ig.sum_rows() * (1.0 / T);
		grads.db_ct = grads.db_ct.sum_rows() * (1.0 / T);
		grads.db_og = grads.db_og.sum_rows() * (1.0 / T);

		return grads;
	}

	void UpdateWeights(const LSTMGradients& gradients, long double learning_rate) {

		auto check_nan = [](const Matrix& m, const std::string& name) {
			for (size_t i = 0; i < m.getRows(); ++i) {
				for (size_t j = 0; j < m.getCols(); ++j) {
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

		Weights_for_FG_HS -= gradients.dW_fg_hs * learning_rate;
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
		Displacements_for_OG -= gradients.db_og * learning_rate;
	}

	void save_matrix(std::ofstream& file, const Matrix& m) const {
		file << m.getRows() << " " << m.getCols() << "\n";
		for (size_t i = 0; i < m.getRows(); ++i) {
			for (size_t j = 0; j < m.getCols(); ++j) {
				file << m[i][j] << " ";
			}
			file << "\n";
		}
	}

	void load_matrix(std::ifstream& file, Matrix& m) {
		size_t rows, cols;
		file >> rows >> cols;
		m = Matrix(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				file >> m[i][j];
			}
		}
	}

	void save_vector(std::ofstream& file, const std::vector<Matrix>& vec) const {
		file << vec.size() << "\n";
		for (const auto& m : vec) {
			save_matrix(file, m);
		}
	}

	void load_vector(std::ifstream& file, std::vector<Matrix>& vec) {
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
	
	/*SimpleLSTM lstm(2, 1, 64);
	lstm.load("LSTM_state.txt");
	// Генерируем входные данные размером 10x5
	Matrix inputs1(2, 1);
	inputs1.set_row(0, { { SimpleLSTM::normalize('М') } });
	inputs1.set_row(1, { {SimpleLSTM::normalize('Д')} });

	Matrix inputs2(2, 1);
	inputs2.set_row(0, { {SimpleLSTM::normalize('м')} });
	inputs2.set_row(1, { {SimpleLSTM::normalize('д')} });

	Matrix inputs3(2, 1);
	inputs3.set_row(0, { {SimpleLSTM::normalize('М')} });
	inputs3.set_row(1, { {SimpleLSTM::normalize('д')} });

	Matrix inputs4(2, 1);
	inputs4.set_row(0, { {SimpleLSTM::normalize('Д')} });
	inputs4.set_row(1, { {SimpleLSTM::normalize('м')} });

	Matrix inputs5(2, 1);
	inputs5.set_row(0, { {SimpleLSTM::normalize('Д')} });
	inputs5.set_row(1, { {SimpleLSTM::normalize('М')} });

	Matrix targets1(2, 1);
	targets1.set_row(0, { {SimpleLSTM::normalize('М')} });
	targets1.set_row(1, { {0.0} });

	for (size_t i = 0; i < std::numeric_limits<size_t>::max(); i ++) {
		lstm.vector_Train({ inputs1, targets1, inputs2, targets1, inputs3, targets1, inputs4, targets1, inputs5, targets1 }, 100000000, 0.001, 0.00000001L);


		lstm.CalculationAll_states(targets1, 2);

		std::cout << lstm.GetOutput_states() << std::endl;
		lstm.save("LSTM_state.txt");
	}

	return 0;*/
}