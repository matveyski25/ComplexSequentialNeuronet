#include "HeaderLib_ComplexSequentialNeuronet.h"

using Matrix = std::vector<std::vector<long double>>;
using Vector = std::vector<long double>;

Matrix matrix_addition(const Matrix& matrix1, const Matrix& matrix2) {
	// Проверка на пустые матрицы
	if (matrix1.empty() || matrix2.empty() || matrix1[0].empty() || matrix2[0].empty()) {
		return Matrix();
	}

	// Проверка совпадения размеров
	const size_t rows = matrix1.size();
	const size_t cols = matrix1[0].size();

	if (rows != matrix2.size() || cols != matrix2[0].size()) {
		return Matrix();
	}

	// Проверка целостности матриц
	for (const auto& row : matrix1) {
		if (row.size() != cols) {
			return Matrix();
		}
	}
	for (const auto& row : matrix2) {
		if (row.size() != cols) {
			return Matrix();
		}
	}

	// Создание и заполнение результирующей матрицы
	Matrix result(rows, Vector(cols, 0.0));

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result[i][j] = matrix1[i][j] + matrix2[i][j];
		}
	}

	return result;
}

Matrix matrix_multiplication(const Matrix& matrix1, const Matrix& matrix2) {
	// Проверка на пустые матрицы
	if (matrix1.empty() || matrix2.empty()) {
		return Matrix();
	}

	// Проверка согласованности размеров
	size_t cols1 = matrix1[0].size();
	size_t rows2 = matrix2.size();
	if (cols1 != rows2){
		return Matrix();
	}

	// Проверка, что все строки матриц имеют одинаковую длину
	for (const auto& row : matrix1) {
		if (row.size() != cols1) {
			return Matrix();
		}
	}

	size_t cols2 = matrix2[0].size();
	for (const auto& row : matrix2) {
		if (row.size() != cols2) {
			return Matrix();
		}
	}

	// Создание результирующей матрицы правильного размера
	size_t result_rows = matrix1.size();
	size_t result_cols = matrix2[0].size();
	Matrix result(result_rows, Vector(result_cols, 0.0));

	// Вычисление произведения
	for (size_t i = 0; i < result_rows; ++i) {
		for (size_t k = 0; k < cols1; ++k) {
			for (size_t j = 0; j < result_cols; ++j) {
				result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}

	return result;
}

Matrix hadamard_product(const Matrix& matrix1, const Matrix& matrix2) {
	// Проверка на пустые матрицы и строки
	if (matrix1.empty() || matrix2.empty() ||
		matrix1[0].empty() || matrix2[0].empty()) {
		return Matrix();
	}

	// Проверка совпадения размеров матриц
	const size_t rows = matrix1.size();
	const size_t cols = matrix1[0].size();

	if (rows != matrix2.size() || cols != matrix2[0].size()) {
		return Matrix();
	}

	// Проверка целостности матриц
	for (const auto& row : matrix1)
		if (row.size() != cols) return Matrix();

	for (const auto& row : matrix2)
		if (row.size() != cols) return Matrix();

	// Создание результирующей матрицы
	Matrix result(rows, Vector(cols, 0.0));

	// Поэлементное умножение
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result[i][j] = matrix1[i][j] * matrix2[i][j];
		}
	}

	return result;
}

//using namespace std;
namespace ActivationFunctions {
	bool StepFunction(long double value, long double step = 0.0) {
		if (value >= step) {
			return 1;
		}
		else {
			return 0;
		}
	}
	Matrix StepFunction(const Matrix & matx) {
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = StepFunction(matx[i][j]);
			}
		}
		return matx_;
	}
	long double Sigmoid(long double value) {
		return 1.0L / (1.0L + std::exp(-value));
	}
	Matrix Sigmoid(const Matrix & matx) {
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = Sigmoid(matx[i][j]);
			}
		}
		return matx_;
	}
	long double Tanh(long double value) {
		return std::tanhl(value);
	}
	Matrix Tanh(const Matrix & matx) {
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = Tanh(matx[i][j]);
			}
		}
		return matx_;
	}
	long double ReLU(long double value) {
		return std::fmaxl(0, value);
	}
	Matrix ReLU(const Matrix & matx) {
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = ReLU(matx[i][j]);
			}
		}
		return matx_;
	}
	long double LeakyReLU(long double value, long double a = 0.001) {
		if (value >= 0) {
			return value;
		}
		else {
			return (a * value);
		}
	}
	Matrix LeakyReLU(const Matrix & matx, const Matrix & ax) {
		if (matx.size() != ax.size()) {
			throw std::invalid_argument("Некорректный размер матрицы");
		}
		else {
			for (size_t errid = 0; errid < matx.size(); errid++) {
				if (matx[errid].size() != ax[errid].size()) {
					throw std::invalid_argument("Некорректный размер матрицы");
				}
			}
		}
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = LeakyReLU(matx[i][j], ax[i][j]);
			}
		}
		return matx_;
	}
	Matrix LeakyReLU(const Matrix & matx, const long double a) {
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = LeakyReLU(matx[i][j], a);
			}
		}
		return matx_;
	}
	long double Swish(long double value, long double b = 1.0) {
		long double x = value * b;
		// Ограничение для предотвращения переполнения exp(x)
		x = std::max(x, -700.0L); 
		x = std::min(x, 700.0L); 
		return value * (1.0L / (1.0L + std::exp(-x)));
	}
	Matrix Swish(const Matrix & matx, const Matrix & bx) {
		if (matx.size() != bx.size()) {
			throw std::invalid_argument("Некорректный размер матрицы");
		}
		else {
			for (size_t errid = 0; errid < matx.size(); errid++) {
				if (matx[errid].size() != bx[errid].size()) {
					throw std::invalid_argument("Некорректный размер матрицы");
				}
			}
		}
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = Swish(matx[i][j], bx[i][j]);
			}
		}
		return matx_;
	}
	Matrix Swish(const Matrix & matx, const long double b) {
		auto matx_ = matx;
		for (size_t i = 0; i < matx.size(); i++) {
			for (size_t j = 0; j < matx[i].size(); j++) {
				matx_[i][j] = Swish(matx[i][j], b);
			}
		}
		return matx_;
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
	Matrix matrix_random(const Matrix & matrix, long double a = std::numeric_limits<long double>::lowest(), long double b = std::numeric_limits<long double>::max()) {
		if (matrix.empty() || matrix[0].empty()) {
			return Matrix();
		}
		if (a > b) {
			std::swap(a, b);
		}
		auto result = matrix;
		for (size_t i = 0; i < result.size(); i++) {
			for (size_t j = 0; j < result[i].size(); j++) {
				result[i][j] = random(a, b);
			}
		}
	}
}

class SimpleSNT {
public:
	SimpleSNT(size_t Number_states_ = 1, size_t lenght_states_ = 1, size_t Hidden_size = 10) {
		this->Input_size = lenght_states_;
		this->Hidden_size = Hidden_size;
		Matrix matx1_(this->Hidden_size, Vector(this->Hidden_size));
		Matrix matx2_(this->Hidden_size, Vector(this->Input_size));
		Matrix matx3_(1, Vector(this->Hidden_size));
		this->Weights_for_FG_HS = ActivationFunctions::matrix_random(matx1_);
		this->Weights_for_IG_HS = ActivationFunctions::matrix_random(matx1_);
		this->Weights_for_CT_HS = ActivationFunctions::matrix_random(matx1_);
		this->Weights_for_OG_HS = ActivationFunctions::matrix_random(matx1_);
		this->Weights_for_FG_IS = ActivationFunctions::matrix_random(matx2_);
		this->Weights_for_IG_IS = ActivationFunctions::matrix_random(matx2_);
		this->Weights_for_CT_IS = ActivationFunctions::matrix_random(matx2_);
		this->Weights_for_OG_IS = ActivationFunctions::matrix_random(matx2_);
		this->Displacement_for_FG = ActivationFunctions::matrix_random(matx3_)[0];
		this->Displacement_for_IG = ActivationFunctions::matrix_random(matx3_)[0];
		this->Displacement_for_CT = ActivationFunctions::matrix_random(matx3_)[0];
		this->Displacement_for_OG = ActivationFunctions::matrix_random(matx3_)[0];

		this->Input_states.resize(Number_states_, Vector(this->Input_size, 0.0));
		this->Output_states.resize(Number_states_, Vector(this->Input_size, 0.0));
		this->Hidden_states.resize(Number_states_, Vector(this->Hidden_size, 0.0));
	}

	~SimpleSNT() = default;
	void SetInput_states(const Matrix & Input_states_) {
		if (this->Input_states.size() != Input_states_.size()) {

			this->Weights_for_IG_IS.resize(Input_states_.size(), Vector(this->Input_size));
			this->Weights_for_CT_IS.resize(Input_states_.size(), Vector(this->Input_size));
			this->Weights_for_OG_IS.resize(Input_states_.size(), Vector(this->Input_size));
			this->Weights_for_FG_IS.resize(Input_states_.size(), Vector(this->Input_size));

			this->Weights_for_FG_IS = ActivationFunctions::matrix_random(Matrix(this->Hidden_size, Vector(this->Input_size)));
			this->Weights_for_IG_IS = ActivationFunctions::matrix_random(Matrix(this->Hidden_size, Vector(this->Input_size)));
			this->Weights_for_CT_IS = ActivationFunctions::matrix_random(Matrix(this->Hidden_size, Vector(this->Input_size)));
			this->Weights_for_OG_IS = ActivationFunctions::matrix_random(Matrix(this->Hidden_size, Vector(this->Input_size)));
		}
		this->Input_states = Input_states_;
		this->Hidden_states.resize(this->Input_states.size(), Vector(this->Hidden_size, 0.0));
		this->Output_states.resize(this->Input_states.size(), Vector(this->Input_size, 0.0));
	}
	void SetWeights(const Matrix & weights_I_F, const Matrix & weights_I_I, const Matrix& weights_I_C, const Matrix& weights_I_O, const Matrix& weights_H_F, const Matrix& weights_H_I, const Matrix& weights_H_C, const Matrix& weights_H_O) {
		auto flag = 
			weights_I_F.empty() || weights_H_F.empty() || weights_I_F.size() != this->Hidden_size || weights_H_F.size() != this->Hidden_size ||
			weights_I_I.empty() || weights_H_I.empty() || weights_I_I.size() != this->Hidden_size || weights_H_I.size() != this->Hidden_size ||
			weights_I_C.empty() || weights_H_C.empty() || weights_I_C.size() != this->Hidden_size || weights_H_C.size() != this->Hidden_size ||
			weights_I_O.empty() || weights_H_O.empty() || weights_I_O.size() != this->Hidden_size || weights_H_O.size() != this->Hidden_size;
		if(flag == 0){
			for (size_t i = 0; i < this->Hidden_size; i++) {
				flag = flag || 
					weights_I_F[i].size() != this->Input_size || weights_H_F[i].size() != this->Hidden_size ||
					weights_I_I[i].size() != this->Input_size || weights_H_I[i].size() != this->Hidden_size ||
					weights_I_C[i].size() != this->Input_size || weights_H_C[i].size() != this->Hidden_size ||
					weights_I_O[i].size() != this->Input_size || weights_H_O[i].size() != this->Hidden_size;
			}
			if (flag == 0) {
				this->Weights_for_FG_HS = weights_H_F;
				this->Weights_for_IG_HS = weights_H_I;
				this->Weights_for_CT_HS = weights_H_C;
				this->Weights_for_OG_HS = weights_H_O;
				this->Weights_for_FG_IS = weights_I_F;
				this->Weights_for_IG_IS = weights_I_I;
				this->Weights_for_CT_IS = weights_I_C;
				this->Weights_for_OG_IS = weights_I_O;
			}
			else {
				throw std::invalid_argument("Input matrix size is incorrect");
			}
		}
		else {
			throw std::invalid_argument("Input matrix size is incorrect");
		}
	}
	void SetDisplacements(const Matrix& displacements_F, const Matrix& displacements_I, const Matrix& displacements_C, const Matrix& displacements_O) {
		auto flag =
			displacements_F.empty() || displacements_F.size() != this->Hidden_size ||
			displacements_I.empty() || displacements_I.size() != this->Hidden_size ||
			displacements_C.empty() || displacements_C.size() != this->Hidden_size ||
			displacements_O.empty() || displacements_O.size() != this->Hidden_size;
		if (flag == 0) {
			for (size_t i = 0; i < this->Hidden_size; i++) {
				flag = flag ||
					displacements_F[i].size() != this->Input_size ||
					displacements_I[i].size() != this->Input_size ||
					displacements_C[i].size() != this->Input_size ||
					displacements_O[i].size() != this->Input_size;
			}
			if (flag == 0) {
				this->Weights_for_FG_HS = displacements_F;
				this->Weights_for_IG_HS = displacements_I;
				this->Weights_for_CT_HS = displacements_C;
				this->Weights_for_OG_HS = displacements_O;
			}
			else {
				throw std::invalid_argument("Input matrix size is incorrect");
			}
		}
		else {
			throw std::invalid_argument("Input matrix size is incorrect");
		}
	}
	void SetRandomWeights(long double a = -50.0L, long double b = 50.0L) {
		Matrix matx1_(this->Hidden_size, Vector(this->Hidden_size));
		Matrix matx2_(this->Hidden_size, Vector(this->Input_size));
		SetWeights(ActivationFunctions::matrix_random(matx1_, a, b)
			, ActivationFunctions::matrix_random(matx1_, a, b)
			, ActivationFunctions::matrix_random(matx1_, a, b)
			, ActivationFunctions::matrix_random(matx1_, a, b)
			, ActivationFunctions::matrix_random(matx2_, a, b)
			, ActivationFunctions::matrix_random(matx2_, a, b)
			, ActivationFunctions::matrix_random(matx2_, a, b)
			, ActivationFunctions::matrix_random(matx2_, a, b)
		);
	}
	void SetRandomDisplacements(long double a = -50.0L, long double b = 50.0L) {
		Matrix matx_(1, Vector(this->Hidden_size));
		SetDisplacements(ActivationFunctions::matrix_random(matx_, a, b)
			, ActivationFunctions::matrix_random(matx_, a, b)
			, ActivationFunctions::matrix_random(matx_, a, b)
			, ActivationFunctions::matrix_random(matx_, a, b)
		);
	}
	/*void SetAll(const std::vector <long double> Input_states_, const std::vector <long double> weights, const std::vector <long double> displacements) {
		SetInput_states(Input_states_);
		SetWeights(weights);
		SetDisplacements(displacements);
	}*/
	void CalculationAll_states() {
		for (size_t i = 0; i < this->Input_states.size(); i++) {
			n_state_Сalculation(i);
		}
	}
	Matrix GetOutput_states() {
		Matrix result = this->Output_states;
		return result;
	}
private:
	size_t Input_size;
	size_t Hidden_size;
	Matrix Input_states;
	Matrix Hidden_states;
	Matrix Output_states;
	Matrix Weights_for_FG_HS;
	Matrix Weights_for_IG_HS;
	Matrix Weights_for_CT_HS;
	Matrix Weights_for_OG_HS;
	Matrix Weights_for_FG_IS;
	Matrix Weights_for_IG_IS;
	Matrix Weights_for_CT_IS;
	Matrix Weights_for_OG_IS;
	Vector Displacement_for_FG;
	Vector Displacement_for_IG;
	Vector Displacement_for_CT;
	Vector Displacement_for_OG;


	std::vector <Matrix> StepСalculation(Vector Hidden_State, Vector Last_State, Vector Input_State) {
		// Forget Gate: решаем, что забыть
		auto ForgetGate = ActivationFunctions::Sigmoid(
			matrix_addition(
				matrix_addition(
					matrix_multiplication(this->Weights_for_FG_HS, { Hidden_State }),
					matrix_multiplication(this->Weights_for_FG_IS, { Input_State })),
				{ Displacement_for_FG }
			)
		);

		// Input Gate: решаем, что обновить
		auto InputGate = ActivationFunctions::Sigmoid(
			matrix_addition(
				matrix_addition(
					matrix_multiplication(this->Weights_for_IG_HS, { Hidden_State }),
					matrix_multiplication(this->Weights_for_IG_IS, { Input_State })),
				{ Displacement_for_IG }
			)
		);

		// Новый кандидат для ячейки
		auto Ct_candidate = ActivationFunctions::Tanh(
			matrix_addition(
				matrix_addition(
					matrix_multiplication(this->Weights_for_CT_HS, { Hidden_State }),
					matrix_multiplication(this->Weights_for_CT_IS, { Input_State })),
				{ Displacement_for_FG }
			)
		);

		// Обновляем состояние ячейки
		auto NewState = matrix_addition(hadamard_product(ForgetGate, { Last_State }), hadamard_product(InputGate, Ct_candidate));

		// Output Gate: решаем, что передать в hidden_state
		auto OutputGate = ActivationFunctions::Sigmoid(
			matrix_addition(
				matrix_addition(
					matrix_multiplication(this->Weights_for_OG_HS, { Hidden_State }),
					matrix_multiplication(this->Weights_for_OG_IS, { Input_State })),
				{ Displacement_for_OG }
			)
		);

		// Новое скрытое состояние
		auto NewHidden_state = hadamard_product(OutputGate, ActivationFunctions::Tanh(NewState));

		return { NewState, NewHidden_state };
	}

	void n_state_Сalculation(size_t number) {
		auto vec_ = std::move(StepСalculation(this->Hidden_states[number - 1], this->Output_states[number - 1], this->Input_states[number]));
		this->Output_states[number] = std::move(vec_[0][0]);
		this->Hidden_states[number] = std::move(vec_[1][0]);
	}
};

int main() {
	/*SimpleSNT a(3);
	a.SetRandomWeights();
	a.CalculationAllNeurons();
	auto b = a.GetOutputNeurons();
	for (int i = 0; i < b.size(); i++) {
		std::cout << b[i] << "\t";
	}
	a.SetInputNeurons({ 50, 3, 4 });
	a.CalculationAllNeurons();
	b = a.GetOutputNeurons();
	for (int i = 0; i < b.size(); i++) {
		std::cout << b[i] << "\t";
	}*/
	return 0;
}