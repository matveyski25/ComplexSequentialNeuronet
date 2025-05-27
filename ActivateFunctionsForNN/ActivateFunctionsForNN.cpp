#include "HeaderActivateFunctionsForNN.h"

namespace ActivationFunctions {
	bool StepFunction(long double value, long double step = 0.0) {
		if (value >= step) {
			return 1;
		}
		else {
			return 0;
		}
	}
	MatrixXld StepFunction(const MatrixXld& matx, long double step = 0.0) {
		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = matx(i, j) >= step ? 1.0L : 0.0L;
			}
		}
		return result;
	}
	long double Sigmoid(long double value) {
		return 1.0L / (1.0L + std::exp(-value));
	}
	MatrixXld Sigmoid(const MatrixXld& matx) {
		return matx.unaryExpr([](long double x) {
			x = std::max(-700.0L, std::min(700.0L, x)); // Ограничение
			return 1.0L / (1.0L + std::exp(-x));
			});
	}
	long double Tanh(long double value) {
		return std::tanhl(value);
	}
	MatrixXld Tanh(const MatrixXld& matx) {
		return matx.unaryExpr([](long double x) {
			x = std::max<long double>(-700.0L, std::min<long double>(700.0L, x));
			return std::tanhl(x);
			});
	}
	long double ReLU(long double value) {
		return std::fmaxl(0, value);
	}
	MatrixXld ReLU(const MatrixXld& matx) {
		return matx.unaryExpr([](auto x) { return std::fmaxl(0.0L, x); });
	}
	long double LeakyReLU(long double value, long double a = 0.001) {
		if (value >= 0) {
			return value;
		}
		else {
			return (a * value);
		}
	}
	MatrixXld LeakyReLU(const MatrixXld& matx, const MatrixXld& a) {
		if (matx.rows() != a.rows() || matx.cols() != a.cols()) {
			throw std::invalid_argument("MatrixXld dimensions must match");
		}

		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= 0.0L)
					? matx(i, j)
					: a(i, j) * matx(i, j);
			}
		}
		return result;
	}
	MatrixXld LeakyReLU(const MatrixXld& matx, long double a = 0.001L) {
		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= 0.0L)
					? matx(i, j)
					: a * matx(i, j);
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
	MatrixXld Swish(const MatrixXld& matx, const MatrixXld& b) {
		if (matx.rows() != b.rows() || matx.cols() != b.cols()) {
			throw std::invalid_argument("MatrixXld dimensions must match");
		}

		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				long double x = matx(i, j) * b(i, j);
				x = std::max(x, -700.0L);
				x = std::min(x, 700.0L);
				result(i, j) = matx(i, j) / (1.0L + std::exp(-x));
			}
		}
		return result;
	}
	MatrixXld Swish(const MatrixXld& matx, long double b = 1.0L) {
		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				long double x = matx(i, j) * b;
				x = std::max(x, -700.0L);
				x = std::min(x, 700.0L);
				result(i, j) = matx(i, j) / (1.0L + std::exp(-x));
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
	long double random(long double a = 0.0L, long double b = 1.0L) {
		if (a >= b) {
			throw std::invalid_argument("a must be less than b");
		}

		static std::mt19937_64 generator(std::random_device{}());
		std::uniform_real_distribution<long double> distribution(a, b);
		return distribution(generator);
	}
	MatrixXld matrix_random(size_t rows, size_t cols, long double a = 0.0L, long double b = 1.0L) {
		return MatrixXld::Random(rows, cols) * (b - a) + MatrixXld::Constant(rows, cols, a);
	}
	MatrixXld matrix_random(const MatrixXld& matrix, long double a = 0.0L, long double b = 1.0L) {
		return MatrixXld::Random(matrix.rows(), matrix.cols()) * (b - a) + MatrixXld::Constant(matrix.rows(), matrix.cols(), a);
	}
}