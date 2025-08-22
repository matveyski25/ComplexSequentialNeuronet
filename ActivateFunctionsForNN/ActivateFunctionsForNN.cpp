#include "HeaderActivateFunctionsForNN.h"

namespace ActivationFunctions {
	bool StepFunction(double value, double step) {
		if (value >= step) {
			return 1;
		}
		else {
			return 0;
		}
	}
	MatrixXld StepFunction(const MatrixXld& matx, double step) {
		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = matx(i, j) >= step ? 1.0L : 0.0L;
			}
		}
		return result;
	}
	double Sigmoid(double value) {
		return 1.0L / (1.0L + std::exp(-value));
	}
	MatrixXld Sigmoid(const MatrixXld& matx, double norm) {
		return matx.unaryExpr([&](double x) {
			x = std::max(-norm, std::min(norm, x)); // Ограничение
			return 1.0 / (1.0 + std::exp(-x));
			});
	}
	double Tanh(double value) {
		return std::tanh(value);
	}
	MatrixXld Tanh(const MatrixXld& matx, double norm) {
		return matx.unaryExpr([&](double x) {
			x = std::max<double>(-norm, std::min<double>(norm, x));
			return std::tanh(x);
			});
	}
	double ReLU(double value) {
		return std::fmaxl(0, value);
	}
	MatrixXld ReLU(const MatrixXld& matx) {
		return matx.unaryExpr([](auto x) { return std::fmax(0.0, x); });
	}
	double LeakyReLU(double value, double a) {
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
				result(i, j) = (matx(i, j) >= 0.0)
					? matx(i, j)
					: a(i, j) * matx(i, j);
			}
		}
		return result;
	}
	MatrixXld LeakyReLU(const MatrixXld& matx, double a) {
		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= 0.0)
					? matx(i, j)
					: a * matx(i, j);
			}
		}
		return result;
	}
	double Swish(double value, double b) {
		double x = value * b;
		// Ограничение для предотвращения переполнения exp(x)
		x = std::max(x, -700.0);
		x = std::min(x, 700.0);
		return value * (1.0L / (1.0L + std::exp(-x)));
	}
	MatrixXld Swish(const MatrixXld& matx, const MatrixXld& b) {
		if (matx.rows() != b.rows() || matx.cols() != b.cols()) {
			throw std::invalid_argument("MatrixXld dimensions must match");
		}

		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				double x = matx(i, j) * b(i, j);
				x = std::max(x, -700.0);
				x = std::min(x, 700.0);
				result(i, j) = matx(i, j) / (1.0L + std::exp(-x));
			}
		}
		return result;
	}
	MatrixXld Swish(const MatrixXld& matx, double b) {
		MatrixXld result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				double x = matx(i, j) * b;
				x = std::max(x, -700.0);
				x = std::min(x, 700.0);
				result(i, j) = matx(i, j) / (1.0L + std::exp(-x));
			}
		}
		return result;
	}
	std::vector<double> Softmax(const std::vector<double>& values) {
		if (values.empty()) {
			throw std::invalid_argument("Input vector is empty");
		}
		double max_val = *std::max_element(values.begin(), values.end());
		double sum = 0.0;
		std::vector<double> result;
		result.reserve(values.size());
		for (auto v : values) {
			double exp_val = std::exp(v - max_val);
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
	VectorXld Softmax(const VectorXld& x, double clamp_val, double eps) {
		// 1) Клэмпим входы
		VectorXld x_clamped = x.unaryExpr([&](double v) {
			return std::max(-clamp_val, std::min(clamp_val, v));
			});

		// 2) Вычисляем максимум
		double x_max = x_clamped.maxCoeff();

		// 3) Вычисляем экспоненты от (x - max)
		VectorXld exp_x = x_clamped.unaryExpr([&](double v) {
			return std::exp(v - x_max);
			});

		// 4) Сумма с eps
		double sum_exp = exp_x.sum() + eps;

		// 5) Нормировка
		return exp_x.array() / sum_exp;
	}
	double random(double a, double b) {
		if (a >= b) {
			throw std::invalid_argument("a must be less than b");
		}

		static std::mt19937_64 generator(std::random_device{}());
		std::uniform_real_distribution<double> distribution(a, b);
		return distribution(generator);
	}
	MatrixXld matrix_random(size_t rows, size_t cols, double a, double b ) {
		return MatrixXld::Random(rows, cols) * (b - a) + MatrixXld::Constant(rows, cols, a);
	}
	MatrixXld matrix_random(const MatrixXld& matrix, double a, double b) {
		return MatrixXld::Random(matrix.rows(), matrix.cols()) * (b - a) + MatrixXld::Constant(matrix.rows(), matrix.cols(), a);
	}
}