#include "HeaderActivateFunctionsForNN.h"

namespace ActivationFunctions {
	bool StepFunction(long double value, long double step) {
		if (value >= step) {
			return 1;
		}
		else {
			return 0;
		}
	}
	MatrixXld StepFunction(const MatrixXld& matx, long double step) {
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
	MatrixXld Sigmoid(const MatrixXld& matx, long double norm) {
		return matx.unaryExpr([&](long double x) {
			x = std::max(-norm, std::min(norm, x)); // �����������
			return 1.0L / (1.0L + std::exp(-x));
			});
	}
	long double Tanh(long double value) {
		return std::tanhl(value);
	}
	MatrixXld Tanh(const MatrixXld& matx, long double norm) {
		return matx.unaryExpr([&](long double x) {
			x = std::max<long double>(-norm, std::min<long double>(norm, x));
			return std::tanhl(x);
			});
	}
	long double ReLU(long double value) {
		return std::fmaxl(0, value);
	}
	MatrixXld ReLU(const MatrixXld& matx) {
		return matx.unaryExpr([](auto x) { return std::fmaxl(0.0L, x); });
	}
	long double LeakyReLU(long double value, long double a) {
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
	MatrixXld LeakyReLU(const MatrixXld& matx, long double a) {
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
	long double Swish(long double value, long double b) {
		long double x = value * b;
		// ����������� ��� �������������� ������������ exp(x)
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
	MatrixXld Swish(const MatrixXld& matx, long double b) {
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
			// ������� ����������� ������������� ��� ������� ����������
			throw std::runtime_error("��� �������� � Softmax �������");
		}
		for (auto& val : result) {
			val /= sum;
		}
		return result;
	}
	VectorXld Softmax(const VectorXld& x, long double clamp_val, long double eps){
		// 1) ������� �����
		VectorXld x_clamped = x.unaryExpr([&](long double v) {
			return std::max(-clamp_val, std::min(clamp_val, v));
			});

		// 2) ��������� ��������
		long double x_max = x_clamped.maxCoeff();

		// 3) ��������� ���������� �� (x - max)
		VectorXld exp_x = x_clamped.unaryExpr([&](long double v) {
			return std::exp(v - x_max);
			});

		// 4) ����� � eps
		long double sum_exp = exp_x.sum() + eps;

		// 5) ����������
		return exp_x.array() / sum_exp;
	}
	long double random(long double a, long double b) {
		if (a >= b) {
			throw std::invalid_argument("a must be less than b");
		}

		static std::mt19937_64 generator(std::random_device{}());
		std::uniform_real_distribution<long double> distribution(a, b);
		return distribution(generator);
	}
	MatrixXld matrix_random(size_t rows, size_t cols, long double a, long double b ) {
		return MatrixXld::Random(rows, cols) * (b - a) + MatrixXld::Constant(rows, cols, a);
	}
	MatrixXld matrix_random(const MatrixXld& matrix, long double a, long double b) {
		return MatrixXld::Random(matrix.rows(), matrix.cols()) * (b - a) + MatrixXld::Constant(matrix.rows(), matrix.cols(), a);
	}
}