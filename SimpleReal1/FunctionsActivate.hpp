#pragma once

#include "RealizationMatrix.hpp"

namespace FunctionsActivate{}

#ifdef EIGEN_MATRIX
namespace FunctionsActivate {
	using LinearAlgebra::BaseMatrix;
	using LinearAlgebra::BaseRowVector;
	using LinearAlgebra::BaseVector;

	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseStepFunction(const BaseMatrix<T, Enable>& matx, double step) {
		BaseMatrix<T, Enable> result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = matx(i, j) >= step ? T(1) : T(0);
			}
		}
		return result;
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseSigmoid(const BaseMatrix<T, Enable>& matx, double norm) {
		return matx.unaryExpr([&](T x) {
			x = std::max(-norm, std::min(norm, x)); // Ограничение
			return static_cast<T, Enable>(1 / (1 + std::exp(-x)));
			});
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseTanh(const BaseMatrix<T, Enable>& matx, double norm) {
		return matx.unaryExpr([&](T x) {
			x = std::max(-norm, std::min(norm, x));
			return static_cast<T, Enable>(std::tanh(x));
			});
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseReLU(const BaseMatrix<T, Enable>& matx) {
		return matx.unaryExpr([](T x) { return static_cast<T, Enable>(std::max(T(0), x)); });
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseLeakyReLU(const BaseMatrix<T, Enable>& matx, const BaseMatrix<T, Enable>& a) {
		if (matx.rows() != a.rows() || matx.cols() != a.cols()) {
			throw std::invalid_argument("BaseMatrix<T, Enable> dimensions must match");
		}

		BaseMatrix<T, Enable> result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= T(0))
					? matx(i, j)
					: a(i, j) * matx(i, j);
			}
		}
		return result;
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseLeakyReLU(const BaseMatrix<T, Enable>& matx, double a) {
		BaseMatrix<T, Enable> result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= T(0))
					? matx(i, j)
					: static_cast<T, Enable>(a * matx(i, j));
			}
		}
		return result;
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseSwish(const BaseMatrix<T, Enable>& matx, const BaseMatrix<T, Enable>& b, double norm) {
		if (matx.rows() != b.rows() || matx.cols() != b.cols()) {
			throw std::invalid_argument("BaseMatrix<T, Enable> dimensions must match");
		}

		return matx.binaryExpr(b, [&](T m, T bb) {
			double x = m * bb;
			x = std::max(-norm, std::min(norm, x));
			return static_cast<T, Enable>(m * (1.0 / (1.0 + std::exp(-x))));
			});
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseMatrix<T, Enable> baseSwish(const BaseMatrix<T, Enable>& matx, double b, double norm) {
		return matx.unaryExpr([&](T m) {
			double x = m * b;
			x = std::max(-norm, std::min(norm, x));
			return static_cast<T, Enable>(m * (1.0 / (1.0 + std::exp(-x))));
			});
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseVector<T, Enable> baseSoftmax(const BaseVector<T, Enable>& x, double clamp_val, double eps) {
		// 1) Клэмпим входы
		BaseVector<T, Enable> x_clamped = x.unaryExpr([&](T v) {
			return std::max(-clamp_val, std::min(clamp_val, v));
			});

		// 2) Вычисляем максимум
		double x_max = x_clamped.maxCoeff();

		// 3) Вычисляем экспоненты от (x - max)
		BaseVector<T, Enable> exp_x = (x_clamped.array() - static_cast<T, Enable>(x_max)).exp();

		// 4) Сумма с eps
		double sum_exp = exp_x.sum() + eps;

		// 5) Нормировка
		return exp_x.array() / sum_exp;
	}
	template<typename T = float, typename Enable = std::enable_if_t<std::is_arithmetic_v<T>>> 
	BaseRowVector<T, Enable> baseSoftmax(const BaseRowVector<T, Enable>& x, double clamp_val, double eps) {
		// 1) Клэмпим входы
		BaseRowVector<T, Enable> x_clamped = x.unaryExpr([&](T v) {
			return std::max(-clamp_val, std::min(clamp_val, v));
			});

		// 2) Вычисляем максимум
		double x_max = x_clamped.maxCoeff();

		// 3) Вычисляем экспоненты от (x - max)
		BaseRowVector<T, Enable> exp_x = (x_clamped.array() - static_cast<T, Enable>(x_max)).exp();

		// 4) Сумма с eps
		double sum_exp = exp_x.sum() + eps;

		// 5) Нормировка
		return exp_x.array() / sum_exp;
	}
}
#endif //EIGEN_MATRIX