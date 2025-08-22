#pragma once

#include "RealizationMatrix.hpp"

namespace FunctionsActivate{}

#ifdef EIGEN_MATRIX
namespace FunctionsActivate {
	using LinearAlgebra::BaseMatrix;
	using LinearAlgebra::BaseRowVector;
	using LinearAlgebra::BaseVector;

	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseStepFunction(const BaseMatrix<T>& matx, double step) {
		BaseMatrix<T> result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = matx(i, j) >= step ? T(1) : T(0);
			}
		}
		return result;
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseSigmoid(const BaseMatrix<T>& matx, double norm) {
		return matx.unaryExpr([&](T x) {
			x = std::max(-norm, std::min(norm, x)); // Ограничение
			return static_cast<T>(1 / (1 + std::exp(-x)));
			});
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseTanh(const BaseMatrix<T>& matx, double norm) {
		return matx.unaryExpr([&](T x) {
			x = std::max(-norm, std::min(norm, x));
			return static_cast<T>(std::tanh(x));
			});
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseReLU(const BaseMatrix<T>& matx) {
		return matx.unaryExpr([](T x) { return static_cast<T>(std::max(T(0), x)); });
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseLeakyReLU(const BaseMatrix<T>& matx, const BaseMatrix<T>& a) {
		if (matx.rows() != a.rows() || matx.cols() != a.cols()) {
			throw std::invalid_argument("BaseMatrix<T> dimensions must match");
		}

		BaseMatrix<T> result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= T(0))
					? matx(i, j)
					: a(i, j) * matx(i, j);
			}
		}
		return result;
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseLeakyReLU(const BaseMatrix<T>& matx, double a) {
		BaseMatrix<T> result(matx.rows(), matx.cols());
		for (Eigen::Index i = 0; i < matx.rows(); ++i) {
			for (Eigen::Index j = 0; j < matx.cols(); ++j) {
				result(i, j) = (matx(i, j) >= T(0))
					? matx(i, j)
					: static_cast<T>(a * matx(i, j));
			}
		}
		return result;
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseSwish(const BaseMatrix<T>& matx, const BaseMatrix<T>& b, double norm) {
		if (matx.rows() != b.rows() || matx.cols() != b.cols()) {
			throw std::invalid_argument("BaseMatrix<T> dimensions must match");
		}

		return matx.binaryExpr(b, [&](T m, T bb) {
			double x = m * bb;
			x = std::max(-norm, std::min(norm, x));
			return static_cast<T>(m * (1.0 / (1.0 + std::exp(-x))));
			});
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseMatrix<T> baseSwish(const BaseMatrix<T>& matx, double b, double norm) {
		return matx.unaryExpr([&](T m) {
			double x = m * b;
			x = std::max(-norm, std::min(norm, x));
			return static_cast<T>(m * (1.0 / (1.0 + std::exp(-x))));
			});
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseVector<T> baseSoftmax(const BaseVector<T>& x, double clamp_val, double eps) {
		// 1) Клэмпим входы
		BaseVector<T> x_clamped = x.unaryExpr([&](T v) {
			return std::max(-clamp_val, std::min(clamp_val, v));
			});

		// 2) Вычисляем максимум
		double x_max = x_clamped.maxCoeff();

		// 3) Вычисляем экспоненты от (x - max)
		BaseVector<T> exp_x = (x_clamped.array() - static_cast<T>(x_max)).exp();

		// 4) Сумма с eps
		double sum_exp = exp_x.sum() + eps;

		// 5) Нормировка
		return exp_x.array() / sum_exp;
	}
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	BaseRowVector<T> baseSoftmax(const BaseRowVector<T>& x, double clamp_val, double eps) {
		// 1) Клэмпим входы
		BaseRowVector<T> x_clamped = x.unaryExpr([&](T v) {
			return std::max(-clamp_val, std::min(clamp_val, v));
			});

		// 2) Вычисляем максимум
		double x_max = x_clamped.maxCoeff();

		// 3) Вычисляем экспоненты от (x - max)
		BaseRowVector<T> exp_x = (x_clamped.array() - static_cast<T>(x_max)).exp();

		// 4) Сумма с eps
		double sum_exp = exp_x.sum() + eps;

		// 5) Нормировка
		return exp_x.array() / sum_exp;
	}
}
#endif //EIGEN_MATRIX