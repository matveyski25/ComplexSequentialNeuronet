#pragma once

namespace LinearAlgebra {}

#define EIGEN_MATRIX

#ifdef EIGEN_MATRIX
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26495 6255 6294)
#endif

#include "../Lib/Eigen/Core"
#include "../Lib/Eigen/Dense"

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#define TEMPLATE_ARITH(T) template<typename T = float, typename = std::enable_if_t<std::is_arithmetic_v<T>>>

namespace LinearAlgebra {
	TEMPLATE_ARITH(T)
	using BaseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	TEMPLATE_ARITH(T) 
	using BaseRowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>; // Вектор-строка
	TEMPLATE_ARITH(T) 
	using BaseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;    // Вектор-столбец
}

#endif // EIGEN_MATRIX
