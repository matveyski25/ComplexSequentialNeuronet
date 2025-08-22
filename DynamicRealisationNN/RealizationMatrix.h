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

namespace LinearAlgebra {
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	using BaseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	using BaseRowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>; // Вектор-строка
	template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	using BaseVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;    // Вектор-столбец
}

#endif // EIGEN_MATRIX
