#pragma once

#include <iostream>

#include <vector>
#include <random>

//#define EIGEN_NO_DEBUG

#ifdef _MSC_VER
// Отключаем только специфичные warning'и, которые возникают из-за Eigen
#pragma warning(push)
#pragma warning(disable : 26495 6255 6294)  // Только эти номера, только на время include
#endif

#include <eigen-3.4.0/Eigen/Dense>
#include <eigen-3.4.0/Eigen/Core>

#ifdef _MSC_VER
#pragma warning(pop)
#endif



using MatrixXld = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXld = Eigen::Matrix<long double, 1, Eigen::Dynamic>; // Вектор-строка
using VectorXld = Eigen::Matrix<long double, Eigen::Dynamic, 1>;    // Вектор-столбец

namespace ActivationFunctions {
	bool StepFunction(long double value, long double step);
	MatrixXld StepFunction(const MatrixXld& matx, long double step);
	long double Sigmoid(long double value);
	MatrixXld Sigmoid(const MatrixXld& matx);
	long double Tanh(long double value);
	MatrixXld Tanh(const MatrixXld& matx);
	long double ReLU(long double value);
	MatrixXld ReLU(const MatrixXld& matx);
	long double LeakyReLU(long double value, long double a);
	MatrixXld LeakyReLU(const MatrixXld& matx, const MatrixXld& a);
	MatrixXld LeakyReLU(const MatrixXld& matx, long double a);
	long double Swish(long double value, long double b);
	MatrixXld Swish(const MatrixXld& matx, const MatrixXld& b);
	MatrixXld Swish(const MatrixXld& matx, long double b);
	std::vector<long double> Softmax(const std::vector<long double>& values);
	VectorXld Softmax(const VectorXld& values);
	long double random(long double a, long double b);
	MatrixXld matrix_random(size_t rows, size_t cols, long double a = 0.0L, long double b = 1.0L);
	MatrixXld matrix_random(const MatrixXld& matrix, long double, long double);
}