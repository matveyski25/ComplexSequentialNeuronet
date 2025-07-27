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


namespace ActivationFunctions {
	using MatrixXld = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
	using RowVectorXld = Eigen::Matrix<double, 1, Eigen::Dynamic>; // Вектор-строка
	using VectorXld = Eigen::Matrix<double, Eigen::Dynamic, 1>;    // Вектор-столбец

	bool StepFunction(double value, double step = 0.0);
	MatrixXld StepFunction(const MatrixXld& matx, double step = 0.0);
	double Sigmoid(double value);
	MatrixXld Sigmoid(const MatrixXld& matx, double norm = 700.0L);
	double Tanh(double value);
	MatrixXld Tanh(const MatrixXld& matx, double norm = 700.0L);
	double ReLU(double value);
	MatrixXld ReLU(const MatrixXld& matx);
	double LeakyReLU(double value, double a = 0.001);
	MatrixXld LeakyReLU(const MatrixXld& matx, const MatrixXld& a);
	MatrixXld LeakyReLU(const MatrixXld& matx, double a = 0.001L);
	double Swish(double value, double b = 1.0);
	MatrixXld Swish(const MatrixXld& matx, const MatrixXld& b);
	MatrixXld Swish(const MatrixXld& matx, double b = 1.0L);
	std::vector<double> Softmax(const std::vector<double>& values);
	VectorXld Softmax(const VectorXld& x, double clamp_val = 700.0L, double eps = 1e-8L);
	double random(double a = 0.0L, double b = 1.0L);
	MatrixXld matrix_random(size_t rows, size_t cols, double a = 0.0L, double b = 1.0L);
	MatrixXld matrix_random(const MatrixXld& matrix, double a = 0.0L, double b = 1.0L);
}