#pragma once

#include <Eigen/eigen>

template<typename OutputType = Eigen::VectorXd, typename BatchOutputType = Eigen::MatrixXd> class L2Cost
{
public:
	double operator()(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return dif.cwiseProduct(dif).sum();
	}
};


template<> class L2Cost<double, double>
{
public:
	double operator()(const double& output, const double& expected)
	{
		const double dif = output - expected;
		return dif * dif;
	}
};


template<typename OutputType = Eigen::VectorXd, typename BatchOutputType = Eigen::MatrixXd> class L1Cost
{
public:
	double operator()(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return dif.abs().sum();
	}
};


template<> class L1Cost<double, double>
{
public:
	double operator()(const double& output, const double& expected)
	{
		return abs(output - expected);
	}
};