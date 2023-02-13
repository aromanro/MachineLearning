#pragma once

#include <Eigen/eigen>

template<typename OutputType = Eigen::VectorXd, typename BatchOutputType = Eigen::MatrixXd> class L2Cost
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return dif.cwiseProduct(dif);
	}

	OutputType operator()(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		BatchOutputType dif = output - expected;
		dif = dif.cwiseProduct(dif);

		for (unsigned int i = 0; i < output.cols(); ++i)
			sum += dif.col(i);
		
		return sum;
	}
};


template<> class L2Cost<double, Eigen::RowVectorXd>
{
public:
	double operator()(const double& output, const double& expected)
	{
		const double dif = output - expected;
		return dif * dif;
	}

	double operator()(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		const Eigen::RowVectorXd dif = output - expected;

		return dif.cwiseProduct(dif).sum();
	}
};


template<typename OutputType = Eigen::VectorXd, typename BatchOutputType = Eigen::MatrixXd> class L1Cost
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return dif.cwiseAbs();
	}

	OutputType operator()(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		BatchOutputType dif = output - expected;
		dif = dif.cwiseAbs();

		for (unsigned int i = 0; i < output.cols(); ++i)
			sum += dif.col(i);

		return sum;
	}
};


template<> class L1Cost<double, Eigen::RowVectorXd>
{
public:
	double operator()(const double& output, const double& expected)
	{
		return abs(output - expected);
	}

	double operator()(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		const Eigen::RowVectorXd dif = output - expected;

		return dif.cwiseAbs().sum();
	}
};