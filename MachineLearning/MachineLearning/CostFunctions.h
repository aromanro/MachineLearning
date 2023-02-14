#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

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

		sum /= dif.cols();
		
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

		return dif.cwiseProduct(dif).sum() / dif.cols();
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

		sum /= dif.cols();

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

		return dif.cwiseAbs().sum() / dif.cols();
	}
};

template<typename OutputType = Eigen::VectorXd, typename BatchOutputType = Eigen::MatrixXd> class CrossEntropyCost
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected)
	{
		return -(output * expected.log() + (1. - expected) * (1. - output).log());
	}

	OutputType operator()(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		for (unsigned int i = 0; i < output.cols(); ++i)
			sum -= output.col(i) * expected.col(i).log() + (1. - expected.col(i)) * (1. - output.col(i)).log();

		sum /= output.cols();

		return sum;
	}
};


template<> class CrossEntropyCost<double, Eigen::RowVectorXd>
{
public:
	double operator()(const double& output, const double& expected)
	{
		return -(output * log(expected) + (1. - expected) * log(1. - output));
	}

	double operator()(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		const Eigen::MatrixXd one = Eigen::RowVectorXd::Ones(output.cols());
		const Eigen::MatrixXd exp = expected; // I needed to do this to fight an Eigen issue

		const Eigen::MatrixXd oe = one - expected;
		const Eigen::MatrixXd oo = one - output;

		return -(output.cwiseProduct(exp.log()) + oe.cwiseProduct(oo.log())).sum() / output.cols();
	}
};