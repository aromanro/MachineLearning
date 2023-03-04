#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

template<typename OutputType = Eigen::VectorXd/*, typename BatchOutputType = Eigen::MatrixXd*/> class L2Loss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return dif.cwiseProduct(dif);
	}

	/*
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
	*/

	OutputType derivative(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return 2. * dif;
	}

	/*
	OutputType derivative(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		BatchOutputType dif = output - expected;

		for (unsigned int i = 0; i < output.cols(); ++i)
			sum += dif.col(i);

		sum /= dif.cols();

		return 2. * sum;
	}
	*/
};


template<typename OutputType = Eigen::VectorXd/*, typename BatchOutputType = Eigen::MatrixXd*/> class L1Loss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected)
	{
		const OutputType dif = output - expected;
		return dif.cwiseAbs();
	}

	/*
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
	*/

	// not really differentiable but that should not stop us :)
	OutputType derivative(const OutputType& output, const OutputType& expected)
	{
		OutputType dif = output - expected;

		for (unsigned int i = 0; i < dif.cols(); ++i)
			dif(i) = (dif(i) < 0) ? -1 : 1;

		return dif;
	}

	/*
	OutputType derivative(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		BatchOutputType dif = output - expected;
		for (unsigned int j = 0; j < dif.cols(); ++j)
			for (unsigned int i = 0; i < dif.rows(); ++i)
				dif(i, j) = (dif(i, j) < 0) ? -1 : 1;
		
		for (unsigned int i = 0; i < output.cols(); ++i)
			sum += dif.col(i);

		sum /= dif.cols();

		return sum;
	}
	*/
};


template<> class L1Loss<double/*, Eigen::RowVectorXd*/>
{
public:
	double operator()(const double& output, const double& expected)
	{
		return abs(output - expected);
	}

	/*
	double operator()(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		const Eigen::RowVectorXd dif = output - expected;

		return dif.cwiseAbs().sum() / dif.cols();
	}
	*/

	double derivative(const double& output, const double& expected)
	{
		return ((output - expected) < 0) ? -1 : 1;
	}

	/*
	double derivative(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		Eigen::RowVectorXd dif = output - expected;

		for (unsigned int i = 0; i < dif.cols(); ++i)
			dif(i) = (dif(i) < 0) ? -1 : 1;

		return dif.sum() / dif.cols();
	}
	*/
};

template<typename OutputType = Eigen::VectorXd/*, typename BatchOutputType = Eigen::MatrixXd*/> class BinaryCrossEntropyLoss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected)
	{
		return -(output * expected.log() + (1. - expected) * (1. - output).log());
	}

	/*
	OutputType operator()(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		for (unsigned int i = 0; i < output.cols(); ++i)
			sum -= output.col(i) * expected.col(i).log() + (1. - expected.col(i)) * (1. - output.col(i)).log();

		sum /= output.cols();

		return sum;
	}
	*/

	OutputType derivative(const OutputType& output, const OutputType& expected)
	{
		return output / (expected + 1E-10) + (1. - output) / (1. + 1E-10 - expected);
	}

	/*
	OutputType derivative(const BatchOutputType& output, const BatchOutputType& expected)
	{
		OutputType sum = OutputType::Zero(output.rows());

		for (unsigned int i = 0; i < output.cols(); ++i)
			sum -= output.col(i) * expected.col(i).log() + (1. - expected.col(i)) * (1. - output.col(i)).log();

		sum /= output.cols();

		return sum;
	}
	*/
};


template<> class BinaryCrossEntropyLoss<double/*, Eigen::RowVectorXd*/>
{
public:
	double operator()(const double& output, const double& expected)
	{
		return -(output * log(expected) + (1. - expected) * log(1. - output));
	}

	/*
	double operator()(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		const Eigen::MatrixXd one = Eigen::MatrixXd::Ones(output.rows(), output.cols());
		const Eigen::MatrixXd exp = expected; // I needed to do this to fight an Eigen issue

		const Eigen::MatrixXd oe = one - expected;
		const Eigen::MatrixXd oo = one - output;

		return -(output.cwiseProduct(exp.log()) + oe.cwiseProduct(oo.log())).sum() / output.cols();
	}
	*/

	double derivative(const double& output, const double& expected)
	{
		return output / (expected + 1E-10) + (1. - output) / (1. + 1E-10 - expected);
	}

	/*
	double derivative(const Eigen::RowVectorXd& output, const Eigen::RowVectorXd& expected)
	{
		const Eigen::MatrixXd one = Eigen::MatrixXd::Ones(output.rows(), output.cols());

		const Eigen::MatrixXd oe = one - expected;
		const Eigen::MatrixXd oo = one - output;

		return (output.cwiseProduct(expected.cwiseInverse()) + oo.cwiseProduct(oe.cwiseInverse())).sum() / output.cols();
	}
	*/
};

