#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

template<typename OutputType = Eigen::VectorXd/*, typename BatchOutputType = Eigen::MatrixXd*/> class L2Loss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected) const
	{
		const OutputType dif = output - expected;
		return dif.cwiseProduct(dif);
	}

	OutputType derivative(const OutputType& output, const OutputType& expected) const
	{
		const OutputType dif = output - expected;
		return 2. * dif;
	}
};


template<typename OutputType = Eigen::VectorXd/*, typename BatchOutputType = Eigen::MatrixXd*/> class L1Loss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected) const 
	{
		const OutputType dif = output - expected;
		return dif.cwiseAbs();
	}

	// not really differentiable but that should not stop us :)
	OutputType derivative(const OutputType& output, const OutputType& expected) const
	{
		OutputType dif = output - expected;

		for (unsigned int i = 0; i < dif.cols(); ++i)
			dif(i) = (dif(i) < 0) ? -1 : 1;

		return dif;
	}
};


template<> class L1Loss<double/*, Eigen::RowVectorXd*/>
{
public:
	double operator()(const double& output, const double& expected) const
	{
		return abs(output - expected);
	}

	double derivative(const double& output, const double& expected) const
	{
		return ((output - expected) < 0) ? -1 : 1;
	}
};

template<typename OutputType = Eigen::VectorXd/*, typename BatchOutputType = Eigen::MatrixXd*/> class BinaryCrossEntropyLoss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected) const
	{
		return -(output * expected.log() + (1. - expected) * (1. - output).log());
	}

	OutputType derivative(const OutputType& output, const OutputType& expected) const
	{
		return output / (expected + 1E-10) + (1. - output) / (1. + 1E-10 - expected);
	}
};


template<> class BinaryCrossEntropyLoss<double/*, Eigen::RowVectorXd*/>
{
public:
	double operator()(const double& output, const double& expected)
	{
		return -(output * log(expected) + (1. - expected) * log(1. - output));
	}

	double derivative(const double& output, const double& expected)
	{
		return output / (expected + 1E-10) + (1. - output) / (1. + 1E-10 - expected);
	}
};

