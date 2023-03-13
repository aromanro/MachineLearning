#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

template<typename OutputType = Eigen::VectorXd> class L2Loss
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

template<> class L2Loss<double>
{
public:
	double operator()(const double& output, const double& expected) const
	{
		const double dif = output - expected;
		return dif * dif;
	}

	double derivative(const double& output, const double& expected) const
	{
		const double dif = output - expected;
		return 2. * dif;
	}
};


template<typename OutputType = Eigen::VectorXd> class L1Loss
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


template<> class L1Loss<double>
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

template<typename OutputType = Eigen::VectorXd> class BinaryCrossEntropyLoss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& expected) const
	{
		static const double eps = 1E-10;
		const OutputType ones = OutputType::Ones(output.size());
		const OutputType epsv = OutputType::Constant(output.size(), eps);

		return -(expected.cwiseProduct((output + epsv).log()) + (ones - expected).cwiseProduct((ones + epsv - output).log()));
	}

	OutputType derivative(const OutputType& output, const OutputType& expected) const
	{
		return output - expected;
	}
};


template<> class BinaryCrossEntropyLoss<double>
{
public:
	double operator()(const double& output, const double& expected) const
	{
		static const double eps = 1E-10;
		return -(expected * log(output + eps) + (1. - expected) * log(1. + eps - output));
	}

	double derivative(const double& output, const double& expected) const
	{
		return output - expected;
	}
};

