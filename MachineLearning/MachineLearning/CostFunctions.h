#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

template<typename OutputType = Eigen::VectorXd> class L2Loss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& target) const
	{
		const OutputType dif = output - target;
		return dif.cwiseProduct(dif);
	}

	OutputType derivative(const OutputType& output, const OutputType& target) const
	{
		const OutputType dif = output - target;
		return 2. * dif;
	}
};

template<> class L2Loss<double>
{
public:
	double operator()(const double& output, const double& target) const
	{
		const double dif = output - target;
		return dif * dif;
	}

	double derivative(const double& output, const double& target) const
	{
		const double dif = output - target;
		return 2. * dif;
	}
};


template<typename OutputType = Eigen::VectorXd> class L1Loss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& target) const 
	{
		const OutputType dif = output - target;
		return dif.cwiseAbs();
	}

	// not really differentiable but that should not stop us :)
	OutputType derivative(const OutputType& output, const OutputType& target) const
	{
		OutputType dif = output - target;

		for (unsigned int i = 0; i < dif.cols(); ++i)
			dif(i) = (dif(i) < 0) ? -1 : 1;

		return dif;
	}
};


template<> class L1Loss<double>
{
public:
	double operator()(const double& output, const double& target) const
	{
		return abs(output - target);
	}

	double derivative(const double& output, const double& target) const
	{
		return ((output - target) < 0) ? -1 : 1;
	}
};

template<typename OutputType = Eigen::VectorXd> class BinaryCrossEntropyLoss
{
public:
	OutputType operator()(const OutputType& output, const OutputType& target) const
	{
		static const double eps = 1E-10;
		const OutputType ones = OutputType::Ones(output.size());
		const OutputType epsv = OutputType::Constant(output.size(), eps);

		return -(target.cwiseProduct((output + epsv).log()) + (ones - target).cwiseProduct((ones + epsv - output).log()));
	}

	OutputType derivative(const OutputType& output, const OutputType& target) const
	{
		return output - target;
	}
};


template<> class BinaryCrossEntropyLoss<double>
{
public:
	double operator()(const double& output, const double& target) const
	{
		static const double eps = 1E-10;
		return -(target * log(output + eps) + (1. - target) * log(1. + eps - output));
	}

	double derivative(const double& output, const double& target) const
	{
		return output - target;
	}
};

