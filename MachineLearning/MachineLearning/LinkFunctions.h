#pragma once


#include <math.h>

template<typename InputOutputType = Eigen::VectorXd> class IdentityFunction
{
public:
	const InputOutputType& operator()(const InputOutputType& input)
	{
		return input;
	}

	const InputOutputType getGradient(const InputOutputType& input)
	{
		return InputOutputType::Ones(input.size());
	}
};

template<> class IdentityFunction<double>
{
public:
	const double& operator()(const double& input)
	{
		return input;
	}

	const double getGradient(const double& input)
	{
		return 1;
	}
};


template<typename InputOutputType, typename WeightsType> class LogisticFunction
{
public:
	LogisticFunction(int size = 1)
	{
		beta0 = InputOutputType::Zeros(size);
		beta = InputOutputType::Ones(size);
	}

	void setParams(const WeightsType& b0, const WeightsType& b)
	{
		beta0 = b0;
		beta = b;
	}

	const InputOutputType operator()(const InputOutputType& input)
	{
		return (1. + (-(beta.cwiseProduct(input) + beta0)).exp()).cwiseInverse();
	}

	const InputOutputType derivative(const InputOutputType& input)
	{
		const InputOutputType fx = operator()(input);

		return fx.cwiseProduct(InputOutputType::Ones(fx.size()) - fx);
	}

protected:
	WeightsType beta0;
	WeightsType beta;
};

template<> class LogisticFunction<double, double>
{
public:
	LogisticFunction()
		: beta0(0), beta(1)
	{
	}

	void setParams(double b0, double b)
	{
		beta0 = b0;
		beta = b;
	}

	const double operator()(const double& input)
	{
		return 1. / (1. + exp(-(input * beta + beta0)));
	}

	const double derivative(const double& input)
	{
		const double fx = operator()(input);

		return fx * (1. - fx);
	}

protected:
	double beta0;
	double beta;
};

template<typename InputOutputType> class RELUFunction
{
public:
	RELUFunction()
	{
	}

	const InputOutputType operator()(const InputOutputType& input)
	{
		InputOutputType out = input;

		for (unsigned int i = 0; i < out.size(); ++i)
			if (out(i) < 0) out(i) = 0;
			
		return out;
	}

	const InputOutputType derivative(const InputOutputType& input)
	{
		InputOutputType out = input;

		for (unsigned int i = 0; i < out.size(); ++i)
			if (out(i) < 0) out(i) = 0;
			else out(i) = 1;

		return out;
	}
};

template<> class RELUFunction<double>
{
public:
	RELUFunction()
	{
	}

	const double operator()(const double& input)
	{
		return (input < 0) ? 0 : input;
	}

	const double derivative(const double& input)
	{
		return (input < 0) ? 0 : 1;
	}
};
