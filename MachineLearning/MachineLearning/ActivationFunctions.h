#pragma once

#include <cmath>


#include <Eigen/eigen>
//#include <unsupported/Eigen/MatrixFunctions>

template<typename InputOutputType = Eigen::VectorXd> class IdentityFunction
{
public:
	const InputOutputType& operator()(const InputOutputType& input) const
	{
		return input;
	}

	const InputOutputType derivative(const InputOutputType& input) const
	{
		return InputOutputType::Ones(input.size());
	}
};

template<> class IdentityFunction<double>
{
public:
	const double& operator()(const double& input) const
	{
		return input;
	}

	const double derivative(const double& input) const
	{
		return 1;
	}
};


template<typename InputOutputType = Eigen::VectorXd, typename WeightsType = InputOutputType> class SigmoidFunction
{
public:
	SigmoidFunction(int size = 1)
	{
		beta0 = InputOutputType::Zero(size);
		beta = InputOutputType::Ones(size);
	}

	void setParams(const WeightsType& b0, const WeightsType& b)
	{
		beta0 = b0;
		beta = b;
	}

	const InputOutputType operator()(const InputOutputType& input) const
	{
		InputOutputType v(input.rows(), input.cols());
		
		for (int i = 0; i < input.rows(); ++i)
			for (int j = 0; j < input.cols(); ++j)
				v(i, j) = exp(-(beta(i, j) * input(i , j) + beta0(i,j)));

		return (InputOutputType::Ones(input.rows(), input.cols()) + v).cwiseInverse();
	}

	const InputOutputType derivative(const InputOutputType& input) const
	{
		const InputOutputType fx = operator()(input);

		return fx.cwiseProduct(InputOutputType::Ones(fx.rows(), fx.cols()) - fx);
	}

protected:
	WeightsType beta0;
	WeightsType beta;
};

template<> class SigmoidFunction<double, double>
{
public:
	SigmoidFunction()
		: beta0(0), beta(1)
	{
	}

	void setParams(double b0, double b)
	{
		beta0 = b0;
		beta = b;
	}

	const double operator()(const double& input) const
	{
		return 1. / (1. + exp(-(input * beta + beta0)));
	}

	const double derivative(const double& input) const
	{
		const double fx = operator()(input);

		return fx * (1. - fx);
	}

protected:
	double beta0;
	double beta;
};



template<typename InputOutputType = Eigen::VectorXd> class TanhFunction
{
public:
	TanhFunction(int size = 1)
	{
	}

	const InputOutputType operator()(const InputOutputType& input) const
	{
		const SigmoidFunction<InputOutputType, InputOutputType> sigmoid(static_cast<int>(input.size()));

		return 2. * sigmoid(2. * input) - InputOutputType::Ones(input.rows(), input.cols());
	}

	const InputOutputType derivative(const InputOutputType& input) const
	{
		const InputOutputType fx = operator()(input);

		return InputOutputType::Ones(fx.rows(), fx.cols()) - fx.cwiseProduct(fx);
	}
};

template<> class TanhFunction<double>
{
public:
	TanhFunction()
	{
	}

	const double operator()(const double& input) const
	{
		return tanh(input);
	}

	const double derivative(const double& input) const
	{
		const double fx = operator()(input);

		return 1. - fx * fx;
	}
};

template<typename InputOutputType = Eigen::VectorXd> class SoftplusFunction
{
public:
	SoftplusFunction(int size = 1)
	{
	}

	const InputOutputType operator()(const InputOutputType& input) const
	{
		InputOutputType v(input.rows(), input.cols());

		for (int i = 0; i < input.rows(); ++i)
			for (int j = 0; j < input.cols(); ++j)
				v(i, j) = exp(input(i, j)) + 1;

		return v;
	}

	const InputOutputType derivative(const InputOutputType& input) const
	{
		const InputOutputType fx = operator()(-input);

		return fx.cwiseInverse();
	}
};

template<> class SoftplusFunction<double>
{
public:
	SoftplusFunction()
	{
	}

	const double operator()(const double& input) const
	{
		return 1. + exp(input);
	}

	const double derivative(const double& input) const
	{
		const double fx = operator()(-input);

		return 1. / fx;
	}
};



template<typename InputOutputType = Eigen::VectorXd> class RELUFunction
{
public:
	RELUFunction()
	{
	}

	const InputOutputType operator()(const InputOutputType& input) const
	{
		InputOutputType out = input;

		for (unsigned int i = 0; i < out.size(); ++i)
			if (out(i) < 0) out(i) = 0;
			
		return out;
	}

	const InputOutputType derivative(const InputOutputType& input) const
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

	const double operator()(const double& input) const
	{
		return (input < 0) ? 0 : input;
	}

	const double derivative(const double& input) const
	{
		return (input < 0) ? 0 : 1;
	}
};


template<typename InputOutputType = Eigen::VectorXd> class LeakyRELUFunction
{
public:
	LeakyRELUFunction()
	{
	}

	void setParams(double a)
	{
		alpha = a;
	}

	const InputOutputType operator()(const InputOutputType& input) const
	{
		InputOutputType out = input;

		for (unsigned int i = 0; i < out.size(); ++i)
			out(i) *= ((out(i) < 0) ? alpha : 1.);

		return out;
	}

	const InputOutputType derivative(const InputOutputType& input) const
	{
		InputOutputType out = input;

		for (unsigned int i = 0; i < out.size(); ++i)
			out(i) = (out(i) < 0) ? alpha : 1.;

		return out;
	}

protected:
	double alpha = 0.01;
};

template<> class LeakyRELUFunction<double>
{
public:
	LeakyRELUFunction()
	{
	}

	void setParams(double a)
	{
		alpha = a;
	}

	const double operator()(const double& input) const
	{
		return ((input < 0) ? alpha : 1.) * input;
	}

	const double derivative(const double& input) const
	{
		return (input < 0) ? alpha : 1.;
	}

protected:
	double alpha = 0.01;
};

template<typename InputOutputType = Eigen::VectorXd> class SoftmaxFunction
{
public:
	const InputOutputType& operator()(const InputOutputType& input) const
	{
		InputOutputType output;
		output.resize(input.size());

		double sum = 0;
		for (int i = 0; i < input.size(); ++i)
		{
			const double v = exp(input(i));
			output(i) = v;
			sum += v;
		}

		return output / sum;
	}

	const Eigen::MatrixXd derivative(const InputOutputType& input) const
	{
		Eigen::MatrixXd output;
		output.resize(input.size(), input.size());

		InputOutputType fx = operator()(input);

		for (int i = 0; i < input.size(); ++i)
			for (int j = 0; j < input.size(); ++j)
				output(i, j) = fx(i) * (((i == j) ? 1. : 0.) - fx(j));

		return output;
	}
};
