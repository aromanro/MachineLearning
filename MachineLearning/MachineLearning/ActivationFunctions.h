#pragma once

#include <cmath>


#include <Eigen/eigen>
//#include <unsupported/Eigen/MatrixFunctions>

namespace ActivationFunctions
{

	template<typename InputOutputType = Eigen::VectorXd> class IdentityFunction
	{
	public:
		IdentityFunction(int size = 1)
		{
		}

		const InputOutputType& operator()(const InputOutputType& input) const
		{
			return input;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			return InputOutputType::Ones(input.size());
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}
	};

	template<> class IdentityFunction<double>
	{
	public:
		IdentityFunction(int size = 1)
		{
		}

		const double& operator()(const double& input) const
		{
			return input;
		}

		const double derivative(const double& input) const
		{
			return 1;
		}

		static bool isDerivativeMatrix()
		{
			return false;
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

		InputOutputType operator()(const InputOutputType& input)
		{
			if (beta0.size() != input.size() || beta.size() != input.size())
			{
				beta0 = InputOutputType::Zero(input.size());
				beta = InputOutputType::Ones(input.size());
			}

			InputOutputType v(input.size());

			for (int i = 0; i < input.size(); ++i)
				v(i) = exp(-(beta(i) * input(i) + beta0(i)));

			return (InputOutputType::Ones(input.size()) + v).cwiseInverse();
		}

		InputOutputType derivative(const InputOutputType& input)
		{
			const InputOutputType fx = operator()(input);

			return fx.cwiseProduct(InputOutputType::Ones(fx.rows(), fx.cols()) - fx);
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}

	protected:
		InputOutputType beta0;
		InputOutputType beta;
	};

	template<> class SigmoidFunction<double, double>
	{
	public:
		SigmoidFunction(int size = 1)
			: beta0(0), beta(1)
		{
		}

		void setParams(double b0, double b)
		{
			beta0 = b0;
			beta = b;
		}

		double operator()(const double& input)
		{
			return 1. / (1. + exp(-(input * beta + beta0)));
		}

		double derivative(const double& input)
		{
			const double fx = operator()(input);

			return fx * (1. - fx);
		}

		static bool isDerivativeMatrix()
		{
			return false;
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

		InputOutputType operator()(const InputOutputType& input) const
		{
			const SigmoidFunction<InputOutputType, InputOutputType> sigmoid(static_cast<int>(input.size()));

			return 2. * sigmoid(2. * input) - InputOutputType::Ones(input.rows(), input.cols());
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			const InputOutputType fx = operator()(input);

			return InputOutputType::Ones(fx.rows(), fx.cols()) - fx.cwiseProduct(fx);
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}
	};

	template<> class TanhFunction<double>
	{
	public:
		TanhFunction(int size = 1)
		{
		}

		double operator()(const double& input) const
		{
			return tanh(input);
		}

		double derivative(const double& input) const
		{
			const double fx = operator()(input);

			return 1. - fx * fx;
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}
	};

	template<typename InputOutputType = Eigen::VectorXd> class SoftplusFunction
	{
	public:
		SoftplusFunction(int size = 1)
		{
		}

		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType v(input.size());

			for (int i = 0; i < input.size(); ++i)			
				v(i) = exp(input(i)) + 1;

			return v;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			const InputOutputType fx = operator()(-input);

			return fx.cwiseInverse();
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}
	};

	template<> class SoftplusFunction<double>
	{
	public:
		SoftplusFunction(int size = 1)
		{
		}

		double operator()(const double& input) const
		{
			return 1. + exp(input);
		}

		double derivative(const double& input) const
		{
			const double fx = operator()(-input);

			return 1. / fx;
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}
	};



	template<typename InputOutputType = Eigen::VectorXd> class RELUFunction
	{
	public:
		RELUFunction(int size = 1)
		{
		}

		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType out = input;

			for (unsigned int i = 0; i < out.size(); ++i)
				if (out(i) < 0) out(i) = 0;

			return out;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			InputOutputType out = input;

			for (unsigned int i = 0; i < out.size(); ++i)
				if (out(i) < 0) out(i) = 0;
				else out(i) = 1;

			return out;
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}
	};

	template<> class RELUFunction<double>
	{
	public:
		RELUFunction(int size = 1)
		{
		}

		double operator()(const double& input) const
		{
			return (input < 0) ? 0 : input;
		}

		double derivative(const double& input) const
		{
			return (input < 0) ? 0 : 1;
		}

		bool isDerivativeMatrix() const
		{
			return false;
		}
	};


	template<typename InputOutputType = Eigen::VectorXd> class LeakyRELUFunction
	{
	public:
		LeakyRELUFunction(int size = 1)
		{
		}

		void setParams(double a)
		{
			alpha = a;
		}

		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType out = input;

			for (unsigned int i = 0; i < out.size(); ++i)
				out(i) *= ((out(i) < 0) ? alpha : 1.);

			return out;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			InputOutputType out = input;

			for (unsigned int i = 0; i < out.size(); ++i)
				out(i) = (out(i) < 0) ? alpha : 1.;

			return out;
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}

	protected:
		double alpha = 0.01;
	};

	template<> class LeakyRELUFunction<double>
	{
	public:
		LeakyRELUFunction(int size = 1)
		{
		}

		void setParams(double a)
		{
			alpha = a;
		}

		double operator()(const double& input) const
		{
			return ((input < 0) ? alpha : 1.) * input;
		}

		double derivative(const double& input) const
		{
			return (input < 0) ? alpha : 1.;
		}

		static bool isDerivativeMatrix()
		{
			return false;
		}

	protected:
		double alpha = 0.01;
	};

	template<typename InputOutputType = Eigen::VectorXd> class SoftmaxFunction
	{
	public:
		SoftmaxFunction(int size = 1)
		{
		}

		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType output(input.size());

			const double m = input.maxCoeff();

			double sum = 0;
			for (int i = 0; i < input.size(); ++i)
			{
				const double v = exp(input(i) - m);
				output(i) = v;
				sum += v;
			}

			return output / sum;
		}

		Eigen::MatrixXd derivative(const InputOutputType& input) const
		{
			Eigen::MatrixXd output(input.size(), input.size());

			InputOutputType fx = operator()(input);

			for (int j = 0; j < input.size(); ++j)
				for (int i = 0; i < input.size(); ++i)
					output(i, j) = fx(i) * (((i == j) ? 1. : 0.) - fx(j));

			return output;
		}

		static bool isDerivativeMatrix()
		{
			return true;
		}
	};

}

