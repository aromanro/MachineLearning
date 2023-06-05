#pragma once

#include <cmath>


#include <Eigen/Eigen>
//#include <unsupported/Eigen/MatrixFunctions>

namespace ActivationFunctions
{
	template <typename InputOutputType = Eigen::VectorXd> class BaseIdentity
	{
	public:
		BaseIdentity(int size = 1)
		{
		}

		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "Identity";
		}
	};

	template<typename InputOutputType = Eigen::VectorXd> class IdentityFunction : public BaseIdentity<InputOutputType>
	{
	public:
		const InputOutputType& operator()(const InputOutputType& input) const
		{
			return input;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			return InputOutputType::Ones(input.size());
		}
	};

	template<> class IdentityFunction<double> : public BaseIdentity<double>
	{
	public:
		const double& operator()(const double& input) const
		{
			return input;
		}

		double derivative(const double& input) const
		{
			return 1;
		}
	};

	template<typename InputOutputType = Eigen::VectorXd> class BaseSigmoid
	{
	public:
		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "Sigmoid";
		}

	protected:
		InputOutputType beta0;
		InputOutputType beta;
	};

	template<typename InputOutputType = Eigen::VectorXd, typename WeightsType = InputOutputType> class SigmoidFunction : public BaseSigmoid<InputOutputType>
	{
	public:
		using BaseType = BaseSigmoid<InputOutputType>;

		SigmoidFunction(int size = 1)
		{
			BaseType::beta0 = InputOutputType::Zero(size);
			BaseType::beta = InputOutputType::Ones(size);
		}

		void setParams(const WeightsType& b0, const WeightsType& b)
		{
			BaseType::beta0 = b0;
			BaseType::beta = b;
		}

		InputOutputType operator()(const InputOutputType& input)
		{
			if (BaseType::beta0.size() != input.size() || BaseType::beta.size() != input.size())
			{
				BaseType::beta0 = InputOutputType::Zero(input.size());
				BaseType::beta = InputOutputType::Ones(input.size());
			}

			InputOutputType v(input.size());

			for (int i = 0; i < input.size(); ++i)
				v(i) = exp(-(BaseType::beta(i) * input(i) + BaseType::beta0(i)));

			return (InputOutputType::Ones(input.size()) + v).cwiseInverse();
		}

		InputOutputType derivative(const InputOutputType& input)
		{
			const InputOutputType fx = operator()(input);

			return fx.cwiseProduct(InputOutputType::Ones(fx.rows(), fx.cols()) - fx);
		}
	};

	template<> class SigmoidFunction<double, double> : public BaseSigmoid<double>
	{
	public:
		using BaseType = BaseSigmoid<double>;

		void setParams(double b0, double b)
		{
			BaseType::beta0 = b0;
			BaseType::beta = b;
		}

		double operator()(const double& input)
		{
			return 1. / (1. + exp(-(input * BaseType::beta + BaseType::beta0)));
		}

		double derivative(const double& input)
		{
			const double fx = operator()(input);

			return fx * (1. - fx);
		}
	};

	template<typename InputOutputType = Eigen::VectorXd> class BaseTanh
	{
	public:
		BaseTanh(int size = 1)
		{
		}

		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "Tanh";
		}
	};


	template<typename InputOutputType = Eigen::VectorXd> class TanhFunction : public BaseTanh<InputOutputType>
	{
	public:
		InputOutputType operator()(const InputOutputType& input) const
		{
			SigmoidFunction<InputOutputType, InputOutputType> sigmoid(static_cast<int>(input.size()));

			return 2. * sigmoid(2. * input) - InputOutputType::Ones(input.rows(), input.cols());
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			const InputOutputType fx = operator()(input);

			return InputOutputType::Ones(fx.rows(), fx.cols()) - fx.cwiseProduct(fx);
		}
	};

	template<> class TanhFunction<double> : public BaseTanh<double>
	{
	public:
		double operator()(const double& input) const
		{
			return tanh(input);
		}

		double derivative(const double& input) const
		{
			const double fx = operator()(input);

			return 1. - fx * fx;
		}
	};


	template<typename InputOutputType = Eigen::VectorXd> class BaseSoftplus
	{
	public:
		BaseSoftplus(int size = 1)
		{
		}

		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "Softplus";
		}
	};

	template<typename InputOutputType = Eigen::VectorXd> class SoftplusFunction : public BaseSoftplus<InputOutputType>
	{
	public:
		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType v(input.size());

			for (int i = 0; i < input.size(); ++i)			
				v(i) = log(exp(input(i)) + 1);

			return v;
		}

		InputOutputType derivative(const InputOutputType& input)
		{
			return sigmoid(input);
		}

	private:
		SigmoidFunction<InputOutputType, InputOutputType> sigmoid;
	};

	template<> class SoftplusFunction<double> : public BaseSoftplus<double>
	{
	public:
		double operator()(const double& input) const
		{
			return log(1. + exp(input));
		}

		double derivative(const double& input)
		{
			return sigmoid(input);
		}

	private:
		SigmoidFunction<double, double> sigmoid;
	};


	template<typename InputOutputType = Eigen::VectorXd> class BaseRELU
	{
	public:
		BaseRELU(int size = 1)
		{
		}

		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "RELU";
		}
	};

	template<typename InputOutputType = Eigen::VectorXd> class RELUFunction : public BaseRELU<InputOutputType>
	{
	public:
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
	};

	template<> class RELUFunction<double> : public BaseRELU<double>
	{
	public:
		double operator()(const double& input) const
		{
			return (input < 0) ? 0 : input;
		}

		double derivative(const double& input) const
		{
			return (input < 0) ? 0 : 1;
		}
	};


	template<typename InputOutputType = Eigen::VectorXd> class BaseLeakyRELU
	{
	public:
		BaseLeakyRELU(int size = 1)
		{
		}
		
		void setParams(double a)
		{
			alpha = a;
		}

		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "LeakyRELU";
		}

	protected:
		double alpha = 0.01;
	};


	template<typename InputOutputType = Eigen::VectorXd> class LeakyRELUFunction : public BaseLeakyRELU<InputOutputType>
	{
	public:
		using BaseType = BaseLeakyRELU<InputOutputType>;

		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType out = input;

			for (unsigned int i = 0; i < out.size(); ++i)
				out(i) *= ((out(i) < 0) ? BaseType::alpha : 1.);

			return out;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			InputOutputType out = input;

			for (unsigned int i = 0; i < out.size(); ++i)
				out(i) = (out(i) < 0) ? BaseType::alpha : 1.;

			return out;
		}
	};

	template<> class LeakyRELUFunction<double> : public BaseLeakyRELU<double>
	{
	public:
		using BaseType = BaseLeakyRELU<double>;

		double operator()(const double& input) const
		{
			return ((input < 0) ? BaseType::alpha : 1.) * input;
		}

		double derivative(const double& input) const
		{
			return (input < 0) ? BaseType::alpha : 1.;
		}
	};


	template<typename InputOutputType = Eigen::VectorXd> class BaseSELU
	{
	public:
		BaseSELU(int size = 1)
		{
		}

		void setParams(double a, double b)
		{
			alpha = a;
			scale = b;
		}

		static bool isDerivativeJacobianMatrix()
		{
			return false;
		}

		static std::string getName()
		{
			return "SELU";
		}

	protected:
		double alpha = 1.67326324;
		double scale = 1.05070098;
	};

	template<typename InputOutputType = Eigen::VectorXd> class SELUFunction : public BaseSELU<InputOutputType>
	{
	public:
		using BaseType = BaseSELU<InputOutputType>;

		InputOutputType operator()(const InputOutputType& input) const
		{
			InputOutputType out(input.size());

			for (unsigned int i = 0; i < out.size(); ++i)
				out(i) = BaseType::scale * ((input(i) < 0) ? BaseType::alpha * (exp(input(i)) - 1.) : input(i));

			return out;
		}

		InputOutputType derivative(const InputOutputType& input) const
		{
			InputOutputType out(input.size());

			for (unsigned int i = 0; i < out.size(); ++i)
				out(i) = BaseType::scale * ((input(i) < 0) ? BaseType::alpha * exp(input(i)) : 1.);

			return out;
		}
	};

	template<> class SELUFunction<double> : public BaseSELU<double>
	{
	public:
		using BaseType = BaseSELU<double>;

		double operator()(const double& input) const
		{
			return BaseType::scale * ((input < 0) ? BaseType::alpha * (exp(input) - 1.) : input);
		}

		double derivative(const double& input) const
		{
			return BaseType::scale * ((input < 0) ? BaseType::alpha * exp(input) : 1.);
		}
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

		static bool isDerivativeJacobianMatrix()
		{
			return true;
		}

		static std::string getName()
		{
			return "Softmax";
		}
	};

}

