#pragma once

#include <cmath>
#include <Eigen/Eigen>
//#include <unsupported/Eigen/MatrixFunctions>

namespace LossFunctions
{
	namespace Detail
	{
		inline constexpr double binaryCrossEntropyEpsilon = 1E-10;

		inline double BinaryCrossEntropyFromLogit(double logit, double target)
		{
			// Avoid overflow and cancellation, including confidently correct labels.
			return (logit >= 0. ? (1. - target) * logit : -target * logit)
				+ std::log1p(std::exp(-std::abs(logit)));
		}
	}

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

		static std::string getName()
		{
			return "L2";
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

		static std::string getName()
		{
			return "L2";
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

			for (Eigen::Index i = 0; i < dif.size(); ++i)
				dif(i) = dif(i) < 0 ? -1 : 1;

			return dif;
		}

		static std::string getName()
		{
			return "L1";
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
			return (output - target) < 0 ? -1 : 1;
		}

		static std::string getName()
		{
			return "L1";
		}
	};

	template<typename OutputType = Eigen::VectorXd> class BinaryCrossEntropyLoss
	{
	public:
		OutputType operator()(const OutputType& output, const OutputType& target) const
		{
			constexpr double eps = Detail::binaryCrossEntropyEpsilon;
			const OutputType ones = OutputType::Ones(output.size());

			OutputType l1(output.size());
			OutputType l2(output.size());

			for (int i = 0; i < output.size(); ++i)
			{
				l1(i) = log(output(i) + eps);
				l2(i) = log(1 + eps - output(i));
			}

			return -(target.cwiseProduct(l1) + (ones - target).cwiseProduct(l2));
		}

		OutputType derivative(const OutputType& output, const OutputType& target) const
		{
			constexpr double eps = Detail::binaryCrossEntropyEpsilon;
			OutputType d(output.size());

			for (int i = 0; i < output.size(); ++i)
				d(i) = -target(i) / (output(i) + eps) + (1. - target(i)) / (1. + eps - output(i));

			return d;
		}

		OutputType fromLogits(const OutputType& logits, const OutputType& target) const
		{
			OutputType result(logits.size());
			for (Eigen::Index i = 0; i < logits.size(); ++i)
				result(i) = Detail::BinaryCrossEntropyFromLogit(logits(i), target(i));
			return result;
		}

		static std::string getName()
		{
			return "BinaryCrossEntropy";
		}
	};


	template<> class BinaryCrossEntropyLoss<double>
	{
	public:
		double operator()(const double& output, const double& target) const
		{
			constexpr double eps = Detail::binaryCrossEntropyEpsilon;
			return -(target * log(output + eps) + (1. - target) * log(1. + eps - output));
		}

		double derivative(const double& output, const double& target) const
		{
			constexpr double eps = Detail::binaryCrossEntropyEpsilon;

			return -target / (output + eps) + (1. - target) / (1. + eps - output);
		}

		double fromLogits(double logit, double target) const
		{
			return Detail::BinaryCrossEntropyFromLogit(logit, target);
		}

		static std::string getName()
		{
			return "BinaryCrossEntropy";
		}
	};

	// The logistic-regression loss has the same semantics as binary cross-entropy.
	template<typename T> class LogLoss : public BinaryCrossEntropyLoss<T> {};


	template<typename OutputType = Eigen::VectorXd> class CrossEntropyLoss
	{
	public:
		OutputType operator()(const OutputType& output, const OutputType& target) const
		{
			OutputType l1(output.size());
			for (int i = 0; i < output.size(); ++i)
				l1(i) = log(output(i) + epsilon);
	
			return -target.cwiseProduct(l1);
		}

		OutputType derivative(const OutputType& output, const OutputType& target) const
		{
			OutputType res(output.size());
			for (int i = 0; i < output.size(); ++i)
				res(i) = -target(i) / (output(i) + epsilon);

			return res;
		}

		// The fused softmax loss stays finite even when a probability underflows.
		OutputType fromLogits(const OutputType& logits, const OutputType& target) const
		{
			if (logits.size() == 0) return logits;
			const OutputType shifted = logits.array() - logits.maxCoeff();
			const double logSumExp = std::log(shifted.array().exp().sum());
			return (target.array() * (logSumExp - shifted.array())).matrix();
		}

		static std::string getName()
		{
			return "CrossEntropy";
		}

	private:
		static constexpr double epsilon = 1E-10;
	};


}


