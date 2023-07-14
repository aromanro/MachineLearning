#pragma once

#include <Eigen/Eigen>
//#include <unsupported/Eigen/MatrixFunctions>

namespace LossFunctions
{

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

			for (unsigned int i = 0; i < dif.cols(); ++i)
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
			static const double eps = 1E-10;
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
			static const double eps = 1E-10;
			OutputType d(output.size());

			for (int i = 0; i < output.size(); ++i)
				d(i) = 1. / (output(i) * (1. + eps - output(i)));

			return (output - target).cwiseProduct(d);
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
			static const double eps = 1E-10;
			return -(target * log(output + eps) + (1. - target) * log(1. + eps - output));
		}

		double derivative(const double& output, const double& target) const
		{
			static const double eps = 1E-8;

			return (output - target) / (output * (1. + eps - output));
		}

		static std::string getName()
		{
			return "BinaryCrossEntropy";
		}
	};


	template<typename OutputType = Eigen::VectorXd> class CrossEntropyLoss
	{
	public:
		OutputType operator()(const OutputType& output, const OutputType& target) const
		{
			static const double eps = 1E-10;

			OutputType l1(output.size());
			for (int i = 0; i < output.size(); ++i)
				l1(i) = log(output(i) + eps);
	
			return -target.cwiseProduct(l1);
		}

		OutputType derivative(const OutputType& output, const OutputType& target) const
		{
			static const double eps = 1E-8;

			OutputType res(output.size());
			for (int i = 0; i < output.size(); ++i)
				res(i) = -target(i) / (output(i) + eps);

			return res;
		}

		static std::string getName()
		{
			return "CrossEntropy";
		}
	};


}


