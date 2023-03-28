#pragma once

#include <Eigen/Eigen>

#include "GeneralizedLinearModel.h"
#include "SimpleLinearRegressionSolvers.h"

namespace GLM
{

	class SimpleLinearRegression : public GeneralizedLinearModel<double, double, double, SLRS::SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd>
	{
	public:
		typedef GeneralizedLinearModel<double, double, double, SLRS::SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd> BaseType;

		SimpleLinearRegression() : BaseType(1, 1)
		{
		}

		double Predict(const double& input) override
		{
			return BaseType::W * input + BaseType::b;
		}
	};

	// to not be confused with the general case, this corresponds to a bunch of simple linear regressions, even if multivariable
	template<typename InputType = Eigen::VectorXd> class MultivariateSimpleLinearRegression : public GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::MatrixXd, SLRS::SimpleLinearRegressionSolver<InputType>, Eigen::MatrixXd>
	{
	public:
		typedef GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::MatrixXd, SLRS::SimpleLinearRegressionSolver<InputType>, Eigen::MatrixXd> BaseType;

		MultivariateSimpleLinearRegression(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			BaseType::solver.Initialize(szi, szo);
			BaseType::W = Eigen::MatrixXd::Zero(szo, szi);
			BaseType::b = Eigen::VectorXd::Zero(szo);
		}

		Eigen::VectorXd Predict(const InputType& input) override
		{
			if (input.size() == 1)
				return BaseType::W * input(0) + BaseType::b;

			return BaseType::W.cwiseProduct(input) + BaseType::b;
		}

		ActivationFunctions::IdentityFunction<Eigen::VectorXd> activationFunction;
	};


	template<> class MultivariateSimpleLinearRegression<double> : public GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::MatrixXd, SLRS::SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::RowVectorXd, Eigen::MatrixXd>
	{
	public:
		typedef GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::MatrixXd, SLRS::SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::RowVectorXd, Eigen::MatrixXd> BaseType;

		MultivariateSimpleLinearRegression(int szi = 1, int szo = 1) : BaseType(szi, szo)
		{
		}
	};

}




