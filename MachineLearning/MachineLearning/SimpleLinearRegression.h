#pragma once

#include <Eigen/eigen>

#include "GeneralizedLinearModel.h"
#include "SimpleLinearRegressionSolvers.h"

class SimpleLinearRegression : public GeneralizedLinearModel<double, double, double, SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd>
{
public:
	typedef GeneralizedLinearModel<double, double, double, SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd> BaseType;

	SimpleLinearRegression() : BaseType(1, 1)
	{
	}

	double Predict(const double& input) const override
	{
		return BaseType::W * input + BaseType::b;
	}
};

// to not be confused with the general case, this corresponds to a bunch of simple linear regressions, even if multivariable
template<typename InputType = Eigen::VectorXd> class MultivariateSimpleLinearRegression : public GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::MatrixXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd>
{
public:
	typedef GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::MatrixXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd> BaseType;

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

	Eigen::VectorXd Predict(const InputType& input) const override
	{
		return BaseType::W.cwiseProduct(input) + BaseType::b;
	}

	IdentityFunction<Eigen::VectorXd> activationFunction;
};


template<> class MultivariateSimpleLinearRegression<double> : public GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::MatrixXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::RowVectorXd>
{
public:
	typedef GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::MatrixXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::RowVectorXd> BaseType;

	MultivariateSimpleLinearRegression(int szi = 1, int szo = 1) : BaseType(szi, szo)
	{
	}
};



