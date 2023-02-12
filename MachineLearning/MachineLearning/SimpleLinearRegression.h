#pragma once

#include "GeneralizedLinearModel.h"
#include "Solvers.h"

typedef GeneralizedLinearModel<double, double, double, SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd> SimpleLinearRegression;

// to not be confused with the general case, this corresponds to a bunch of simple linear regressions, even if multivariable
template<typename InputType = Eigen::VectorXd> class MultivariateSimpleLinearRegression : public GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd>
{
public:
	typedef GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd> baseType;

	MultivariateSimpleLinearRegression(int sz = 1) : GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd>(sz)
	{
	}

	const Eigen::VectorXd& Predict(const InputType& input) override
	{
		return baseType::linkFunc((baseType::W.cwiseProduct(input) + baseType::b).eval());
	}
};


template<> class MultivariateSimpleLinearRegression<double> : public GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::MatrixXd>
{
public:
	typedef GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::MatrixXd> baseType;

	MultivariateSimpleLinearRegression(int sz = 1) : GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::MatrixXd>(sz)
	{
	}

	const Eigen::VectorXd& Predict(const double& input) override
	{
		return baseType::linkFunc((baseType::W * input + baseType::b).eval());
	}
};

