#pragma once

#include "GeneralizedLinearModel.h"
#include "Solvers.h"

class SimpleLinearRegression : public GeneralizedLinearModel<double, double, double, SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd>
{
public:
	typedef GeneralizedLinearModel<double, double, double, SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>, Eigen::RowVectorXd> baseType;

	SimpleLinearRegression() : baseType(1, 1)
	{
	}

	const double Predict(const double& input) override
	{
		return baseType::W * input + baseType::b;
	}

	void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput) override
	{
		baseType::solver.AddBatch(batchInput, batchOutput);

		baseType::solver.getWeightsAndBias(baseType::W, baseType::b);
	}
};

// to not be confused with the general case, this corresponds to a bunch of simple linear regressions, even if multivariable
template<typename InputType = Eigen::VectorXd> class MultivariateSimpleLinearRegression : public GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd>
{
public:
	typedef GeneralizedLinearModel<InputType, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<>, Eigen::MatrixXd> baseType;

	MultivariateSimpleLinearRegression(int szi = 1, int szo = 1)
	{
		Initialize(szi, szo);
	}

	void Initialize(int szi = 1, int szo = 1)
	{
		baseType::solver.Initialize(szi, szo);
		baseType::W = Eigen::VectorXd::Zero(szo);
		baseType::b = Eigen::VectorXd::Zero(szo);
	}

	const Eigen::VectorXd Predict(const InputType& input) override
	{
		const Eigen::VectorXd linOut = baseType::W.cwiseProduct(input) + baseType::b;

		return linOut;
	}

	void AddBatch(const Eigen::MatrixXd& batchInput, const Eigen::MatrixXd& batchOutput) override
	{
		baseType::solver.AddBatch(batchInput, batchOutput);

		baseType::solver.getWeightsAndBias(baseType::W, baseType::b);
	}
};


template<> class MultivariateSimpleLinearRegression<double> : public GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::MatrixXd>
{
public:
	typedef GeneralizedLinearModel<double, Eigen::VectorXd, Eigen::VectorXd, SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::MatrixXd>, Eigen::MatrixXd> baseType;

	MultivariateSimpleLinearRegression(int szi = 1, int szo = 1) : baseType(szi, szo)
	{
	}

	const Eigen::VectorXd Predict(const double& input) override
	{
		const Eigen::VectorXd linOut = baseType::W * input + baseType::b;

		return linOut;
	}

	void AddBatch(const Eigen::MatrixXd& batchInput, const Eigen::MatrixXd& batchOutput) override
	{
		baseType::solver.AddBatch(batchInput, batchOutput);

		baseType::solver.getWeightsAndBias(baseType::W, baseType::b);
	}
};

