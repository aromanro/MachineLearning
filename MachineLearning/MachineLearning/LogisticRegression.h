#pragma once

#include "GeneralizedLinearModel.h"

// a special kind of generalized linear regression, with a sigmoid function as the link function and a logistic loss (which is the cross entropy loss for a special case of 'target' values being either 0 or 1)

template<typename T> class LogLoss : public BinaryCrossEntropyLoss<T>
{
};

typedef GradientDescentSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LogLoss<Eigen::VectorXd>> LogisticRegressionGradientDescentSolver;
typedef MomentumSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LogLoss<Eigen::VectorXd>> LogisticRegressionMomentumSolver;
typedef AdaGradSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LogLoss<Eigen::VectorXd>> LogisticRegressionAdaGradSolver;
typedef RMSPropSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LogLoss<Eigen::VectorXd>> LogisticRegressionRMSPropSolver;
typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LogLoss<Eigen::VectorXd>> LogisticRegressionAdamSolver;


template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, class Solver = LogisticRegressionAdamSolver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = Eigen::MatrixXd> class LogisticRegression
	: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
{
public:
	typedef GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType> baseType;

	LogisticRegression(int szi = 1, int szo = 1)
		: baseType(szi, szo)
	{
		// W was initialized to some values between -1 and 1, translate them to values between 0 and 1
		//baseType::W += WeightsType::Ones(baseType::W.rows(), baseType::W.cols());
		//baseType::W *= 0.5;
	}
};

