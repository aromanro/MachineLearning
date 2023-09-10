#pragma once

#include "GeneralizedLinearModel.h"

namespace LossFunctions
{
	// a special kind of generalized linear regression, with a sigmoid function as the link function and a logistic loss (which is the cross entropy loss for a special case of 'target' values being either 0 or 1)

	template<typename T> class LogLoss : public BinaryCrossEntropyLoss<T>
	{
	};

}

namespace SGD
{

	using LogisticRegressionGradientDescentSolver = SGD::GradientDescentSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LossFunctions::LogLoss<Eigen::VectorXd>>;
	using LogisticRegressionMomentumSolver = SGD::MomentumSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LossFunctions::LogLoss<Eigen::VectorXd>>;
	using LogisticRegressionAdaGradSolver = SGD::AdaGradSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LossFunctions::LogLoss<Eigen::VectorXd>>;
	using LogisticRegressionRMSPropSolver = SGD::RMSPropSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LossFunctions::LogLoss<Eigen::VectorXd>>;
	using LogisticRegressionAdamSolver = SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LossFunctions::LogLoss<Eigen::VectorXd>>;
	using LogisticRegressionAdamWSolver = SGD::AdamWSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SigmoidFunction<Eigen::VectorXd, Eigen::MatrixXd>, LossFunctions::LogLoss<Eigen::VectorXd>>;

}

namespace GLM
{

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::LogisticRegressionAdamWSolver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType> class LogisticRegression
		: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
	{
	public:
		using BaseType = GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>;

		LogisticRegression(int szi = 1, int szo = 1)
			: BaseType(szi, szo)
		{
			// W was initialized to some values between -1 and 1, translate them to values between 0 and 1
			//BaseType::W += WeightsType::Ones(BaseType::W.rows(), BaseType::W.cols());
			//BaseType::W *= 0.5;
		}
	};

}


