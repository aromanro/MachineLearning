#pragma once
#include "GeneralizedLinearModel.h"


namespace SGD
{

	typedef SGD::GradientDescentSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>> SoftmaxRegressionGradientDescentSolver;
	typedef SGD::MomentumSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>> SoftmaxRegressionMomentumSolver;
	typedef SGD::AdaGradSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>> SoftmaxRegressionAdaGradSolver;
	typedef SGD::RMSPropSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>> SoftmaxRegressionRMSPropSolver;
	typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>> SoftmaxRegressionAdamSolver;

}

namespace GLM
{

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::SoftmaxRegressionAdamSolver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType> class SoftmaxRegression
		: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
	{
	public:
		typedef GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType> BaseType;

		SoftmaxRegression(int szi = 1, int szo = 1)
			: BaseType(szi, szo)
		{
			// W was initialized to some values between -1 and 1, translate them to values between 0 and 1
			//BaseType::W += WeightsType::Ones(BaseType::W.rows(), BaseType::W.cols());
			//BaseType::W *= 0.5;
		}
	};

}

