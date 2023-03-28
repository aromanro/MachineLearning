#pragma once
#include "GeneralizedLinearModel.h"


namespace SGD
{

	using SoftmaxRegressionGradientDescentSolver = SGD::GradientDescentSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>>;
	using SoftmaxRegressionMomentumSolver = SGD::MomentumSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>>;
	using SoftmaxRegressionAdaGradSolver = SGD::AdaGradSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>>;
	using SoftmaxRegressionRMSPropSolver = SGD::RMSPropSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>>;
	using SoftmaxRegressionAdamSolver = SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::SoftmaxFunction<Eigen::VectorXd>, LossFunctions::CrossEntropyLoss<Eigen::VectorXd>>;

}

namespace GLM
{

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::SoftmaxRegressionAdamSolver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType> class SoftmaxRegression
		: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
	{
	public:
		using BaseType = GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>;

		SoftmaxRegression(int szi = 1, int szo = 1)
			: BaseType(szi, szo)
		{
			// W was initialized to some values between -1 and 1, translate them to values between 0 and 1
			//BaseType::W += WeightsType::Ones(BaseType::W.rows(), BaseType::W.cols());
			//BaseType::W *= 0.5;
		}
	};

}

