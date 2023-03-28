#pragma once

#include "ActivationFunctions.h"
#include "CostFunctions.h"
#include "GradientSolvers.h"
#include "GeneralizedLinearModel.h"

namespace NeuralNetworks
{

	// it's just a generalized linear model, represents a bunch of neurons to be put in a layer
	// but to simplify things, I won't allow anything than defaults with Eigen matrices/vectors for implementation

	// the only thing that can be changed is the solver, either to an entirely different one 
	// or just by specifying the activation and/or cost functions

	template<class Solver = SGD::AdamSolver<>>
	class NeuralSublayer : public GLM::GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Solver>
	{
	public:
		using BaseType = GLM::GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Solver>;

		NeuralSublayer(int szi = 1, int szo = 1) : BaseType(szi, szo)
		{
		}

		void setLastLayer(bool last = true)
		{
			BaseType::solver.lastLayer = last;
		}

		bool getLastLayer() const
		{
			return BaseType::solver.lastLayer;
		}

		void setFirstLayer(bool first = true)
		{
			BaseType::solver.firstLayer = first;
		}

		bool getFirstLayer() const
		{
			return BaseType::solver.firstLayer;
		}

		void setParams(const std::vector<double>& params)
		{
			BaseType::solver.setParams(params);
		}
	};

}


