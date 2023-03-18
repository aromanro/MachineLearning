#pragma once

#include "ActivationFunctions.h"
#include "CostFunctions.h"
#include "GradientSolvers.h"
#include "GeneralizedLinearModel.h"

// it's just a generalized linear model, represents a bunch of neurons to be put in a layer
// but to simplify things, I won't allow anything than defaults with Eigen matrices/vectors for implementation

// the only thing that can be changed is the solver, either to an entirely different one 
// or just by specifying the activation and/or cost functions

template<class Solver = AdamSolver<>>
class NeuralSublayer : public GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Solver>
{
public:
	typedef GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Solver> baseType;

	NeuralSublayer(int szi = 1, int szo = 1) : baseType(szi, szo)
	{
	}

	void setLastLayer(bool last = true)
	{
		baseType::solver.lastLayer = last;
	}

	bool getLastLayer() const
	{
		return baseType::solver.lastLayer;
	}

	void setFirstLayer(bool first = true)
	{
		baseType::solver.firstLayer = first;
	}

	bool getFirstLayer() const
	{
		return baseType::solver.firstLayer;
	}

	void setParams(const std::vector<double>& params)
	{
		baseType::solver.setParams(params);
	}
};

