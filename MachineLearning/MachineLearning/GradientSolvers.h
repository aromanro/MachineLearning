#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

#include "LinkFunctions.h"
#include "CostFunctions.h"

#include <iostream>

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType/*, BatchOutputType*/>>
class GradientDescentSolver
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
	}

	void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		assert(batchInput.cols() == batchOutput.cols());

		input = batchInput;
		output = batchOutput;
	}

	void setPrediction(const BatchOutputType& output)
	{
		pred = output;
	}

	void setLinearPrediction(const BatchOutputType& output) // before calling the link function
	{
		linpred = output;
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b) const
	{
		OutputType lossLinkGrad = OutputType::Zero(output.rows());
		
		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)).cwiseProduct(lossFunction.derivative(pred.col(c), output.col(c)));

		// clip it if necessary
		const double n = sqrt(lossLinkGrad.cwiseProduct(lossLinkGrad).sum());
		if (n > lim)
			lossLinkGrad *= lim / n;

		const double norm = 1. / input.cols();
		b -= alpha * norm * lossLinkGrad;

		WeightsType wAdj = WeightsType::Zero(w.rows(), w.cols());
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c).transpose();

		w -= alpha * norm * wAdj;
	}

protected:
	double alpha = 0.000001;
	double lim = 20.;

	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;
};
