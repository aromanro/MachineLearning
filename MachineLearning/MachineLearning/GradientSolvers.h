#pragma once

#include "LinkFunctions.h"
#include "CostFunctions.h"

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::VectorXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType/*, BatchOutputType*/>>
class GradientDescentSolver
{
public:
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
		BatchOutputType lossLinkGrad = lossFunction.derivative(pred, output) * linkFunction.derivative(linpred);

		b -= alpha * lossLinkGrad;
		w -= alpha * lossLinkGrad * input;
	}

protected:
	double alpha = 0.01;

	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;
};
