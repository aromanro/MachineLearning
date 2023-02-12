#pragma once

#include "LinkFunctions.h"
#include "CostFunctions.h"

template<typename InputType, typename OutputType, typename WeigthsType, class Solver, class BatchInputType, class BatchOutputType = BatchInputType, class LinkFunction = IdentityFunction<OutputType>, class CostFunction = L2Cost<OutputType, BatchOutputType>> 
class GeneralizedLinearModel
{
public:
	GeneralizedLinearModel(int sz = 1)
	{
		Initialize(sz);
	}

	void Initialize(int sz = 1)
	{
		solver.Initialize(sz);
	}

	virtual const OutputType Predict(const InputType& input)
	{
		return linkFunc(W * input + b);
	}

	void AddSample(const InputType& input, const OutputType& output)
	{
		solver.AddSample(input, output);

		W = solver.getWeights();
		b = solver.getBias();
	}

	void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);

		W = solver.getWeights();
		b = solver.getBias();
	}

protected:
	LinkFunction linkFunc;

	WeigthsType W;
	WeigthsType b;

	Solver solver;
};

