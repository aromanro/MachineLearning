#pragma once

#include "LinkFunctions.h"
#include "CostFunctions.h"

template<typename InputType, typename OutputType, typename WeigthsType, class Solver, class BatchInputType, class BatchOutputType = BatchInputType, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType/*, BatchOutputType*/>>
class GeneralizedLinearModel
{
public:
	GeneralizedLinearModel(int szi = 1, int szo = 1)
	{
		Initialize(szi, szo);
	}

	void Initialize(int szi = 1, int szo = 1)
	{
		solver.Initialize(szi, szo);
		W = WeigthsType::Zero(szo, szi);
		b = OutputType::Zero(szo);
	}

	virtual const OutputType Predict(const InputType& input)
	{
		return linkFunc(W * input + b);
	}

	virtual void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);
				
		BatchOutputType pred = BatchOutputType::Zero(batchOutput.rows(), batchOutput.cols());
		BatchOutputType linpred = BatchOutputType::Zero(batchOutput.rows(), batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
		{
			linpred.col(i) = W * batchInput.col(i) + b;
			pred.col(i) = linkFunc(linpred.col(i));
		}
		
		solver.setLinearPrediction(linpred);
		solver.setPrediction(pred);

		solver.getWeightsAndBias(W, b);
	}

	double getLoss() const
	{
		return solver.getLoss();
	}

protected:
	LinkFunction linkFunc;

	WeigthsType W;
	OutputType b;

	Solver solver;
};

template<class Solver, class LinkFunction, class LossFunction>
class GeneralizedLinearModel<double, double, double, Solver, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	GeneralizedLinearModel(int szi = 1, int szo = 1)
	{
		Initialize(szi, szo);
		W = 0;
		b = 0;
	}

	void Initialize(int szi = 1, int szo = 1)
	{
		solver.Initialize(szi, szo);
	}

	virtual const double Predict(const double& input)
	{
		return linkFunc(W * input + b);
	}

	virtual void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);

		// TODO: Make something like this work

		Eigen::RowVectorXd pred = Eigen::RowVectorXd::Zero(batchOutput.cols());
		Eigen::RowVectorXd linpred = Eigen::RowVectorXd::Zero(batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
		{
			linpred(i) = W * batchInput(i) + b;
			pred(i) = linkFunc(linpred(i));
		}

		solver.setLinearPrediction(linpred);
		solver.setPrediction(pred);

		solver.getWeightsAndBias(W, b);
	}

	double getLoss() const
	{
		return solver.getLoss();
	}

protected:
	LinkFunction linkFunc;

	double W;
	double b;

	Solver solver;
};