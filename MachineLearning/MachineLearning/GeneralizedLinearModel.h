#pragma once

#include "ActivationFunctions.h"
#include "CostFunctions.h"

template<typename InputType, typename OutputType, typename WeigthsType, class Solver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
class GeneralizedLinearModel
{
public:
	GeneralizedLinearModel(int szi = 1, int szo = 1)
	{
		Initialize(szi, szo);
	}

	virtual ~GeneralizedLinearModel() {}

	void Initialize(int szi = 1, int szo = 1)
	{
		solver.Initialize(szi, szo);
		// TODO: provide initializers!
		W = WeigthsType::Random(szo, szi); // random between -1 and 1 by default
		b = OutputType::Zero(szo);
	}

	virtual const OutputType Predict(const InputType& input)
	{
		return solver.activationFunction(W * input + b);
	}

	void AddBatchNoParamsAdjustment(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);

		BatchOutputType pred = BatchOutputType::Zero(batchOutput.rows(), batchOutput.cols());
		BatchOutputType linpred = BatchOutputType::Zero(batchOutput.rows(), batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
		{
			linpred.col(i) = W * batchInput.col(i) + b;
			pred.col(i) = solver.activationFunction(linpred.col(i));
		}

		solver.setLinearPrediction(linpred);
		solver.setPrediction(pred);
	}

	virtual void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		AddBatchNoParamsAdjustment(batchInput, batchOutput);

		solver.getWeightsAndBias(W, b);
	}

	double getLoss() const
	{
		return solver.getLoss();
	}

protected:
	WeigthsType W;
	OutputType b;

public:
	Solver solver;
};

template<class Solver>
class GeneralizedLinearModel<double, double, double, Solver, Eigen::RowVectorXd, Eigen::RowVectorXd>
{
public:
	GeneralizedLinearModel(int szi = 1, int szo = 1)
	{
		Initialize(szi, szo);
		W = 0;
		b = 0;
	}

	virtual ~GeneralizedLinearModel() {}

	void Initialize(int szi = 1, int szo = 1)
	{
		solver.Initialize(szi, szo);
	}

	virtual const double Predict(const double& input)
	{
		return solver.activationFunction(W * input + b);
	}

	void AddBatchNoParamsAdjustment(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);

		Eigen::RowVectorXd pred = Eigen::RowVectorXd::Zero(batchOutput.cols());
		Eigen::RowVectorXd linpred = Eigen::RowVectorXd::Zero(batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
		{
			linpred(i) = W * batchInput(i) + b;
			pred(i) = solver.activationFunction(linpred(i));
		}

		solver.setLinearPrediction(linpred);
		solver.setPrediction(pred);
	}

	virtual void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		AddBatchNoParamsAdjustment(batchInput, batchOutput);

		solver.getWeightsAndBias(W, b);
	}

	double getLoss() const
	{
		return solver.getLoss();
	}

protected:
	double W;
	double b;

public:
	Solver solver;
};