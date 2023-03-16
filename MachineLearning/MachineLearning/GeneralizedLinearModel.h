#pragma once

#include "ActivationFunctions.h"
#include "CostFunctions.h"

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeigthsType = Eigen::MatrixXd, class Solver = AdamSolver<>, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
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

	virtual OutputType Predict(const InputType& input) const
	{
		return solver.activationFunction(W * input + b);
	}

	void AddBatchNoParamsAdjustment(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);

		BatchOutputType pred(batchOutput.rows(), batchOutput.cols());
		BatchOutputType linpred(batchOutput.rows(), batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
		{
			linpred.col(i) = W * batchInput.col(i) + b;
			pred.col(i) = solver.activationFunction(linpred.col(i));
		}

		solver.setLinearPrediction(linpred);
		solver.setPrediction(pred);
	}

	BatchOutputType getPrediction() const
	{
		return solver.getPrediction();
	}


	virtual BatchOutputType AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		AddBatchNoParamsAdjustment(batchInput, batchOutput);

		return solver.getWeightsAndBias(W, b);
	}

	double getLoss() const
	{
		return solver.getLoss();
	}

	InputType Backpropagate(const OutputType& grad) const
	{
		return W.transpose() * grad;
	}

	BatchInputType BackpropagateBatch(const BatchOutputType& grad) const
	{
		InputType firstCol = Backpropagate(grad.col(0));
		BatchInputType res(firstCol.size(), grad.cols());

		res.col(0) = firstCol;
		for (int i = 1; i < grad.cols(); ++i)
			res.col(i) = Backpropagate(grad.col(i));

		return res;
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

	virtual double Predict(const double& input) const
	{
		return solver.activationFunction(W * input + b);
	}

	void AddBatchNoParamsAdjustment(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		solver.AddBatch(batchInput, batchOutput);

		Eigen::RowVectorXd pred(batchOutput.cols());
		Eigen::RowVectorXd linpred(batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
		{
			linpred(i) = W * batchInput(i) + b;
			pred(i) = solver.activationFunction(linpred(i));
		}

		solver.setLinearPrediction(linpred);
		solver.setPrediction(pred);
	}

	Eigen::RowVectorXd getPrediction() const
	{
		return solver.getPrediction();
	}

	virtual Eigen::RowVectorXd AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		AddBatchNoParamsAdjustment(batchInput, batchOutput);

		return solver.getWeightsAndBias(W, b);
	}

	double getLoss() const
	{
		return solver.getLoss();
	}

	double Backpropagate(const double& grad) const
	{
		return W * grad;
	}

	Eigen::RowVectorXd BackpropagateBatch(const Eigen::RowVectorXd& grad) const
	{
		Eigen::RowVectorXd res(grad.size());

		for (int i = 0; i < grad.size(); ++i)
			res(i) = Backpropagate(grad(i));

		return res;
	}

protected:
	double W;
	double b;

public:
	Solver solver;
};