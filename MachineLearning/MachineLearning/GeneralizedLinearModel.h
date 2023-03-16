#pragma once

#include <random>

#include "ActivationFunctions.h"
#include "CostFunctions.h"

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, class Solver = AdamSolver<>, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
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

		//W = WeightsType::Random(szo, szi);

		W = WeightsType::Zero(szo, szi); 
		
		// Eigen has a Random generator, but for now I'll stick with this one:
		// it's easier to reproduce issues this way, too
		
		// random between -1 and 1 by default

		std::random_device rd;
		std::mt19937 rde(/*42*/rd());
		const double x = 1. / sqrt(szi);
		std::uniform_real_distribution<> dist(-x, x);
		for (int i = 0; i < szo; ++i)
			for (int j = 0; j < szi; ++j)
				W(i, j) = dist(rde);
		
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
	WeightsType W;
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