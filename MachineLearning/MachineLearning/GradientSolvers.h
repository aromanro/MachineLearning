#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

#include "LinkFunctions.h"
#include "CostFunctions.h"

#include <iostream>

const double eps = 1E-10;

//*************************************************************************************************************************************************************************************************************************************************************************************************************
// Base class for all
//*************************************************************************************************************************************************************************************************************************************************************************************************************

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class GradientDescentSolverBase
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

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c <output.cols(); ++c)
			cost += lossFunction(pred.col(c), output.col(c)).sum();

		return cost;
	}

	double alpha = 0.001;
	double lim = 20.;

protected:
	inline BatchOutputType getGrad()
	{
		BatchOutputType lossLinkGrad;
		lossLinkGrad.resize(output.rows(), output.cols());
		
		const double norm = 1. / input.cols();

		for (int c = 0; c < output.cols(); ++c)
		{
			lossLinkGrad.col(c) = norm * linkFunction.derivative(linpred.col(c)).cwiseProduct(lossFunction.derivative(pred.col(c), output.col(c)));
			// clip it if necessary
			const double n = sqrt(lossLinkGrad.col(c).cwiseProduct(lossLinkGrad.col(c)).sum());
			if (n > lim)
				lossLinkGrad.col(c) *= lim / n;
		}

		return lossLinkGrad;
	}

	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;
};


template<class LinkFunction, class LossFunction>
class GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:

	void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		assert(batchInput.cols() == batchOutput.cols());

		input = batchInput;
		output = batchOutput;
	}

	void setPrediction(const Eigen::RowVectorXd& output)
	{
		pred = output;
	}

	void setLinearPrediction(const Eigen::RowVectorXd& output) // before calling the link function
	{
		linpred = output;
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred(c), output(c));

		return cost;
	}

	double alpha = 0.001;
	double lim = 20.;

protected:
	inline Eigen::RowVectorXd getGrad()
	{
		Eigen::RowVectorXd lossLinkGrad;
		lossLinkGrad.resize(output.cols());

		const double norm = 1. / input.cols();

		for (int c = 0; c < output.cols(); ++c)
		{
			lossLinkGrad(c) = norm * linkFunction.derivative(linpred(c)) * lossFunction.derivative(pred(c), output(c));
			// clip it if necessary
			const double n = sqrt(lossLinkGrad(c) * lossLinkGrad(c));
			if (n > lim)
				lossLinkGrad(c) *= lim / n;
		}

		return lossLinkGrad;
	}

	Eigen::RowVectorXd pred;
	Eigen::RowVectorXd linpred;

	Eigen::RowVectorXd input;
	Eigen::RowVectorXd output;

	LinkFunction linkFunction;
	LossFunction lossFunction;
};


//*************************************************************************************************************************************************************************************************************************************************************************************************************
// The simplest one
//*************************************************************************************************************************************************************************************************************************************************************************************************************


template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class GradientDescentSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction>
{
public:
	typedef  GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		const BatchOutputType lossLinkGrad = BaseType::getGrad();

		b -= BaseType::alpha * lossLinkGrad.rowwise().sum();

		const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
		w -= BaseType::alpha * wAdj;

		BaseType::alpha *= decay; //learning rate could decrease over time
	}


	double decay = 1.;
};

template<class LinkFunction, class LossFunction>
class GradientDescentSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	typedef GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
	}

	void getWeightsAndBias(double& w, double& b)
	{
		const double lossLinkGrad = BaseType::getGrad();

		b -= BaseType::alpha * lossLinkGrad;

		const double wAdj = lossLinkGrad * BaseType::input.transpose();
		w -= BaseType::alpha * wAdj;

		BaseType::alpha *= decay; //learning rate could decrease over time
	}

	double decay = 1.;
};


//*************************************************************************************************************************************************************************************************************************************************************************************************************
// Momentum
//*************************************************************************************************************************************************************************************************************************************************************************************************************



template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class MomentumSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction>
{
public:
	typedef  GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		mW = WeightsType::Zero(szo, szi);
		mb = OutputType::Zero(szo);
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		const BatchOutputType lossLinkGrad = BaseType::getGrad();

		mb = beta * mb - BaseType::alpha * lossLinkGrad.rowwise().sum();

		b += mb;

		const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
		mW = beta * mW - BaseType::alpha * wAdj;

		w += mW;
	}

	double beta = 0.5;

protected:
	WeightsType mW;
	OutputType mb;
};

template<class LinkFunction, class LossFunction>
class MomentumSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	typedef GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		mW = 0;
		mb = 0;
	}

	void getWeightsAndBias(double& w, double& b)
	{
		const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();
	
		mb = beta * mb - BaseType::alpha * lossLinkGrad.sum();
		b += mb;

		const double wAdj = (lossLinkGrad * BaseType::input.transpose())(0);
		mW = beta * mW - BaseType::alpha * wAdj;

		w += mW;
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < BaseType::output.cols(); ++c)
			cost += BaseType::lossFunction(BaseType::pred(c), BaseType::output(c));

		return cost;
	}

	double beta = 0.5;

protected:
	double mW;
	double mb;
};


//*************************************************************************************************************************************************************************************************************************************************************************************************************
// Ada Grad
//*************************************************************************************************************************************************************************************************************************************************************************************************************



template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class AdaGradSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction>
{
public:
	typedef  GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		sW = WeightsType::Zero(szo, szi);
		sb = OutputType::Zero(szo);
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		const BatchOutputType lossLinkGrad = BaseType::getGrad();

		sb += lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();

		const OutputType sba = sb + OutputType::Constant(sb.rows(), sb.cols(), eps);
		b -= BaseType::alpha * lossLinkGrad.rowwise().sum().cwiseProduct(sba.cwiseSqrt().cwiseInverse());

		const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
		sW += wAdj.cwiseProduct(wAdj);

		const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
		w -= BaseType::alpha * wAdj.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());
	}

protected:
	WeightsType sW;
	OutputType sb;
};

template<class LinkFunction, class LossFunction>
class AdaGradSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	typedef GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		sW = 0;
		sb = 0;
	}

	void getWeightsAndBias(double& w, double& b)
	{
		const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();

		sb += lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
		b -= BaseType::alpha * lossLinkGrad.sum() / sqrt(sb + eps);
		
		const double wAdj = (lossLinkGrad * BaseType::input.transpose())(0);
		sW += wAdj * wAdj;

		w -= BaseType::alpha * wAdj / sqrt(sW + eps);
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < BaseType::output.cols(); ++c)
			cost += BaseType::lossFunction(BaseType::pred(c), BaseType::output(c));

		return cost;
	}

protected:
	double sW;
	double sb;
};


//*************************************************************************************************************************************************************************************************************************************************************************************************************
// RMSProp
//*************************************************************************************************************************************************************************************************************************************************************************************************************



template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class RMSPropSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction>
{
public:
	typedef  GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		sW = WeightsType::Zero(szo, szi);
		sb = OutputType::Zero(szo);
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		const BatchOutputType lossLinkGrad = BaseType::getGrad();

		sb = beta * sb + (1. - beta) * lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();

		const OutputType sba = sb + OutputType::Constant(sb.rows(), sb.cols(), eps);
		b -= BaseType::alpha * lossLinkGrad.rowwise().sum().cwiseProduct(sba.cwiseSqrt().cwiseInverse());

		const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
		sW = beta * sW + (1. - beta) * wAdj.cwiseProduct(wAdj);

		const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
		w -= BaseType::alpha * wAdj.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());
	}

	double beta = 0.5;

protected:
	WeightsType sW;
	OutputType sb;
};

template<class LinkFunction, class LossFunction>
class RMSPropSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	typedef GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		sW = 0;
		sb = 0;
	}

	void getWeightsAndBias(double& w, double& b)
	{
		const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();

		sb = beta * sb + (1. - beta) * lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
		b -= BaseType::alpha * lossLinkGrad.sum() / sqrt(sb + eps);

		const double wAdj = (lossLinkGrad * BaseType::input.transpose())(0);
		sW = beta * sW + (1. - beta) * wAdj * wAdj;

		w -= BaseType::alpha * wAdj / sqrt(sW + eps);
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < BaseType::output.cols(); ++c)
			cost += BaseType::lossFunction(BaseType::pred(c), BaseType::output(c));

		return cost;
	}

	double beta = 0.5;

protected:
	double sW;
	double sb;
};

//*************************************************************************************************************************************************************************************************************************************************************************************************************
// Adam
//*************************************************************************************************************************************************************************************************************************************************************************************************************


template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class AdamSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction>
{
public:
	typedef  GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		sW = WeightsType::Zero(szo, szi);
		sb = OutputType::Zero(szo);
		mW = WeightsType::Zero(szo, szi);
		mb = OutputType::Zero(szo);
		step = 0;
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		++step;
		const BatchOutputType lossLinkGrad = BaseType::getGrad();

		const double div1 = 1. / (1. - pow(beta1, step));
		const double div2 = 1. / (1. - pow(beta2, step));

		mb = beta1 * mb - (1. - beta1) * lossLinkGrad.rowwise().sum();
		mb *= div1;
		sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();
		sb *= div2;

		const OutputType sba = sb + OutputType::Constant(sb.rows(), sb.cols(), eps);
		b += BaseType::alpha * mb.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

		const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();

		mW = beta1 * mW - (1. - beta1) * wAdj;
		mW *= div1;
		sW = beta2 * sW + (1. - beta2) * wAdj.cwiseProduct(wAdj);
		sW *= div2;

		const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
		w += BaseType::alpha * mW.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());
	}

	double beta1 = 0.9;
	double beta2 = 0.995;

protected:
	int step = 0;

	WeightsType sW;
	OutputType sb;

	WeightsType mW;
	OutputType mb;
};

template<class LinkFunction, class LossFunction>
class AdamSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	typedef GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction> BaseType;

	void Initialize(int szi = 1, int szo = 1)
	{
		sW = 0;
		sb = 0;
		mb = 0;
		mW = 0;
		step = 0;
	}

	void getWeightsAndBias(double& w, double& b)
	{
		++step;
		const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();

		const double div1 = 1. / (1. - pow(beta1, step));
		const double div2 = 1. / (1. - pow(beta2, step));

		mb = beta1 * mb - (1. - beta1) * lossLinkGrad.sum();
		mb *= div1;
		sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
		sb *= div2;

		b += BaseType::alpha * mb / sqrt(sb + eps);

		const double wAdj = (lossLinkGrad * BaseType::input.transpose())(0);

		mW = beta1 * mW - (1. - beta1) * wAdj;
		mW *= div1;
		sW = beta2 * sW + (1. - beta2) * wAdj * wAdj;
		sW *= div2;

		w += BaseType::alpha * mW / sqrt(sW + eps);
	}

	double beta1 = 0.9;
	double beta2 = 0.995;

protected:
	int step = 0;

	double sW;
	double sb;

	double mW;
	double mb;
};