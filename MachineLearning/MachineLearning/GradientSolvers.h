#pragma once

#include <Eigen/eigen>
#include <unsupported/Eigen/MatrixFunctions>

#include "LinkFunctions.h"
#include "CostFunctions.h"

#include <iostream>

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
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

		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad.cwiseProduct(lossLinkGrad).sum());
		if (n > lim)
			lossLinkGrad *= lim / n;

		b -= alpha * lossLinkGrad;

		WeightsType wAdj = WeightsType::Zero(w.rows(), w.cols());
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c).transpose();

		w -= alpha * wAdj;
		//alpha *= 0.9999; //learning rate could decrease over time
	}

	double getLoss() const
	{
		OutputType cost = OutputType::Zero(output.rows(), output.cols());

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c), output.col(c));

		return cost.sum();
	}


	double alpha = 0.000001;
	double lim = 20.;

protected:
	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;
};

template<class LinkFunction, class LossFunction>
class GradientDescentSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
	}

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

	void getWeightsAndBias(double& w, double& b) const
	{
		double lossLinkGrad = 0.;

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)(0)) * lossFunction.derivative(pred.col(c)(0), output.col(c)(0));


		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad * lossLinkGrad);
		if (n > lim)
			lossLinkGrad *= lim / n;

		b -= alpha * lossLinkGrad;

		double wAdj = 0.;
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c)(0);

		w -= alpha * wAdj;
		//alpha *= 0.9999; //learning rate could decrease over time
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c)(0), output.col(c)(0));

		return cost;
	}


	double alpha = 0.000001;
	double lim = 20.;

protected:
	Eigen::RowVectorXd pred;
	Eigen::RowVectorXd linpred;

	Eigen::RowVectorXd input;
	Eigen::RowVectorXd output;

	LinkFunction linkFunction;
	LossFunction lossFunction;
};



template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class MomentumSolver
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		mW = WeightsType::Zero(szo, szi);
		mb = OutputType::Zero(szo);
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

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		OutputType lossLinkGrad = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)).cwiseProduct(lossFunction.derivative(pred.col(c), output.col(c)));


		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad.cwiseProduct(lossLinkGrad).sum());
		if (n > lim)
			lossLinkGrad *= lim / n;

		const double norm = 1. / input.cols();
		mb = beta * mb - alpha * lossLinkGrad;

		b += mb;

		WeightsType wAdj = WeightsType::Zero(w.rows(), w.cols());
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c).transpose();
		mW = beta * mW - alpha * wAdj;

		w += mW;
	}

	double getLoss() const
	{
		OutputType cost = OutputType::Zero(output.rows(), output.cols());

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c), output.col(c));

		return cost.sum();
	}


	double alpha = 0.000001;
	double beta = 0.5;
	double lim = 20.;

protected:
	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	WeightsType mW;
	OutputType mb;
};

template<class LinkFunction, class LossFunction>
class MomentumSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		mW = 0;
		mb = 0;
	}

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

	void getWeightsAndBias(double& w, double& b)
	{
		double lossLinkGrad = 0.;

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)(0)) * lossFunction.derivative(pred.col(c)(0), output.col(c)(0));


		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad * lossLinkGrad);
		if (n > lim)
			lossLinkGrad *= lim / n;
	
		mb = beta * mb - alpha * lossLinkGrad;

		b += mb;

		double wAdj = 0.;
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c)(0);

		mW = beta * mW - alpha * wAdj;

		w += mW;
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c)(0), output.col(c)(0));

		return cost;
	}


	double alpha = 0.000001;
	double beta = 0.5;
	double lim = 20.;

protected:
	Eigen::RowVectorXd pred;
	Eigen::RowVectorXd linpred;

	Eigen::RowVectorXd input;
	Eigen::RowVectorXd output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	double mW;
	double mb;
};


template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class AdaGradSolver
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		sW = WeightsType::Zero(szo, szi);
		sb = OutputType::Zero(szo);
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

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		OutputType lossLinkGrad = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)).cwiseProduct(lossFunction.derivative(pred.col(c), output.col(c)));


		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad.cwiseProduct(lossLinkGrad).sum());
		if (n > lim)
			lossLinkGrad *= lim / n;


		sb += lossLinkGrad.cwiseProduct(lossLinkGrad);

		OutputType sba = sb + OutputType::Constant(sb.rows(), sb.cols(), 0.000001);
		b -= alpha * lossLinkGrad.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

		WeightsType wAdj = WeightsType::Zero(w.rows(), w.cols());
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c).transpose();

		sW += wAdj.cwiseProduct(wAdj);

		WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), 0.000001);
		w -= alpha * wAdj.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());
	}

	double getLoss() const
	{
		OutputType cost = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c), output.col(c));

		return cost.sum();
	}


	double alpha = 0.1;
	double lim = 20.;

protected:
	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	WeightsType sW;
	OutputType sb;
};

template<class LinkFunction, class LossFunction>
class AdaGradSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		sW = 0;
		sb = 0;
	}

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

	void getWeightsAndBias(double& w, double& b)
	{
		double lossLinkGrad = 0.;

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)(0)) * lossFunction.derivative(pred.col(c)(0), output.col(c)(0));


		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;


		// clip it if necessary
		const double n = sqrt(lossLinkGrad * lossLinkGrad);
		if (n > lim)
			lossLinkGrad *= lim / n;


		sb += lossLinkGrad * lossLinkGrad;

		b -= alpha * lossLinkGrad / sqrt((abs(sb) < 0.00001) ? 0.00001 : sb);
		
		double wAdj = 0.;
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c)(0);

		sW += wAdj * wAdj;

		w -= alpha * wAdj / sqrt((abs(sW) < 0.00001) ? 0.00001 : sW);
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c)(0), output.col(c)(0));

		return cost;
	}


	double alpha = 0.1;
	double lim = 20.;
protected:

	Eigen::RowVectorXd pred;
	Eigen::RowVectorXd linpred;

	Eigen::RowVectorXd input;
	Eigen::RowVectorXd output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	double sW;
	double sb;
};


template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class RMSPropSolver
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		sW = WeightsType::Zero(szo, szi);
		sb = OutputType::Zero(szo);
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

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		OutputType lossLinkGrad = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)).cwiseProduct(lossFunction.derivative(pred.col(c), output.col(c)));

		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad.cwiseProduct(lossLinkGrad).sum());
		if (n > lim)
			lossLinkGrad *= lim / n;

		sb = beta * sb + (1. - beta) * lossLinkGrad.cwiseProduct(lossLinkGrad);

		OutputType sba = sb + OutputType::Constant(sb.rows(), sb.cols(), 0.000001);
		b -= alpha * lossLinkGrad.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

		WeightsType wAdj = WeightsType::Zero(w.rows(), w.cols());
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c).transpose();

		sW = beta * sW + (1. - beta) * wAdj.cwiseProduct(wAdj);

		WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), 0.000001);
		w -= alpha * wAdj.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());
	}

	double getLoss() const
	{
		OutputType cost = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c), output.col(c));

		return cost.sum();
	}


	double alpha = 0.01;
	double beta = 0.5;
	double lim = 20.;

protected:
	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	WeightsType sW;
	OutputType sb;
};

template<class LinkFunction, class LossFunction>
class RMSPropSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		sW = 0;
		sb = 0;
	}

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

	void getWeightsAndBias(double& w, double& b)
	{
		double lossLinkGrad = 0.;

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)(0)) * lossFunction.derivative(pred.col(c)(0), output.col(c)(0));

		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad * lossLinkGrad);
		if (n > lim)
			lossLinkGrad *= lim / n;


		sb = beta * sb + (1. - beta) * lossLinkGrad * lossLinkGrad;

		b -= alpha * lossLinkGrad / sqrt((abs(sb) < 0.00001) ? 0.00001 : sb);

		double wAdj = 0.;
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c)(0);

		sW = beta * sW + (1. - beta) * wAdj * wAdj;

		w -= alpha * wAdj / sqrt((abs(sW) < 0.00001) ? 0.00001 : sW);
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c)(0), output.col(c)(0));

		return cost;
	}


	double alpha = 0.01;
	double beta = 0.5;
	double lim = 20.;

protected:
	int step;

	Eigen::RowVectorXd pred;
	Eigen::RowVectorXd linpred;

	Eigen::RowVectorXd input;
	Eigen::RowVectorXd output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	double sW;
	double sb;
};


template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd, class LinkFunction = IdentityFunction<OutputType>, class LossFunction = L2Loss<OutputType>>
class AdamSolver
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		sW = WeightsType::Zero(szo, szi);
		sb = OutputType::Zero(szo);
		mW = WeightsType::Zero(szo, szi);
		mb = OutputType::Zero(szo);
		step = 0;
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

	void getWeightsAndBias(WeightsType& w, OutputType& b)
	{
		++step;
		OutputType lossLinkGrad = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)).cwiseProduct(lossFunction.derivative(pred.col(c), output.col(c)));

		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad.cwiseProduct(lossLinkGrad).sum());
		if (n > lim)
			lossLinkGrad *= lim / n;

		const double div1 = 1. / (1. - pow(beta1, step));
		const double div2 = 1. / (1. - pow(beta2, step));

		mb = beta1 * mb - (1. - beta1) * lossLinkGrad;
		mb *= div1;
		sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad);
		sb *= div2;

		OutputType sba = sb + OutputType::Constant(sb.rows(), sb.cols(), 0.000001);
		b += alpha * mb.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

		WeightsType wAdj = WeightsType::Zero(w.rows(), w.cols());
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c).transpose();

		mW = beta1 * mW - (1. - beta1) * wAdj;
		mW *= div1;
		sW = beta2 * sW + (1. - beta2) * wAdj.cwiseProduct(wAdj);
		sW *= div2;

		WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), 0.000001);
		w += alpha * mW.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());
	}

	double getLoss() const
	{
		OutputType cost = OutputType::Zero(output.rows());

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c), output.col(c));

		return cost.sum();
	}


	double alpha = 0.01;
	double beta1 = 0.7;
	double beta2 = 0.9;
	double lim = 20.;

protected:
	int step = 0;

	BatchOutputType pred;
	BatchOutputType linpred;

	BatchInputType input;
	BatchOutputType output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	WeightsType sW;
	OutputType sb;

	WeightsType mW;
	OutputType mb;
};

template<class LinkFunction, class LossFunction>
class AdamSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, LinkFunction, LossFunction>
{
public:
	void Initialize(int szi = 1, int szo = 1)
	{
		sW = 0;
		sb = 0;
		mb = 0;
		mW = 0;
		step = 0;
	}

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

	void getWeightsAndBias(double& w, double& b)
	{
		++step;
		double lossLinkGrad = 0.;

		for (int c = 0; c < output.cols(); ++c)
			lossLinkGrad += linkFunction.derivative(linpred.col(c)(0)) * lossFunction.derivative(pred.col(c)(0), output.col(c)(0));

		const double norm = 1. / input.cols();
		lossLinkGrad *= norm;

		// clip it if necessary
		const double n = sqrt(lossLinkGrad * lossLinkGrad);
		if (n > lim)
			lossLinkGrad *= lim / n;


		const double div1 = 1. / (1. - pow(beta1, step));
		const double div2 = 1. / (1. - pow(beta2, step));

		mb = beta1 * mb - (1. - beta1) * lossLinkGrad;
		mb *= div1;
		sb = beta2 * sb + (1. - beta2) * lossLinkGrad * lossLinkGrad;
		sb *= div2;

		b += alpha * mb / sqrt((abs(sb) < 0.00001) ? 0.00001 : sb);

		double wAdj = 0.;
		for (int c = 0; c < output.cols(); ++c)
			wAdj += lossLinkGrad * input.col(c)(0);

		mW = beta1 * mW - (1. - beta1) * wAdj;
		mW *= div1;
		sW = beta2 * sW + (1. - beta2) * wAdj * wAdj;
		sW *= div2;

		w += alpha * mW / sqrt((abs(sW) < 0.00001) ? 0.00001 : sW);
	}

	double getLoss() const
	{
		double cost = 0;

		for (int c = 0; c < output.cols(); ++c)
			cost += lossFunction(pred.col(c)(0), output.col(c)(0));

		return cost;
	}


	double alpha = 0.01;
	double beta1 = 0.7;
	double beta2 = 0.9;
	double lim = 20.;

protected:
	int step = 0;

	Eigen::RowVectorXd pred;
	Eigen::RowVectorXd linpred;

	Eigen::RowVectorXd input;
	Eigen::RowVectorXd output;

	LinkFunction linkFunction;
	LossFunction lossFunction;

	double sW;
	double sb;

	double mW;
	double mb;
};