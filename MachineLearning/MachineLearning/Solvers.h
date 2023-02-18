#pragma once

#include <Eigen/eigen>

template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::VectorXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd> 
class SimpleLinearRegressionSolver {
public:
	SimpleLinearRegressionSolver(int sz = 1)
	{
		Initialize(sz);
	}

	void Initialize(int sz = 1)
	{
		size = sz;
		xaccum = InputType::Zero(size);
		x2accum = InputType::Zero(size);
		xyaccum = InputType::Zero(size);
		yaccum = OutputType::Zero(size);
		count = 0;
	}

	void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
	{
		assert(batchInput.cols() == batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
			AddSample(batchInput.col(i), batchOutput.col(i));
	}

	void setPrediction(const BatchOutputType& output)
	{
	}

	void setLinearPrediction(const BatchOutputType& output)
	{
	}

	void getWeightsAndBias(WeightsType& w, OutputType& b) const
	{
		if (!count) {
			w = WeightsType::Zero(size);
			b = OutputType::Zero(size);
			return;
		}

		w = (count * xyaccum - xaccum.cwiseProduct(yaccum)).cwiseProduct((count * x2accum - xaccum.cwiseProduct(xaccum)).cwiseInverse());
		b = (yaccum - w.cwiseProduct(xaccum)) / count;
	}

	const long long int getSize() const
	{
		return size;
	}

protected:
	void AddSample(const InputType& input, const OutputType& output)
	{
		xaccum += input;
		x2accum += input.cwiseProduct(input);
		xyaccum += input.cwiseProduct(output);
		yaccum += output;
		++count;
	}

	InputType xaccum;
	InputType x2accum;
	OutputType xyaccum;
	OutputType yaccum;
	unsigned long long int count;
	unsigned long long int size;
};


template<> class SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::VectorXd, Eigen::RowVectorXd, Eigen::MatrixXd> {
public:
	SimpleLinearRegressionSolver(int sz = 1)
	{
		Initialize(sz);
	}

	void Initialize(int sz = 1)
	{
		size = sz;
		xaccum = 0;
		x2accum = 0;
		xyaccum = Eigen::VectorXd::Zero(size);
		yaccum = Eigen::VectorXd::Zero(size);
		count = 0;
	}

	void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::MatrixXd& batchOutput)
	{
		assert(batchInput.cols() == batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
			AddSample(batchInput(i), batchOutput.col(i));
	}

	void setPrediction(const Eigen::MatrixXd& output)
	{
	}

	void setLinearPrediction(const Eigen::MatrixXd& output)
	{
	}

	void getWeightsAndBias(Eigen::VectorXd& w, Eigen::VectorXd& b) const
	{
		if (!count) {
			w = Eigen::VectorXd::Zero(size);
			b = Eigen::VectorXd::Zero(size);
			return;
		}

		w = (count * xyaccum - xaccum * yaccum) / (count * x2accum - xaccum * xaccum);
		b = (yaccum - w * xaccum) / count;
	}

	const long long int getSize() const
	{
		return size;
	}

protected:
	void AddSample(const double& input, const Eigen::VectorXd& output)
	{
		xaccum += input;
		x2accum += input * input;
		xyaccum += input * output;
		yaccum += output;
		++count;
	}

	double xaccum;
	double x2accum;
	Eigen::VectorXd xyaccum;
	Eigen::VectorXd yaccum;
	unsigned long long int count;
	unsigned long long int size;
};

template<> class SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd> 
{
public:
	SimpleLinearRegressionSolver() :
		xaccum(0), x2accum(0), xyaccum(0), yaccum(0), count(0)
	{
	}

	void Initialize(int sz = 1) // the parameter is ignored for this one
	{
		xaccum = 0;
		x2accum = 0;
		xyaccum = 0;
		yaccum = 0;
		count = 0;
	}

	void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
	{
		assert(batchInput.cols() == batchOutput.cols());

		for (unsigned int i = 0; i < batchInput.cols(); ++i)
			AddSample(batchInput.col(i)(0), batchOutput.col(i)(0));
	}

	void setPrediction(const Eigen::RowVectorXd& output)
	{
	}

	void setLinearPrediction(const Eigen::RowVectorXd& output)
	{
	}

	void getWeightsAndBias(double& w, double& b) const
	{
		if (!count) {
			w = b = 0;
			return;
		}

		w = (count * xyaccum - xaccum * yaccum) / (count * x2accum - xaccum * xaccum);
		b = (yaccum - w * xaccum) / count;
	}

	const long long int getSize() const
	{
		return 1;
	}

protected:
	void AddSample(const double& input, const double& output)
	{
		xaccum += input;
		x2accum += input * input;
		xyaccum += input * output;
		yaccum += output;
		++count;
	}

	double xaccum;
	double x2accum;
	double xyaccum;
	double yaccum;
	unsigned long long int count;
};


