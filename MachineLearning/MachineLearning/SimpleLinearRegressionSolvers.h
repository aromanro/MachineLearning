#pragma once

#include <Eigen/Eigen>

#include "ActivationFunctions.h"

namespace SLRS
{

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType>
	class SimpleLinearRegressionSolver {
	public:
		SimpleLinearRegressionSolver(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			size = szo;
			xaccum = InputType::Zero(szi);
			x2accum = InputType::Zero(szi);
			xyaccum = InputType::Zero(size);
			yaccum = OutputType::Zero(size);
			count = 0;
		}

		void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			for (unsigned int i = 0; i < batchInput.cols(); ++i)
				AddSample(batchInput.col(i), batchOutput.col(i));

			target = batchOutput;
		}

		void setPrediction(const BatchOutputType& output)
		{
			pred = output;
		}

		void setLinearPrediction(const BatchOutputType& output)
		{
		}

		Eigen::MatrixXd getWeightsAndBias(WeightsType& w, OutputType& b) const
		{
			if (!count)
				return Eigen::MatrixXd();

			WeightsType wi;
			if (xaccum.size() == 1)
				wi = (count * xyaccum - xaccum(0) * yaccum) / (count * x2accum(0) - xaccum(0) * xaccum(0));
			else
				wi = (count * xyaccum - xaccum.cwiseProduct(yaccum)).cwiseProduct((count * x2accum - xaccum.cwiseProduct(xaccum)).cwiseInverse());

			w = wi;

			if (xaccum.size() == 1)
				b = (yaccum - wi * xaccum(0)) / count;
			else
				b = (yaccum - wi.cwiseProduct(xaccum)) / count;

			return Eigen::MatrixXd(); // an empty matrix, no need of it, it won't be used
		}

		const long long int getSize() const
		{
			return size;
		}

		double getLoss() const
		{
			double cost = 0;

			for (int c = 0; c < target.cols(); ++c)
				cost += lossFunction(pred.col(c), target.col(c)).sum();

			return cost;
		}

	protected:
		void AddSample(const InputType& input, const OutputType& output)
		{
			xaccum += input;

			x2accum += input.cwiseProduct(input);

			if (input.size() == 1)
				xyaccum += input(0) * output;
			else
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

		BatchOutputType pred;
		BatchOutputType target;

	public:
		ActivationFunctions::IdentityFunction<Eigen::VectorXd> activationFunction;
		LossFunctions::L2Loss<Eigen::VectorXd> lossFunction;
	};


	template<> class SimpleLinearRegressionSolver<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd> {
	public:
		SimpleLinearRegressionSolver(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			size = szo;
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

			target = batchOutput;
		}

		void setPrediction(const Eigen::MatrixXd& output)
		{
			pred = output;
		}

		void setLinearPrediction(const Eigen::MatrixXd& output)
		{
		}

		Eigen::MatrixXd getWeightsAndBias(Eigen::MatrixXd& w, Eigen::VectorXd& b) const
		{
			if (!count)
				return Eigen::MatrixXd();

			w = (count * xyaccum - xaccum * yaccum) / (count * x2accum - xaccum * xaccum);
			b = (yaccum - w * xaccum) / count;

			return Eigen::MatrixXd();
		}

		const long long int getSize() const
		{
			return size;
		}

		double getLoss() const
		{
			double cost = 0;

			for (int c = 0; c < target.cols(); ++c)
				cost += lossFunction(pred.col(c), target.col(c)).sum();

			return cost;
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

		Eigen::MatrixXd pred;
		Eigen::MatrixXd target;

	public:
		ActivationFunctions::IdentityFunction<Eigen::VectorXd> activationFunction;
		LossFunctions::L2Loss<Eigen::VectorXd> lossFunction;
	};

	template<> class SimpleLinearRegressionSolver<double, double, double, Eigen::RowVectorXd>
	{
	public:
		SimpleLinearRegressionSolver() :
			xaccum(0), x2accum(0), xyaccum(0), yaccum(0), count(0)
		{
		}

		void Initialize(int szi = 1, int szo = 1) // the parameters are ignored for this one
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

			target = batchOutput;
		}

		void setPrediction(const Eigen::RowVectorXd& output)
		{
			pred = output;
		}

		void setLinearPrediction(const Eigen::RowVectorXd& output)
		{
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b) const
		{
			if (!count) {
				w = b = 0;
				return Eigen::RowVectorXd();
			}

			w = (count * xyaccum - xaccum * yaccum) / (count * x2accum - xaccum * xaccum);
			b = (yaccum - w * xaccum) / count;

			return Eigen::RowVectorXd();
		}

		const long long int getSize() const
		{
			return 1;
		}

		double getLoss() const
		{
			double cost = 0;

			for (int c = 0; c < target.cols(); ++c)
				cost += lossFunction(pred.col(c), target.col(c)).sum();

			return cost;
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

		Eigen::RowVectorXd pred;
		Eigen::RowVectorXd target;

	public:
		ActivationFunctions::IdentityFunction<double> activationFunction;
		LossFunctions::L2Loss<Eigen::VectorXd> lossFunction;
	};

}


