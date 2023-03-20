#pragma once

#include <Eigen/eigen>

namespace Norm
{

	template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd> class Normalizer
	{
	public:
		Normalizer(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			samplesCount = 0;

			sumInput = InputType::Zero(szi);
			sum2Input = InputType::Zero(szi);

			sumOutput = OutputType::Zero(szo);
			sum2Output = OutputType::Zero(szo);
		}

		const InputType getAverageInput() const
		{
			return sumInput / samplesCount;
		}

		const OutputType getAverageOutput() const
		{
			return sumOutput / samplesCount;
		}


		const InputType getVarianceInput() const
		{
			InputType avg = getAverageInput();

			return sum2Input / samplesCount - avg.cwiseProduct(avg);
		}

		const OutputType getVarianceOutput() const
		{
			OutputType avg = getAverageOutput();

			return sum2Output / samplesCount - avg.cwiseProduct(avg);
		}

		void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			for (int i = 0; i < batchInput.cols(); ++i)
			{
				sumInput += batchInput.col(i);
				sumOutput += batchOutput.col(i);

				sum2Input += batchInput.col(i).cwiseProduct(batchInput.col(i));
				sum2Output += batchOutput.col(i).cwiseProduct(batchOutput.col(i));
			}

			samplesCount += static_cast<int>(batchInput.cols());
		}

	protected:
		int samplesCount;

		InputType sumInput;
		OutputType sumOutput;

		InputType sum2Input;
		OutputType sum2Output;
	};


	template<> class Normalizer<double, double, Eigen::RowVectorXd, Eigen::RowVectorXd>
	{
	public:
		Normalizer(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			samplesCount = 0;

			sumInput = 0;
			sum2Input = 0;

			sumOutput = 0;
			sum2Output = 0;
		}

		const double getAverageInput() const
		{
			return sumInput / samplesCount;
		}

		const double getAverageOutput() const
		{
			return sumOutput / samplesCount;
		}


		const double getVarianceInput() const
		{
			double avg = getAverageInput();

			return sum2Input / samplesCount - avg * avg;
		}

		const double getVarianceOutput() const
		{
			double avg = getAverageOutput();

			return sum2Output / samplesCount - avg * avg;
		}

		void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			for (int i = 0; i < batchInput.cols(); ++i)
			{
				sumInput += batchInput(i);
				sumOutput += batchOutput(i);

				sum2Input += batchInput(i) * batchInput(i);
				sum2Output += batchOutput(i) * batchOutput(i);
			}

			samplesCount += static_cast<int>(batchInput.cols());
		}

	protected:
		int samplesCount;

		double sumInput;
		double sumOutput;

		double sum2Input;
		double sum2Output;
	};


	template<> class Normalizer<Eigen::VectorXd, double, Eigen::MatrixXd, Eigen::RowVectorXd>
	{
	public:
		Normalizer(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			samplesCount = 0;

			sumInput = Eigen::VectorXd::Zero(szi);
			sum2Input = Eigen::VectorXd::Zero(szi);

			sumOutput = 0;
			sum2Output = 0;
		}

		const Eigen::VectorXd getAverageInput() const
		{
			return sumInput / samplesCount;
		}

		const double getAverageOutput() const
		{
			return sumOutput / samplesCount;
		}


		const Eigen::VectorXd getVarianceInput() const
		{
			Eigen::VectorXd avg = getAverageInput();

			return sum2Input / samplesCount - avg.cwiseProduct(avg);
		}

		const double getVarianceOutput() const
		{
			double avg = getAverageOutput();

			return sum2Output / samplesCount - avg * avg;
		}

		void AddBatch(const Eigen::MatrixXd& batchInput, const Eigen::RowVectorXd& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			for (int i = 0; i < batchInput.cols(); ++i)
			{
				sumInput += batchInput.col(i);
				sumOutput += batchOutput(i);

				sum2Input += batchInput.col(i).cwiseProduct(batchInput.col(i));
				sum2Output += batchOutput(i) * batchOutput(i);
			}

			samplesCount += static_cast<int>(batchInput.cols());
		}

	protected:
		int samplesCount;

		Eigen::VectorXd sumInput;
		double sumOutput;

		Eigen::VectorXd sum2Input;
		double sum2Output;
	};

}
