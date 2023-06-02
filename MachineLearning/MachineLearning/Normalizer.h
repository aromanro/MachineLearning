#pragma once

#include <Eigen/Eigen>

namespace Norm
{

	template<typename InputType = Eigen::VectorXd, typename BatchInputType = Eigen::MatrixXd> class Normalizer
	{
	public:
		Normalizer(int sz = 1)
		{
			Initialize(sz);
		}

		void Initialize(int sz = 1)
		{
			samplesCount = 0;

			sumInput = InputType::Zero(sz);
			sum2Input = InputType::Zero(sz);
		}

		InputType getAverage() const
		{
			return sumInput / samplesCount;
		}

		InputType getVariance() const
		{
			InputType avg = getAverage();

			return sum2Input / samplesCount - avg.cwiseProduct(avg);
		}

		void AddBatch(const BatchInputType& batchInput)
		{
			for (int i = 0; i < batchInput.cols(); ++i)
			{
				sumInput += batchInput.col(i);

				sum2Input += batchInput.col(i).cwiseProduct(batchInput.col(i));
			}

			samplesCount += static_cast<int>(batchInput.cols());
		}

	private:
		int samplesCount;

		InputType sumInput;
		InputType sum2Input;
	};


	template<> class Normalizer<double, Eigen::RowVectorXd>
	{
	public:
		Normalizer(int sz = 1)
		{
			Initialize(sz);
		}

		void Initialize(int sz = 1)
		{
			samplesCount = 0;

			sumInput = 0;
			sum2Input = 0;
		}

		double getAverage() const
		{
			return sumInput / samplesCount;
		}

		double getVariance() const
		{
			double avg = getAverage();

			return sum2Input / samplesCount - avg * avg;
		}

		void AddBatch(const Eigen::RowVectorXd& batchInput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			for (int i = 0; i < batchInput.cols(); ++i)
			{
				sumInput += batchInput(i);

				sum2Input += batchInput(i) * batchInput(i);
			}

			samplesCount += static_cast<int>(batchInput.cols());
		}

	private:
		int samplesCount;
		double sumInput;
		double sum2Input;
	};

	template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = Eigen::MatrixXd> class InputOutputNormalizer
	{
	public:
		InputOutputNormalizer(int szi = 1, int szo = 1)
		{
			Initialize(szi, szo);
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			inputNormalizer.Initialize(szi);
			outputNormalizer.Initialize(szo);
		}

		InputType getAverageInput() const
		{
			return inputNormalizer.getAverage();
		}

		OutputType getAverageOutput() const
		{
			return outputNormalizer.getAverage();
		}


		InputType getVarianceInput() const
		{
			return inputNormalizer.getVariance();
		}

		OutputType getVarianceOutput() const
		{
			return outputNormalizer.getVariance();
		}

		void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			inputNormalizer.AddBatch(batchInput);
			outputNormalizer.AddBatch(batchOutput);
		}

	private:
		Normalizer<InputType, BatchInputType> inputNormalizer;
		Normalizer<OutputType, BatchOutputType> outputNormalizer;
	};

}
