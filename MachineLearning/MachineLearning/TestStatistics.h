#pragma once

namespace Utils {

	class TestStatistics
	{
	public:
		void AddPrediction(bool prediction, bool trueValue)
		{
			if (trueValue)
			{
				if (prediction) ++truePositives;
				else ++falseNegatives;
			}
			else
			{
				if (prediction) ++falsePositives;
				else ++trueNegatives;
			}
		}

		int getTruePositives() const
		{
			return truePositives;
		}

		int getTrueNegatives() const
		{
			return trueNegatives;
		}

		int getFalsePositives() const
		{
			return falsePositives;
		}

		int getFalseNegatives() const
		{
			return falseNegatives;
		}

		double getAccuracy() const
		{
			return static_cast<double>(truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
		}

		double getSpecificity() const
		{
			return static_cast<double>(trueNegatives) / (trueNegatives + falsePositives);
		}

		double getPrecision() const
		{
			return static_cast<double>(truePositives) / (truePositives + falsePositives);
		}

		double getRecall() const
		{
			return static_cast<double>(truePositives) / (truePositives + falseNegatives);
		}

		double getF1Score() const
		{
			return 2 * getPrecision() * getRecall() / (getPrecision() + getRecall());
		}

		double getBalancedAccuracy() const
		{
			return 0.5 * (getSpecificity() + getRecall());
		}

		void Clear()
		{
			truePositives = 0;
			trueNegatives = 0;
			falsePositives = 0;
			falseNegatives = 0;
		}

	protected:
		int truePositives = 0;
		int trueNegatives = 0;
		int falsePositives = 0;
		int falseNegatives = 0;
	};

}


