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

		long long int getTruePositives() const
		{
			return truePositives;
		}

		long long int getTrueNegatives() const
		{
			return trueNegatives;
		}

		long long int getFalsePositives() const
		{
			return falsePositives;
		}

		long long int getFalseNegatives() const
		{
			return falseNegatives;
		}

		double getAccuracy() const
		{
			return static_cast<double>(truePositives + trueNegatives) / static_cast<double>(truePositives + trueNegatives + falsePositives + falseNegatives);
		}

		double getSpecificity() const
		{
			return static_cast<double>(trueNegatives) / static_cast<double>(trueNegatives + falsePositives);
		}

		double getPrecision() const
		{
			return static_cast<double>(truePositives) / static_cast<double>(truePositives + falsePositives);
		}

		double getRecall() const
		{
			return static_cast<double>(truePositives) / static_cast<double>(truePositives + falseNegatives);
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

		void PrintStatistics(const std::string& name)
		{
			std::cout << std::endl << name << " true positives: " << getTruePositives() << ", true negatives: " << getTrueNegatives() << ", false positives: " << getFalsePositives() << ", false negatives: " << getFalseNegatives() << std::endl;

			std::cout << name << " accuracy: " << getAccuracy() << std::endl;
			std::cout << name << " specificity: " << getSpecificity() << std::endl;
			std::cout << name << " precision: " << getPrecision() << std::endl;
			std::cout << name << " recall: " << getRecall() << std::endl;
			std::cout << name << " F1 score: " << getF1Score() << std::endl;

			std::cout << std::endl;
		}

	protected:
		long long int truePositives = 0;
		long long int trueNegatives = 0;
		long long int falsePositives = 0;
		long long int falseNegatives = 0;
	};

}


