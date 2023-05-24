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

		// hit
		long long int getTruePositives() const
		{
			return truePositives;
		}

		// correct rejection
		long long int getTrueNegatives() const
		{
			return trueNegatives;
		}

		// type I error
		long long int getFalsePositives() const
		{
			return falsePositives;
		}

		// type II error
		long long int getFalseNegatives() const
		{
			return falseNegatives;
		}

		long long int getPositives() const
		{
			return truePositives + falseNegatives;
		}

		long long int getNegatives() const
		{
			return trueNegatives + falsePositives;
		}

		long long int getPositivePredictions() const
		{
			return truePositives + falsePositives;
		}

		long long int getNegativePredictions() const
		{
			return trueNegatives + falseNegatives;
		}


		long long int getTotal() const
		{
			return getPositives() + getNegatives();
		}

		double getPrevalence() const
		{
			return static_cast<double>(getPositives()) / static_cast<double>(getTotal());
		}

		// correct predictions / total predictions
		double getAccuracy() const
		{
			return static_cast<double>(truePositives + trueNegatives) / static_cast<double>(getTotal());
		}

		// true negative rate, selectivity
		double getSpecificity() const
		{
			return static_cast<double>(trueNegatives) / static_cast<double>(getNegatives());
		}

		// positive predictive value
		double getPrecision() const
		{
			return static_cast<double>(truePositives) / static_cast<double>(getPositivePredictions());
		}

		// 1 - precision
		double getFalseDiscoveryRate() const
		{
			return static_cast<double>(falsePositives) / static_cast<double>(getPositivePredictions());
		}

		// also called true positive rate, hit rate, sensitivity
		double getRecall() const
		{
			return static_cast<double>(truePositives) / static_cast<double>(getPositives());
		}

		// false negative rate: 1 - true positive rate
		double getMissRate() const
		{
			return static_cast<double>(falseNegatives) / static_cast<double>(getPositives());
		}

		// false positive rate, fall-out
		double getFalsePositiveRate() const
		{
			return static_cast<double>(falsePositives) / static_cast<double>(getNegatives());
		}

		// 1 - negative predictive value
		double getFalseOmissionRate() const
		{
			return static_cast<double>(falseNegatives) / static_cast<double>(getNegativePredictions());
		}

		// 1 - false omission rate
		double getNegativePredictiveValue() const
		{
			return static_cast<double>(trueNegatives) / static_cast<double>(getNegativePredictions());
		}

		// true positive rate / false positive rate
		double getPositiveLikelihoodRatio() const
		{
			return getRecall() / getFalsePositiveRate();
		}

		// false negative rate / true negative rate
		double getNegativeLikelihoodRatio() const
		{
			return getMissRate() / getSpecificity();
		}

		// diagnostic odds ratio
		double getDiagnosticOddsRatio() const
		{
			return getPositiveLikelihoodRatio() / getNegativeLikelihoodRatio();
		}

		// critical success index
		double getThreatScore() const
		{
			return static_cast<double>(truePositives) / static_cast<double>(getPositives() + falsePositives);
		}

		// prevalence threshold
		double getPrevalenceThreshold() const
		{
			const double sfpr = sqrt(getFalsePositiveRate());

			return sfpr / (sqrt(getMissRate()) + sfpr);
		}

		// informedness
		double getInformedness() const
		{
			return getRecall() +  getSpecificity() - 1.;
		}

		// markedness
		double getMarkedness() const
		{
			return getPrecision() + getNegativePredictiveValue() - 1.;
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

		void Add(const TestStatistics& other)
		{
			truePositives += other.truePositives;
			trueNegatives += other.trueNegatives;
			falsePositives += other.falsePositives;
			falseNegatives += other.falseNegatives;
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

	private:
		long long int truePositives = 0;
		long long int trueNegatives = 0;
		long long int falsePositives = 0;
		long long int falseNegatives = 0;
	};

}


