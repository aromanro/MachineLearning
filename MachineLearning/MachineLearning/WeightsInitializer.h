#pragma once

#include <random>

namespace Initializers
{

	class WeightsInitializerInterface
	{
	public:
		virtual ~WeightsInitializerInterface() {}

		virtual double get() = 0;
	};

	class WeightsInitializerImpl : public WeightsInitializerInterface
	{
	public:
		WeightsInitializerImpl()
		{
			std::random_device rd;

			rde.seed(rd());
		}

	protected:
		std::mt19937 rde;
	};

	// Warnging: do not use this for a neural network! Symmetry must be broken by random initialization!
	class WeightsInitializerZero : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerZero()
		{
		}

		double get() override
		{
			return 0;
		}
	};

	class WeightsInitializerForXorNetwork : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerForXorNetwork()
			: dist(0.55, 0.95), neg(0)
		{
		}

		double get() override
		{
			double v = dist(rde);

			const unsigned long long int r = neg % 2;

			if (r)
				v *= -1;

			++neg;

			return v;
		}

	protected:
		std::uniform_real_distribution<> dist;
		unsigned long long int neg;
	};

	class WeightsInitializerUniform : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerUniform(double low = -1., double high = 1.)
			: dist(low, high)
		{
		}

		double get() override
		{
			return dist(rde);
		}

	protected:
		std::uniform_real_distribution<> dist;
	};

}
