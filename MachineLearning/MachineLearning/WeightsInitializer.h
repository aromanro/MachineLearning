#pragma once

#include <random>

namespace Initializers
{

	class WeightsInitializerInterface
	{
	public:
		virtual ~WeightsInitializerInterface() = default;

		virtual double get(int nrIn = 1, int nrOut = 1) = 0;
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
		WeightsInitializerZero() = default;

		double get(int nrIn = 1, int nrOut = 1) override
		{
			return 0;
		}
	};

	class WeightsInitializerForXorNetwork : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerForXorNetwork() = default;

		double get(int nrIn = 1, int nrOut = 1) override
		{
			double v = dist(rde);

			const unsigned long long int r = neg % 2;

			if (r)
				v *= -1;

			++neg;

			return v;
		}

	private:
		std::uniform_real_distribution<> dist{0.6, 0.9};
		unsigned long long int neg = 0;
	};

	class WeightsInitializerUniform : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerUniform(double low = -1., double high = 1.)
			: dist(low, high)
		{
		}

		double get(int nrIn = 1, int nrOut = 1) override
		{
			return dist(rde);
		}

	private:
		std::uniform_real_distribution<> dist;
	};

	class WeightsInitializerXavierUniform : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerXavierUniform() = default;
		
		double get(int nrIn = 1, int nrOut = 1) override
		{
			const double x = 1. / sqrt(nrIn);
			std::uniform_real_distribution<> dist(-x, x);
			return dist(rde);
		}
	};

	class WeightsInitializerHeUniform : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerHeUniform() = default;

		double get(int nrIn = 1, int nrOut = 1) override
		{
			const double x = sqrt(2. / nrIn);
			std::uniform_real_distribution<> dist(-x, x);
			return dist(rde);
		}
	};

	class WeightsInitializerGlorotUniform : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerGlorotUniform() = default;

		double get(int nrIn = 1, int nrOut = 1) override
		{
			const double x = sqrt(6. / (nrIn + nrOut));
			std::uniform_real_distribution<> dist(-x, x);
			return dist(rde);
		}
	};

	class WeightsInitializerXavierNormal : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerXavierNormal() = default;

		double get(int nrIn = 1, int nrOut = 1) override
		{
			const double x = 1. / sqrt(nrIn);
			std::normal_distribution<> dist(0, x);
			return dist(rde);
		}
	};

	class WeightsInitializerHeNormal : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerHeNormal() = default;


		double get(int nrIn = 1, int nrOut = 1) override
		{
			const double x = sqrt(2. / nrIn);
			std::normal_distribution<> dist(0, x);
			return dist(rde);
		}
	};

	class WeightsInitializerGlorotNormal : public WeightsInitializerImpl
	{
	public:
		WeightsInitializerGlorotNormal() = default;


		double get(int nrIn = 1, int nrOut = 1) override
		{
			const double x = sqrt(6. / (nrIn + nrOut));
			std::normal_distribution<> dist(0, x);
			return dist(rde);
		}
	};

}
