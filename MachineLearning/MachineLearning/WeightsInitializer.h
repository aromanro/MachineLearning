#pragma once

#include <random>

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


class WeightsInitializerForXorNetwork : public WeightsInitializerImpl
{
public:
	WeightsInitializerForXorNetwork() 
		: dist(0.6, 0.9), neg(0)
	{
	}

	double get() override
	{
		double v = dist(rde);

		// there are 4 weights in the hidden layer for the minimal xor solving neural network
		unsigned long long int r = neg % 4;

		if (r <= 1)
			v *= -1;
		
		++neg;
		
		return v;
	}

protected:
	std::uniform_real_distribution<> dist;
	unsigned long long int neg;
};
