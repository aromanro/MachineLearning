#pragma once

class NormalizerBase
{
public:
	NormalizerBase()
	{
	}

	double getAverage() const
	{
		return average;
	}

	double getStd() const
	{
		return std;
	}

protected:
	double average = 0.;
	double std = 1.;
};

