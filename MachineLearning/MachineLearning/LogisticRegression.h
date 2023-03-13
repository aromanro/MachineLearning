#pragma once

#include "GeneralizedLinearModel.h"

// a special kind of generalized linear regression, with a sigmoid function as the link function and a logistic loss (which is the cross entropy loss for a special case of 'expected' values being either 0 or 1)

template<typename T> class LogLoss : public BinaryCrossEntropyLoss<T>
{
};


template<typename InputType, typename OutputType, typename WeightsType, class Solver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType> class LogisticRegression
	: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
{
public:
	typedef GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType> baseType;

	LogisticRegression(int szi = 1, int szo = 1)
	{
		baseType::Initialize(szi, szo);
		// W was initialized to some values between -1 and 1, translate them to values between 0 and 1
		baseType::W += WeightsType::Ones(baseType::W.rows(), baseType::W.cols());
		baseType::W *= 0.5;
	}
};

