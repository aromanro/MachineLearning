#pragma once

#include "GeneralizedLinearModel.h"

// a special kind of generalized linear regression, with a sigmoid function as the link function and a logistic loss (which is the cross entropy loss for a special case of 'expected' values being either 0 or 1)

typedef BinaryCrossEntropyLoss<double/*, Eigen::RowVectorXd*/> LogLoss;

template<typename InputType, typename OutputType, typename WeightsType, class Solver, class BatchInputType, class BatchOutputType = BatchInputType> class LogisticRegression
	: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType, SigmoidFunction<InputType, WeightsType>, LogLoss>
{
public:
};

