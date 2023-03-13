#pragma once

#include "GeneralizedLinearModel.h"

// the generalized one with an identity link function
// this one is truly linear (in the parameters, one can get out of it polynomial regression as it will be obvious from the examples, this is a source of confusion for some)

template<typename InputType, typename OutputType, typename WeightsType, class Solver, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType> class GeneralLinearModel 
	: public GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
{
public:
	typedef GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType> baseType;

	GeneralLinearModel(int szi = 1, int szo = 1)
		: baseType(szi, szo)
	{
	}
};

