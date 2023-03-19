#pragma once

#include "NeuralSublayer.h"

// the typical neuron is nothing else than a generalized linear model with a single output 

template<class Solver = AdamSolver<>>
class Neuron : public NeuralSublayer<Solver>
{
public:
	typedef NeuralSublayer<Solver> BaseType;

	Neuron(int szi = 1) 
		: BaseType(szi, 1)
	{
	}
};

