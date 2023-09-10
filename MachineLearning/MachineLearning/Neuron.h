#pragma once

#include "NeuralSublayer.h"
#include "GradientSolvers.h"

namespace NeuralNetworks
{

	// the typical neuron is nothing else than a generalized linear model with a single output 

	template<class Solver = SGD::AdamWSolver<>>
	class Neuron : public NeuralSublayer<Solver>
	{
	public:
		using BaseType = NeuralSublayer<Solver>;

		explicit Neuron(int szi = 1)
			: BaseType(szi, 1)
		{
		}
	};

}
