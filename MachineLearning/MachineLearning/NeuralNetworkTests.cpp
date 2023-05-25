#include "Tests.h"


bool NeuralNetworksTests()
{
	return XORNeuralNetworksTests() && IrisNeuralNetworkTest() && NeuralNetworkTestsMNIST();
}