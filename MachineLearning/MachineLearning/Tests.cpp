#include "Tests.h"

double linearFunction(double x)
{
	return 3. * x + 2.;
}

double linearFunction2(double x)
{
	return 7. * x - 4.;
}

double linearFunction3(double x)
{
	return -2. * x + 1.;
}

double quadraticFunction(double x)
{
	return (x - 45) * (x - 50) / 50.;
}

double polyFunction(double x)
{
	return (x - 5) * (x - 60) * (x - 80) / 1000.;
}


bool AllTests()
{
//#define ONLY_NEURAL 1
#ifdef ONLY_NEURAL
	return NeuralNetworksTests();
#else
	bool res = SimpleLinearRegressionTests();

	if (res)
	{
		std::cout << std::endl << "With gradient descent: " << std::endl << std::endl;

		res = LinearRegressionTests();

		if (res)
		{
			std::cout << std::endl << "Logistic regression: " << std::endl << std::endl;

			res = LogisticRegressionTests();

			if (res)
			{
				std::cout << std::endl << "Neural: " << std::endl << std::endl;

				res = NeuralNetworksTests();
			}
		}
	}

	return res;
#endif	
}