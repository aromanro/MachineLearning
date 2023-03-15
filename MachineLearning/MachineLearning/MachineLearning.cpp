// MachineLearning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include "Tests.h"

int main()
{
	SimpleLinearRegressionTests();

	std::cout << std::endl << "With gradient descent: " << std::endl << std::endl;

	LinearRegressionTests();
	
	std::cout << std::endl << "Logistic regression: " << std::endl << std::endl;

	LogisticRegressionTests();

	return 0;
}


