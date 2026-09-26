// MachineLearning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdio.h>

#include "Tests.h"
#include "RuntimeSettings.h"

int main()
{
	std::cout << "Eigen thread limit: " << RuntimeSettings::ConfigureEigenThreads();
#ifndef EIGEN_HAS_OPENMP
	std::cout << " (OpenMP disabled in this build)";
#endif
	std::cout << std::endl;
	if (!AllTests()) return -1;

	getchar();

	return 0;
}


