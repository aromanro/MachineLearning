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
