#pragma once


#include "SimpleLinearRegression.h"
#include "GradientSolvers.h"
#include "LogisticRegression.h"
#include "Normalizer.h"

#include "DataFile.h"
#include "GnuPlot.h"


#include <random>
#include <chrono>

#define _USE_MATH_DEFINES 1
#include <math.h>


bool LoadData(std::vector<std::pair<std::vector<double>, uint8_t>>& trainingRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& testRecords);
void SetDataIntoMatrices(const std::vector<std::pair<std::vector<double>, uint8_t>>& records, Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs);

double linearFunction(double x);
double linearFunction2(double x);
double linearFunction3(double x);
double quadraticFunction(double x);
double polyFunction(double x);

bool SimpleLinearRegressionTests();
bool LinearRegressionTests();
bool LogisticRegressionTests();
bool SoftmaxTests();
bool NeuralNetworksTests();

bool AllTests();

