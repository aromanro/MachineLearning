#pragma once


#include "SimpleLinearRegression.h"
#include "GradientSolvers.h"
#include "LogisticRegression.h"
#include "Normalizer.h"

#include "DataFile.h"
#include "GnuPlot.h"

#include "CSVDataFile.h"
#include "TestStatistics.h"


#include <random>
#include <chrono>

#define _USE_MATH_DEFINES 1
#include <math.h>

// EMNIST data loading and splitting
bool LoadData(std::vector<std::pair<std::vector<double>, uint8_t>>& trainingRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& testRecords, bool augment = false);
bool LoadData(std::vector<std::pair<std::vector<double>, uint8_t>>& trainingRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& validationRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& testRecords, bool augment = false, double percentage = 0.95);
void SetDataIntoMatrices(const std::vector<std::pair<std::vector<double>, uint8_t>>& records, Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs);

void Shuffle(std::vector<Utils::IrisDataset::Record>& records, int nrTraining);


double linearFunction(double x);
double linearFunction2(double x);
double linearFunction3(double x);
double quadraticFunction(double x);
double polyFunction(double x);

bool SimpleLinearRegressionTests();
bool LinearRegressionTests();
bool LogisticRegressionTests();
bool SoftmaxTests();

bool XORNeuralNetworksTests();
bool IrisNeuralNetworkTest();
bool NeuralNetworkTestsMNIST();
bool NeuralNetworksTests();

bool AllTests();

