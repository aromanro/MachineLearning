#include "Tests.h"
#include "WeightsInitializer.h"
#include "NeuralNetwork.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "MNISTDatabase.h"
#include "Softmax.h"

bool XORNeuralNetworksTests()
{
	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distBool(0, 1);

	const int nrTests = 1;

	// this alleviates the convergence issue
	// there are 16 local minima for xor where the 3 neurons network could get 'stuck'
	// ocassionally it might reach one but from my tests it can sometimes get out of it
	Initializers::WeightsInitializerForXorNetwork weightsInitializer;

	const double alpha = 0.01;
	const double beta1 = 0.7;
	const double beta2 = 0.9;
	const double lim = 0.1;

	// increasing the number of neurons from the minimum of 2 will help with the convergence
	// because the dimension of space is increased, there are paths out of the former local minimums that exist in the small dimensional space
	// using the special weights initializer is no longer necessary, but it shouldn't hurt either, in fact it seems to still help with the convergence
	// different parameters might be needed for an efficient convergence but I did not play with those yet
	const int numHiddenNeurons = 2;

	const int batchSize = 4;

	std::cout << std::endl << "XOR with neural network implemented 'in place' out of generalized linear models (last layer neuron being a logistic regression)" << std::endl << std::endl;

	Eigen::MatrixXd t(numHiddenNeurons, batchSize); // bogus target for hidden layer, won't be used except for its size
	Eigen::MatrixXd x, y;
	Eigen::VectorXd in(2);

	// try a simple neural network to solve the xor:
	int failures = 0;
	for (int trial = 0; trial < nrTests; ++trial)
	{
		std::cout << std::endl << "Trial: " << trial << std::endl << std::endl;

		// RMSProp or momentum also work

		// works with some other last neuron, such as one that has a tanh activation function, but I like the logistic one more
		//typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::TanhFunction<>> LastLayerRegressionAdamSolver;
		//GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, LastLayerRegressionAdamSolver> modelLastLayer(2, 1);

		GLM::LogisticRegression<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, SGD::LogisticRegressionAdamSolver> modelLastLayer(numHiddenNeurons, 1);

		modelLastLayer.getSolver().alpha = alpha;
		//modelLastLayer.getSolver().beta = beta1; // for RMSPropSolver set alpha to 0.001, otherwise it can stick into a local minimum, for momentum alpha = 0.1 seems to work
		modelLastLayer.getSolver().beta1 = beta1;
		modelLastLayer.getSolver().beta2 = beta2;
		modelLastLayer.getSolver().lim = lim;

		modelLastLayer.getSolver().firstLayer = false;

		// kind of works with tanh as well, it just seems to have a bigger chance to end up in a local minimum
		// works with others, too, but they might need some other parameters (for example, smaller aplha)
		//typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		GLM::GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerModel(2, numHiddenNeurons);

		hiddenLayerModel.getSolver().alpha = alpha;
		//hiddenLayerModel.getSolver().beta = beta1; // for RMSPropSolver set alpha to 0.001, otherwise it can stick into a local minimum, for momentum alpha = 0.1 seems to work
		hiddenLayerModel.getSolver().beta1 = beta1;
		hiddenLayerModel.getSolver().beta2 = beta2;
		hiddenLayerModel.getSolver().lim = lim;

		hiddenLayerModel.getSolver().lastLayer = false;

		hiddenLayerModel.Initialize(weightsInitializer);

		x.resize(2, batchSize);
		y.resize(1, batchSize);

		int lowLoss = 0;
		for (int i = 0; i <= 10000000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				const int x1 = distBool(rde);
				const int x2 = distBool(rde);
				x(0, b) = x1;
				x(1, b) = x2;

				y(0, b) = (x1 ^ x2);
			}

			// forward propagation
			// first layer:

			hiddenLayerModel.AddBatchNoParamsAdjustment(x, t);
			Eigen::MatrixXd pred = hiddenLayerModel.getPrediction();

			// forward and backward for the last layer:
			const Eigen::MatrixXd grad = modelLastLayer.AddBatch(pred, y);

			// now backpropagate the gradient to previous layer:
			pred = modelLastLayer.BackpropagateBatch(grad);

			// now do the adjustments as well in the first layer
			hiddenLayerModel.AddBatch(x, pred);

			if (i % 10000 == 0)
			{
				const double loss = modelLastLayer.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;

				if (loss < 1E-2)
					++lowLoss;
				else
					lowLoss = 0;

				if (lowLoss > 5) break;
			}
		}

		if (lowLoss <= 5)
			std::cout << "Failure to converge!" << std::endl;

		in(0) = 0;
		in(1) = 0;

		x = hiddenLayerModel.Predict(in);
		double res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 0 0 = " << res << std::endl;

		if (lowLoss > 5 && res >= 0.5)
			lowLoss = 0;

		in(0) = 0;
		in(1) = 1;

		x = hiddenLayerModel.Predict(in);
		res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 0 1 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 0;

		x = hiddenLayerModel.Predict(in);
		res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 1 0 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 1;

		x = hiddenLayerModel.Predict(in);
		res = modelLastLayer.Predict(x)(0);
		std::cout << "XOR 1 1 = " << res << std::endl;

		if (res >= 0.5)
			lowLoss = 0;

		//if (lowLoss < 5) return false;

		if (lowLoss <= 5) ++failures;
	}

	std::cout << std::endl << "Failures: " << failures << std::endl;

	std::cout << std::endl << std::endl << "XOR with the multilayer perceptron implementation" << std::endl << std::endl;

	const int failures_first = failures;


	failures = 0;
	x.resize(2, batchSize);
	y.resize(1, batchSize);

	for (int trial = 0; trial < nrTests; ++trial)
	{
		std::cout << std::endl << "Trial: " << trial << std::endl << std::endl;

		// with more neurons and even more layers it still works, for example { 2, 7, 5, 1 }, for some complex setup the initialization of weights should probably left to default
		NeuralNetworks::MultilayerPerceptron<> neuralNetwork({ 2, numHiddenNeurons, 1 });

		neuralNetwork.setParams({ alpha, lim, beta1, beta2 });
		neuralNetwork.InitializHiddenLayers(weightsInitializer);

		int lowLoss = 0;
		for (int i = 0; i <= 10000000; ++i)
		{
			for (int b = 0; b < batchSize; ++b)
			{
				const int x1 = distBool(rde);
				const int x2 = distBool(rde);
				x(0, b) = x1;
				x(1, b) = x2;

				y(0, b) = (x1 ^ x2);
			}

			neuralNetwork.ForwardBackwardStep(x, y);

			if (i % 10000 == 0)
			{
				const double loss = neuralNetwork.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;

				if (loss < 1E-2)
					++lowLoss;
				else
					lowLoss = 0;

				if (lowLoss > 5) break;
			}
		}

		if (lowLoss <= 5)
			std::cout << "Failure to converge!" << std::endl;

		in(0) = 0;
		in(1) = 0;
		double res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 0 0 = " << res << std::endl;

		if (res >= 0.5)
			lowLoss = 0;

		in(0) = 0;
		in(1) = 1;
		res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 0 1 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 0;
		res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 1 0 = " << res << std::endl;

		if (res <= 0.5)
			lowLoss = 0;

		in(0) = 1;
		in(1) = 1;
		res = neuralNetwork.Predict(in)(0);
		std::cout << "XOR 1 1 = " << res << std::endl;

		if (res >= 0.5)
			lowLoss = 0;

		if (lowLoss <= 5) ++failures;
	}

	std::cout << std::endl << "Failures: " << failures << std::endl;

	return failures_first == 0 && failures == 0;
}
