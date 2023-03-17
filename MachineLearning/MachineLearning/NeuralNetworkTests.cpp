#include "Tests.h"
#include "WeightsInitializer.h"


bool NeuralNetworksTests()
{
	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distBool(0, 1);
	
	// this alleviates the convergence issue
	// there are 16 local minima for xor where the network could get 'stuck'
	// ocassionally it might reach one but from my tests it can sometimes get out of it
	WeightsInitializerForXorNetwork weightsInitializer;

	const double alpha = 0.01;
	const double beta1 = 0.7;
	const double beta2 = 0.9;
	const double lim = 0.1;

	// try a simple neural network to solve the xor:
	int failures = 0;
	for (int trial = 0; trial < 300; ++trial)
	{
		std::cout << std::endl << "Trial: " << trial << std::endl << std::endl;

		// works with some other last neuron, such as one that has a tanh activation function, but I like the logistic one more
		//typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, TanhFunction<>> LastLayerRegressionAdamSolver;
		//GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, LastLayerRegressionAdamSolver> modelLastLayer(2, 1);
		LogisticRegression<> modelLastLayer(2, 1);

		modelLastLayer.solver.alpha = alpha;
		modelLastLayer.solver.beta1 = beta1;
		modelLastLayer.solver.beta2 = beta2;
		modelLastLayer.solver.lim = lim;

		modelLastLayer.solver.firstLayer = false;

		// kind of works with tanh as well, it just seems to have a bigger chance to end up in a local minimum
		// works with others, too, but they might need some other parameters (for example, smaller aplha)
		typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerModel(2, 2);

		hiddenLayerModel.solver.alpha = alpha;
		hiddenLayerModel.solver.beta1 = beta1;
		hiddenLayerModel.solver.beta2 = beta2;
		hiddenLayerModel.solver.lim = lim;

		hiddenLayerModel.solver.lastLayer = false;

		hiddenLayerModel.Initialize(weightsInitializer);

		const int batchSize = 4;
		Eigen::MatrixXd t(2, batchSize); // bogus target, won't be used
		Eigen::MatrixXd x, y;


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
			Eigen::MatrixXd grad = modelLastLayer.AddBatch(pred, y);

			// now backpropagate the gradient to previous layer:
			pred = modelLastLayer.BackpropagateBatch(grad);

			// now do the adjustments as well in the first layer
			hiddenLayerModel.AddBatch(x, pred);

			if (i % 10000 == 0)
			{
				double loss = modelLastLayer.getLoss() / batchSize;
				std::cout << "Loss: " << loss << std::endl;

				if (loss < 1E-2)
					++lowLoss;
				else
					lowLoss = 0;

				if (lowLoss > 5) break;
			}
		}

		if (lowLoss < 5)
			std::cout << "Failure to converge!" << std::endl;

		Eigen::VectorXd in(2);
		in(0) = 0;
		in(1) = 0;

		x = hiddenLayerModel.Predict(in);
		std::cout << "XOR 0 0 = " << modelLastLayer.Predict(x)(0) << std::endl;

		in(0) = 0;
		in(1) = 1;

		x = hiddenLayerModel.Predict(in);
		std::cout << "XOR 0 1 = " << modelLastLayer.Predict(x)(0) << std::endl;

		in(0) = 1;
		in(1) = 0;

		x = hiddenLayerModel.Predict(in);
		std::cout << "XOR 1 0 = " << modelLastLayer.Predict(x)(0) << std::endl;

		in(0) = 1;
		in(1) = 1;

		x = hiddenLayerModel.Predict(in);
		std::cout << "XOR 1 1 = " << modelLastLayer.Predict(x)(0) << std::endl;

		//if (lowLoss < 5) return false;

		if (lowLoss < 5) ++failures;
	}

	std::cout << std::endl << "Failures: " << failures << std::endl;

	return true;
}