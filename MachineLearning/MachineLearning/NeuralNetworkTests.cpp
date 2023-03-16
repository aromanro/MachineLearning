#include "Tests.h"


bool NeuralNetworksTests()
{
	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distBool(0, 1);

	const double alpha = 0.3;

	const double beta1 = 0.7;
	const double beta2 = 0.9;


	// try a simple neural network to solve the xor:

	LogisticRegression<> logisticModelLastLayer(2, 1);
	logisticModelLastLayer.solver.alpha = alpha;
	logisticModelLastLayer.solver.beta1 = beta1;
	logisticModelLastLayer.solver.beta2 = beta2;

	logisticModelLastLayer.solver.firstLayer = false;

	// works also with LeakyRELUFunction, but not as well, at least with the parameters used above
	typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, TanhFunction<>> HiddenLayerRegressionAdamSolver;
	GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerModel(2, 2);

	hiddenLayerModel.solver.alpha = alpha;
	hiddenLayerModel.solver.beta1 = beta1;
	hiddenLayerModel.solver.beta2 = beta2;

	hiddenLayerModel.solver.lastLayer = false;

	const int batchSize = 32;
	Eigen::MatrixXd t(2, batchSize); // bogus target, won't be used
	Eigen::MatrixXd x, y;
	

	x.resize(2, batchSize);
	y.resize(1, batchSize);

	for (int i = 0; i <= 1000; ++i)
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
		Eigen::MatrixXd grad = logisticModelLastLayer.AddBatch(pred, y);

		// now backpropagate the gradient to previous layer:
		pred = logisticModelLastLayer.BackpropagateBatch(grad);

		// now do the adjustments as well in the first layer
		hiddenLayerModel.AddBatch(x, pred);

		if (i % 100 == 0)
		{
			double loss = logisticModelLastLayer.getLoss() / batchSize;
			std::cout << "Loss: " << loss << std::endl;
		}
	}


	Eigen::VectorXd in(2);
	in(0) = 0;
	in(1) = 0;

	x = hiddenLayerModel.Predict(in);
	std::cout << "XOR 0 0 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

	in(0) = 0;
	in(1) = 1;

	x = hiddenLayerModel.Predict(in);
	std::cout << "XOR 0 1 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

	in(0) = 1;
	in(1) = 0;

	x = hiddenLayerModel.Predict(in);
	std::cout << "XOR 1 0 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

	in(0) = 1;
	in(1) = 1;

	x = hiddenLayerModel.Predict(in);
	std::cout << "XOR 1 1 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

	return true;
}