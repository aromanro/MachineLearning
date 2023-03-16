#include "Tests.h"


bool NeuralNetworksTests()
{
	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distBool(0, 1);

	const double alpha = 0.01;
	const double beta1 = 0.7;
	const double beta2 = 0.9;
	const double lim = 1;

	// try a simple neural network to solve the xor:
	int failures = 0;
	for (int trial = 0; trial < 300; ++trial)
	{
		std::cout << std::endl << "Trial: " << trial << std::endl << std::endl;

		LogisticRegression<> logisticModelLastLayer(2, 1);
		logisticModelLastLayer.solver.alpha = alpha;
		logisticModelLastLayer.solver.beta1 = beta1;
		logisticModelLastLayer.solver.beta2 = beta2;
		logisticModelLastLayer.solver.lim = lim;

		logisticModelLastLayer.solver.firstLayer = false;

		/*
		typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerNeuron1(2, 1);
		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerNeuron2(2, 1);

		hiddenLayerNeuron1.solver.alpha = hiddenLayerNeuron2.solver.alpha = alpha;
		hiddenLayerNeuron1.solver.beta1 = hiddenLayerNeuron2.solver.beta1 = beta1;
		hiddenLayerNeuron1.solver.beta2 = hiddenLayerNeuron2.solver.beta2 = beta2;
		hiddenLayerNeuron1.solver.lim = hiddenLayerNeuron2.solver.lim = lim;

		hiddenLayerNeuron1.solver.lastLayer = hiddenLayerNeuron2.solver.lastLayer = false;

		const int batchSize = 4;
		Eigen::MatrixXd t(1, batchSize); // bogus target, won't be used
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

			hiddenLayerNeuron1.AddBatchNoParamsAdjustment(x, t);
			hiddenLayerNeuron2.AddBatchNoParamsAdjustment(x, t);
			Eigen::MatrixXd pred1 = hiddenLayerNeuron1.getPrediction();
			Eigen::MatrixXd pred2 = hiddenLayerNeuron2.getPrediction();

			Eigen::MatrixXd hidInput(2, batchSize);
			hidInput.row(0) = pred1;
			hidInput.row(1) = pred2;

			// forward and backward for the last layer:
			Eigen::MatrixXd grad = logisticModelLastLayer.AddBatch(hidInput, y); // this also adjusts the weights

			// now backpropagate the gradient to previous layer:
			grad = logisticModelLastLayer.BackpropagateBatch(grad);

			// now do the adjustments in the first layer as well 
			hiddenLayerNeuron1.AddBatch(x, grad.row(0));
			hiddenLayerNeuron2.AddBatch(x, grad.row(1));

			if (i % 10000 == 0)
			{
				double loss = logisticModelLastLayer.getLoss() / batchSize;
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

		x.row(0) = hiddenLayerNeuron1.Predict(in);
		x.row(1) = hiddenLayerNeuron2.Predict(in);
		std::cout << "XOR 0 0 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

		in(0) = 0;
		in(1) = 1;

		x.row(0) = hiddenLayerNeuron1.Predict(in);
		x.row(1) = hiddenLayerNeuron2.Predict(in);
		std::cout << "XOR 0 1 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

		in(0) = 1;
		in(1) = 0;

		x.row(0) = hiddenLayerNeuron1.Predict(in);
		x.row(1) = hiddenLayerNeuron2.Predict(in);
		std::cout << "XOR 1 0 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;

		in(0) = 1;
		in(1) = 1;

		x.row(0) = hiddenLayerNeuron1.Predict(in);
		x.row(1) = hiddenLayerNeuron2.Predict(in);
		std::cout << "XOR 1 1 = " << logisticModelLastLayer.Predict(x)(0) << std::endl;
		*/


		typedef AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerModel(2, 2);

		hiddenLayerModel.solver.alpha = alpha;
		hiddenLayerModel.solver.beta1 = beta1;
		hiddenLayerModel.solver.beta2 = beta2;
		hiddenLayerModel.solver.lim = lim;

		hiddenLayerModel.solver.lastLayer = false;

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
			Eigen::MatrixXd grad = logisticModelLastLayer.AddBatch(pred, y);

			// now backpropagate the gradient to previous layer:
			pred = logisticModelLastLayer.BackpropagateBatch(grad);

			// now do the adjustments as well in the first layer
			hiddenLayerModel.AddBatch(x, pred);

			if (i % 10000 == 0)
			{
				double loss = logisticModelLastLayer.getLoss() / batchSize;
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

		//if (lowLoss < 5) return false;

		if (lowLoss < 5) ++failures;
	}

	std::cout << std::endl << "Failures: " << failures << std::endl;

	return true;
}