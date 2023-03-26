#include "Tests.h"
#include "WeightsInitializer.h"
#include "NeuralNetwork.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "Softmax.h"


bool XORNeuralNetworksTests()
{
	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distBool(0, 1);
	
	const int nrTests = 3;

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

		modelLastLayer.solver.alpha = alpha;
		//modelLastLayer.solver.beta = beta1; // for RMSPropSolver set alpha to 0.001, otherwise it can stick into a local minimum, for momentum alpha = 0.1 seems to work
		modelLastLayer.solver.beta1 = beta1;
		modelLastLayer.solver.beta2 = beta2;
		modelLastLayer.solver.lim = lim;

		modelLastLayer.solver.firstLayer = false;

		// kind of works with tanh as well, it just seems to have a bigger chance to end up in a local minimum
		// works with others, too, but they might need some other parameters (for example, smaller aplha)
		//typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		typedef SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>> HiddenLayerRegressionAdamSolver;
		GLM::GeneralizedLinearModel<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, HiddenLayerRegressionAdamSolver> hiddenLayerModel(2, numHiddenNeurons);

		hiddenLayerModel.solver.alpha = alpha;
		//hiddenLayerModel.solver.beta = beta1; // for RMSPropSolver set alpha to 0.001, otherwise it can stick into a local minimum, for momentum alpha = 0.1 seems to work
		hiddenLayerModel.solver.beta1 = beta1;
		hiddenLayerModel.solver.beta2 = beta2;
		hiddenLayerModel.solver.lim = lim;

		hiddenLayerModel.solver.lastLayer = false;

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
				double loss = modelLastLayer.getLoss() / batchSize;
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
		NeuralNetworks::MultilayerPerceptron<> neuralNetwork({2, numHiddenNeurons, 1});

		neuralNetwork.setParams({alpha, lim, beta1, beta2});
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
				double loss = neuralNetwork.getLoss() / batchSize;
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


bool IrisNeuralNetworkTest()
{
	std::cout << std::endl << "Neural Network test for the Iris dataset, Setosa is lineary separable from the other two, but the others two cannot be linearly separated" << std::endl << std::endl;

	Utils::IrisDataset irisDataset;
	irisDataset.setRelativePath("../../Datasets/");
	irisDataset.setDataFileName("iris.data");

	if (!irisDataset.Open()) return false;

	auto records = irisDataset.getAllRecords();
	const int nrTraining = 120;

	// shuffle the data

	std::random_device rd;
	std::mt19937 g(rd());

	// ensure it's shuffled enough to have all enough samples of all classes in the test set
	for (;;)
	{
		int setosa = 0;
		int versicolor = 0;
		int virginica = 0;

		std::shuffle(records.begin(), records.end(), g);
		std::shuffle(records.begin(), records.end(), g);
		std::shuffle(records.begin(), records.end(), g);

		for (auto it = records.begin() + nrTraining; it != records.end(); ++it)
		{
			const auto rec = *it;
			if (std::get<4>(rec) == "Iris-setosa") ++setosa;
			if (std::get<4>(rec) == "Iris-versicolor") ++versicolor;
			if (std::get<4>(rec) == "Iris-virginica") ++virginica;
		}

		if (setosa > 8 && versicolor > 8 && virginica > 8) break;
	}


	//for (auto rec : records)
	//	std::cout << std::get<0>(rec) << ", " << std::get<1>(rec) << ", " << std::get<2>(rec) << ", " << std::get<3>(rec) << ", " << std::get<4>(rec) << std::endl;

	// split the data into training and test sets
	
	std::vector<Utils::IrisDataset::Record> trainingSet(records.begin(), records.begin() + nrTraining);
	std::vector<Utils::IrisDataset::Record> testSet(records.begin() + nrTraining, records.end());

	// create the model
	const int nrOutputs = 3; // 1 only for Setosa, 3 if all three classes are to be predicted

	// normalize the inputs
	Norm::Normalizer normalizer(4, nrOutputs);
	Eigen::MatrixXd x(4, nrTraining);
	Eigen::MatrixXd y(nrOutputs, nrTraining);

	for (int i = 0; i < nrTraining; ++i)
	{
		x(0, i) = std::get<0>(trainingSet[i]);
		x(1, i) = std::get<1>(trainingSet[i]);
		x(2, i) = std::get<2>(trainingSet[i]);
		x(3, i) = std::get<3>(trainingSet[i]);

		y(0, i) = (std::get<4>(trainingSet[i]) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) y(1, i) = (std::get<4>(trainingSet[i]) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) y(2, i) = (std::get<4>(trainingSet[i]) == "Iris-virginica") ? 1 : 0;
	}

	normalizer.AddBatch(x, y);

	const Eigen::MatrixXd avgi = normalizer.getAverageInput();
	const Eigen::MatrixXd istdi = normalizer.getVarianceInput().cwiseSqrt().cwiseInverse();

	for (int i = 0; i < nrTraining; ++i)
	{
		x.col(i) -= avgi;
		x.col(i) = x.col(i).cwiseProduct(istdi);

		trainingSet[i] = std::make_tuple(x(0, i), x(1, i), x(2, i), x(3, i), std::get<4>(trainingSet[i]));
	}

	x.resize(4, 1);
	for (int i = 0; i < testSet.size(); ++i)
	{
		x(0, 0) = std::get<0>(testSet[i]);
		x(1, 0) = std::get<1>(testSet[i]);
		x(2, 0) = std::get<2>(testSet[i]);
		x(3, 0) = std::get<3>(testSet[i]);
		

		x.col(0) -= avgi;
		x.col(0) = x.col(0).cwiseProduct(istdi);

		testSet[i] = std::make_tuple(x(0, 0), x(1, 0), x(2, 0), x(3, 0), std::get<4>(testSet[i]));
	}

	// more layers can be added, and/or made wider, but it will take more time to train. One configuration that I tried: { 4, 246, 512, 127, 63, 27, 9, nrOutputs }
	NeuralNetworks::MultilayerPerceptron<> neuralNetwork({ 4, 27, 9, nrOutputs });

	const double alpha = 0.001;
	const double beta1 = 0.7;
	const double beta2 = 0.9;
	const double lim = 1;

	neuralNetwork.setParams({ alpha, lim, beta1, beta2 });

	//Initializers::WeightsInitializerUniform initializer(-0.01, 0.01);
	//neuralNetwork.Initialize(initializer);


	// train the model

	const int batchSize = 64;

	Eigen::MatrixXd in(4, batchSize);
	Eigen::MatrixXd out(nrOutputs, batchSize);

	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distIntBig(0, nrTraining - 1);
	for (int i = 0; i <= 10000; ++i)
	{
		for (int b = 0; b < batchSize; ++b)
		{
			const int ind = distIntBig(rde);
			const auto& record = trainingSet[ind];

			in(0, b) = std::get<0>(record);
			in(1, b) = std::get<1>(record);
			in(2, b) = std::get<2>(record);
			in(3, b) = std::get<3>(record);

			out(0, b) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
			if (nrOutputs > 1) out(1, b) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
			if (nrOutputs > 2) out(2, b) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;
		}
		neuralNetwork.ForwardBackwardStep(in, out);
		if (i % 1000 == 0)
		{
			double loss = neuralNetwork.getLoss() / batchSize;
			std::cout << "Loss: " << loss << std::endl;
		}
	}


	Utils::TestStatistics setosaStats;
	Utils::TestStatistics versicolorStats;
	Utils::TestStatistics virginicaStats;


	// test the model

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	for (const auto& record : trainingSet)
	{
		in(0, 0) = std::get<0>(record);
		in(1, 0) = std::get<1>(record);
		in(2, 0) = std::get<2>(record);
		in(3, 0) = std::get<3>(record);

		out(0, 0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1, 0) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2, 0) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

		Eigen::VectorXd res = neuralNetwork.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);
	}

	std::cout << std::endl << "Setosa true positives: " << setosaStats.getTruePositives() << ", true negatives: " << setosaStats.getTrueNegatives() << ", false positives: " << setosaStats.getFalsePositives() << ", false negatives: " << setosaStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor true positives: " << versicolorStats.getTruePositives() << ", true negatives: " << versicolorStats.getTrueNegatives() << ", false positives: " << versicolorStats.getFalsePositives() << ", false negatives: " << versicolorStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica true positives: " << virginicaStats.getTruePositives() << ", true negatives: " << virginicaStats.getTrueNegatives() << ", false positives: " << virginicaStats.getFalsePositives() << ", false negatives: " << virginicaStats.getFalseNegatives() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa accuracy: " << setosaStats.getAccuracy() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor accuracy: " << versicolorStats.getAccuracy() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica accuracy: " << virginicaStats.getAccuracy() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa specificity: " << setosaStats.getSpecificity() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor specificity: " << versicolorStats.getSpecificity() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica specificity: " << virginicaStats.getSpecificity() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa precision: " << setosaStats.getPrecision() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor precision: " << versicolorStats.getPrecision() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica precision: " << virginicaStats.getPrecision() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa recall: " << setosaStats.getRecall() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor recall: " << versicolorStats.getRecall() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica recall: " << virginicaStats.getRecall() << std::endl;
	std::cout << std::endl;

	setosaStats.Clear();
	versicolorStats.Clear();
	virginicaStats.Clear();

	std::cout << std::endl << "Test set:" << std::endl;

	for (const auto& record : testSet)
	{
		in(0, 0) = std::get<0>(record);
		in(1, 0) = std::get<1>(record);
		in(2, 0) = std::get<2>(record);
		in(3, 0) = std::get<3>(record);

		out(0, 0) = (std::get<4>(record) == "Iris-setosa") ? 1 : 0;
		if (nrOutputs > 1) out(1, 0) = (std::get<4>(record) == "Iris-versicolor") ? 1 : 0;
		if (nrOutputs > 2) out(2, 0) = (std::get<4>(record) == "Iris-virginica") ? 1 : 0;

		Eigen::VectorXd res = neuralNetwork.Predict(in.col(0));
		setosaStats.AddPrediction(res(0) > 0.5, out(0, 0) > 0.5);
		if (nrOutputs > 1) versicolorStats.AddPrediction(res(1) > 0.5, out(1, 0) > 0.5);
		if (nrOutputs > 2) virginicaStats.AddPrediction(res(2) > 0.5, out(2, 0) > 0.5);
	}

	std::cout << std::endl << "Setosa true positives: " << setosaStats.getTruePositives() << ", true negatives: " << setosaStats.getTrueNegatives() << ", false positives: " << setosaStats.getFalsePositives() << ", false negatives: " << setosaStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor true positives: " << versicolorStats.getTruePositives() << ", true negatives: " << versicolorStats.getTrueNegatives() << ", false positives: " << versicolorStats.getFalsePositives() << ", false negatives: " << versicolorStats.getFalseNegatives() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica true positives: " << virginicaStats.getTruePositives() << ", true negatives: " << virginicaStats.getTrueNegatives() << ", false positives: " << virginicaStats.getFalsePositives() << ", false negatives: " << virginicaStats.getFalseNegatives() << std::endl;
    std::cout << std::endl;
	
	std::cout << "Setosa accuracy: " << setosaStats.getAccuracy() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor accuracy: " << versicolorStats.getAccuracy() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica accuracy: " << virginicaStats.getAccuracy() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa specificity: " << setosaStats.getSpecificity() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor specificity: " << versicolorStats.getSpecificity() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica specificity: " << virginicaStats.getSpecificity() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa precision: " << setosaStats.getPrecision() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor precision: " << versicolorStats.getPrecision() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica precision: " << virginicaStats.getPrecision() << std::endl;
	std::cout << std::endl;

	std::cout << "Setosa recall: " << setosaStats.getRecall() << std::endl;
	if (nrOutputs > 1) std::cout << "Versicolor recall: " << versicolorStats.getRecall() << std::endl;
	if (nrOutputs > 2) std::cout << "Virginica recall: " << virginicaStats.getRecall() << std::endl;
	std::cout << std::endl;

	return true;
}

bool NeuralNetworksTests()
{
	return XORNeuralNetworksTests() && IrisNeuralNetworkTest();
}