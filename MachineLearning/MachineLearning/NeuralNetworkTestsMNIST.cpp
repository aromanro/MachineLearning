#include "Tests.h"
#include "WeightsInitializer.h"
#include "NeuralNetwork.h"
#include "CSVDataFile.h"
#include "TestStatistics.h"
#include "MNISTDatabase.h"
#include "Softmax.h"


bool NeuralNetworkTestsMNIST()
{
	std::cout << "MNIST Neural Network Tests, it will take a long time..." << std::endl;

	const int nrInputs = 28 * 28;
	const int nrOutputs = 10;

	// load the data
	Utils::MNISTDatabase minstTrainDataFiles;
	if (!minstTrainDataFiles.Open()) {
		std::cout << "Couldn't load train data" << std::endl;
		return false;
	}

	std::vector<std::pair<std::vector<double>, uint8_t>> trainingRecords = minstTrainDataFiles.ReadAllImagesAndLabels(true);
	minstTrainDataFiles.Close();

	Utils::MNISTDatabase minstTestDataFiles;
	minstTestDataFiles.setImagesFileName("emnist-digits-test-images-idx3-ubyte");
	minstTestDataFiles.setLabelsFileName("emnist-digits-test-labels-idx1-ubyte");
	if (!minstTestDataFiles.Open()) {
		std::cout << "Couldn't load test data" << std::endl;
		return false;
	}

	std::vector<std::pair<std::vector<double>, uint8_t>> testRecords = minstTestDataFiles.ReadAllImagesAndLabels();
	minstTestDataFiles.Close();


	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(trainingRecords.begin(), trainingRecords.end(), g);

	// split the training data into training and validation sets

	const int nrTrainingRecords = static_cast<int>(trainingRecords.size() * 0.95);

	std::vector<std::pair<std::vector<double>, uint8_t>> validationRecords(trainingRecords.begin() + nrTrainingRecords, trainingRecords.end());
	trainingRecords.resize(nrTrainingRecords);

	// normalize the data
	Norm::InputOutputNormalizer<> pixelsNormalizer(nrInputs, nrOutputs);

	Eigen::MatrixXd trainInputs(nrInputs, trainingRecords.size());
	Eigen::MatrixXd trainOutputs(nrOutputs, trainingRecords.size());

	int rec = 0;
	for (const auto& record : trainingRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			trainInputs(i, rec) = record.first[i];

		for (int i = 0; i < nrOutputs; ++i)
			trainOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}

	pixelsNormalizer.AddBatch(trainInputs, trainOutputs);


	Eigen::MatrixXd validationInputs(nrInputs, validationRecords.size());
	Eigen::MatrixXd validationOutputs(nrOutputs, validationRecords.size());
	Eigen::MatrixXd validationRes(nrOutputs, validationRecords.size());


	Eigen::MatrixXd trainStatsOutputs(nrOutputs, validationRecords.size());
	Eigen::MatrixXd trainStatsRes(nrOutputs, validationRecords.size());

	rec = 0;
	for (const auto& record : validationRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			validationInputs(i, rec) = record.first[i];
		for (int i = 0; i < nrOutputs; ++i)
			validationOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}


	Eigen::MatrixXd testInputs(nrInputs, testRecords.size());
	Eigen::MatrixXd testOutputs(nrOutputs, testRecords.size());


	rec = 0;
	for (const auto& record : testRecords)
	{
		for (int i = 0; i < nrInputs; ++i)
			testInputs(i, rec) = record.first[i];

		for (int i = 0; i < nrOutputs; ++i)
			testOutputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}

	// only inputs and only shifting the average

	trainInputs = trainInputs.colwise() - pixelsNormalizer.getAverageInput();
	validationInputs = validationInputs.colwise() - pixelsNormalizer.getAverageInput();
	testInputs = testInputs.colwise() - pixelsNormalizer.getAverageInput();

	// create the model
	// two hidden layers works quite well: { nrInputs, 1000, 100, nrOutputs } - use XavierUniform weights initializer for it - over 98%
	// also tested { nrInputs, 1000, 600, 100, nrOutputs } - use Glorot uniform weights initializer for it, this one I suspect that it needs different parameters and maybe more iterations
	// a single hidden layer, should be fast enough: { nrInputs, 32, nrOutputs } - over 97%
	// for simple ones the xavier initializer works well, for the deeper ones the glorot one is better

	// tanh activation functions can be also used for the hidden layers, seem to work, but I prefer the leaky relu
	// uncomment this and the commented template parameter if you want to try it, but it won't start from a pretrained network that had leaky relu (as the one I commited on github) 
	//typedef  SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::TanhFunction<>> HiddenLayerAlternative;

	NeuralNetworks::MultilayerPerceptron<SGD::SoftmaxRegressionAdamSolver/*, HiddenLayerAlternative*/> neuralNetwork(/*{nrInputs, 1000, 100, nrOutputs}*/{ nrInputs, 1000, 800, 400, 100, nrOutputs }, { 0.2, 0.2, 0.1, 0, 0 }); // don't use dropout right before the softmax layer

	// initialize the model
	double alpha = 0.0015; // non const, so it can be adjusted
	double decay = 0.95;
	const double beta1 = 0.9;
	const double beta2 = 0.95;
	const double lim = 10;

	neuralNetwork.setParams({ alpha, lim, beta1, beta2 });


	int startEpoch = 0; // set it to something different than 0 if you want to continue training

	bool hasPretrained = false;

	if (startEpoch == 0)
	{
		// load some saved model

		if (!neuralNetwork.loadNetwork("../../data/pretrained.net"))
		{
			std::cout << "Couldn't load the pretrained model" << std::endl;

			//Initializers::WeightsInitializerXavierUniform initializer;
			Initializers::WeightsInitializerGlorotUniform initializer;
			//Initializers::WeightsInitializerHeNormal initializer;
			neuralNetwork.Initialize(initializer);
		}
		else
		{
			alpha *= 0.01;
			neuralNetwork.setParams({ alpha, lim, beta1, beta2 });
			hasPretrained = true;
		}
	}
	else
		// load some saved model
		if (!neuralNetwork.loadNetwork("../../data/neural" + std::to_string(startEpoch - 1) + ".net"))
		{
			std::cout << "Couldn't load the last model" << std::endl;
			return false;
		}


	// train the model

	const int batchSize = 32;

	Eigen::MatrixXd in(nrInputs, batchSize);
	Eigen::MatrixXd out(nrOutputs, batchSize);

	std::default_random_engine rde(42);
	std::uniform_int_distribution<> distIntBig(0, static_cast<int>(trainInputs.cols() - 1));

	// use dropout for input level instead!
	//#define ADD_NOISE 1
#ifdef ADD_NOISE
	const double dropProb = 0.2; // also a hyperparameter
	std::bernoulli_distribution dist(dropProb);
#endif

	std::cout << "Training samples: " << trainInputs.cols() << std::endl;
	const long long int nrBatches = trainInputs.cols() / batchSize;
	std::cout << "Traing batches / epoch: " << nrBatches << std::endl;

	std::cout << "Validation samples: " << validationInputs.cols() << std::endl;
	std::cout << "Test samples: " << testInputs.cols() << std::endl;

	const int nrEpochs = hasPretrained ? 4 : 20; // bigger dropout, more epochs - less if starting from a pretrained model

	if (nrEpochs > 0)
	{
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::vector<double> trainLosses(nrEpochs);
		std::vector<double> validationLosses(nrEpochs);

		std::vector<double> indices(nrEpochs);

		long long int bcnt = 0;
		for (int epoch = startEpoch; epoch < startEpoch + nrEpochs; ++epoch)
		{
			std::cout << "Epoch: " << epoch << " Alpha: " << alpha << std::endl;

			double totalLoss = 0;
			for (int batch = 0; batch < nrBatches; ++batch)
			{
				for (int b = 0; b < batchSize; ++b)
				{
					const int ind = distIntBig(rde);

					in.col(b) = trainInputs.col(ind);

#ifdef ADD_NOISE
					for (int i = 0; i < nrInputs; ++i)
					{
						if (distDrop(rde))
							in(i, b) = 0;
					}
#endif

					out.col(b) = trainOutputs.col(ind);
				}

				neuralNetwork.ForwardBackwardStep(in, out);

				const double loss = neuralNetwork.getLoss() / batchSize;
				totalLoss += loss;

				if (bcnt % 100 == 0)
					std::cout << "Loss: " << loss << std::endl;

				++bcnt;
			}

			std::cout << "Average loss: " << totalLoss / static_cast<double>(nrBatches) << std::endl;


			// stats / epoch

			long long int validCorrect = 0;
			long long int trainCorrect = 0;

			for (int i = 0; i < validationRecords.size(); ++i)
			{
				Eigen::VectorXd res = neuralNetwork.Predict(validationInputs.col(i));
				validationRes.col(i) = res;

				double limp = 0;
				for (int j = 0; j < nrOutputs; ++j)
					limp = std::max(limp, res(j));

				int nr = -1;
				for (int j = 0; j < nrOutputs; ++j)
					if (validationOutputs(j, i) > 0.5)
					{
						if (nr != -1)
							std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
						nr = j;
					}

				int predn = -1;
				for (int n = 0; n < nrOutputs; ++n)
					if (res(n) >= limp)
					{
						if (predn != -1)
							std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
						predn = n;
					}

				if (predn == nr)
					++validCorrect;


				const int ind = distIntBig(rde);

				res = neuralNetwork.Predict(trainInputs.col(ind));
				trainStatsRes.col(i) = res;
				trainStatsOutputs.col(i) = trainOutputs.col(ind);

				limp = 0;

				for (int j = 0; j < nrOutputs; ++j)
					limp = std::max(limp, res(j));

				nr = -1;
				for (int j = 0; j < nrOutputs; ++j)
					if (trainStatsOutputs(j, i) > 0.5)
					{
						if (nr != -1)
							std::cout << "Info from label ambiguous, should not happen: " << nr << " and " << j << std::endl;
						nr = j;
					}

				predn = -1;
				for (int n = 0; n < nrOutputs; ++n)
					if (res(n) >= limp)
					{
						if (predn != -1)
							std::cout << "Ambiguous prediction: " << predn << " and " << n << std::endl;
						predn = n;
					}

				if (predn == nr)
					++trainCorrect;
			}


			const int nrEpoch = epoch - startEpoch;
			trainLosses[nrEpoch] = neuralNetwork.getLoss(trainStatsRes, trainStatsOutputs) / static_cast<double>(validationRecords.size());
			validationLosses[nrEpoch] = neuralNetwork.getLoss(validationRes, validationOutputs) / static_cast<double>(validationRecords.size());
			indices[nrEpoch] = epoch;

			std::cout << "Training loss: " << trainLosses[nrEpoch] << std::endl;
			std::cout << "Validation loss: " << validationLosses[nrEpoch] << std::endl;

			std::cout << "Training accuracy: " << 100. * static_cast<double>(trainCorrect) / static_cast<double>(validationRecords.size()) << "%" << std::endl;
			std::cout << "Validation accuracy: " << 100. * static_cast<double>(validCorrect) / static_cast<double>(validationRecords.size()) << "%" << std::endl << std::endl;

			const std::string fileName = "../../data/neural" + std::to_string(epoch) + ".net";
			neuralNetwork.saveNetwork(fileName);

			// makes the learning rate smaller each epoch
			alpha *= decay;
			neuralNetwork.setLearnRate(alpha);
		}


		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto dif = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

		std::cout << "Training took: " << dif / 1000. << " seconds!" << std::endl;

		{
			Utils::DataFileWriter theFile("../../data/EMNIST.txt");
			theFile.AddDataset(indices, trainLosses);
			theFile.AddDataset(indices, validationLosses);
		}

		Utils::Gnuplot plot;
		plot.setType(Utils::Gnuplot::ChartType::training);
		plot.setCmdFileName("EMNIST.plt");
		plot.setDataFileName("EMNIST.txt");
		plot.Execute();
	}

	// first, on training set:

	std::cout << std::endl << "Training set:" << std::endl;

	Utils::MNISTDatabase::PrintStats(neuralNetwork, trainInputs, trainOutputs, nrOutputs);


	// now, on test set:

	std::cout << std::endl << "Test set:" << std::endl;

	Utils::MNISTDatabase::PrintStats(neuralNetwork, testInputs, testOutputs, nrOutputs);

	return true;
}
