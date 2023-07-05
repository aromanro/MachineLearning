#include "Tests.h"
#include "MNISTDatabase.h"


bool LoadData(std::vector<std::pair<std::vector<double>, uint8_t>>& trainingRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& testRecords, bool augment)
{
	// load the data
	Utils::MNISTDatabase minstTrainDataFiles;
	if (!minstTrainDataFiles.Open()) {
		std::cout << "Couldn't load train data" << std::endl;
		return false;
	}

	trainingRecords = minstTrainDataFiles.ReadAllImagesAndLabels(augment);
	minstTrainDataFiles.Close();

	Utils::MNISTDatabase minstTestDataFiles;
	minstTestDataFiles.setImagesFileName("emnist-digits-test-images-idx3-ubyte");
	minstTestDataFiles.setLabelsFileName("emnist-digits-test-labels-idx1-ubyte");
	if (!minstTestDataFiles.Open()) {
		std::cout << "Couldn't load test data" << std::endl;
		return false;
	}

	testRecords = minstTestDataFiles.ReadAllImagesAndLabels();
	minstTestDataFiles.Close();

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(trainingRecords.begin(), trainingRecords.end(), g);

	return true;
}

bool LoadData(std::vector<std::pair<std::vector<double>, uint8_t>>& trainingRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& validationRecords, std::vector<std::pair<std::vector<double>, uint8_t>>& testRecords, bool augment, double percentage)
{
	if (!LoadData(trainingRecords, testRecords, augment))
		return false;

	// split the training data into training and validation sets

	const int nrTrainingRecords = static_cast<int>(trainingRecords.size() * percentage);

	validationRecords = std::vector<std::pair<std::vector<double>, uint8_t>>(trainingRecords.begin() + nrTrainingRecords, trainingRecords.end());
	trainingRecords.resize(nrTrainingRecords);

	return true;
}


void SetDataIntoMatrices(const std::vector<std::pair<std::vector<double>, uint8_t>>& records, Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs)
{
	int rec = 0;
	for (const auto& record : records)
	{
		for (int i = 0; i < inputs.rows(); ++i)
			inputs(i, rec) = record.first[i];

		for (int i = 0; i < outputs.rows(); ++i)
			outputs(i, rec) = (i == record.second) ? 1 : 0;

		++rec;
	}
}
