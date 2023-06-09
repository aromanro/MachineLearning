#pragma once

#include <vector>
#include <Eigen/Eigen>

template<class Model, typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd> class Ensemble
{
public:
	void addModel(Model* model, double weight)
	{
		models.push_back(model);
		weights.push_back(weight);
	}

	OutputType Predict(const InputType& x)
	{
		OutputType prediction;
		if (models.size() == 0)
			return prediction;
		
		prediction = weights[0] * models[0]->Predict(x);
		for (int i = 1; i < models.size(); ++i)
			prediction += weights[i] * models[i]->Predict(x);

		return prediction;
	}

private:
	std::vector<double> weights;
	std::vector<Model*> models;
};

