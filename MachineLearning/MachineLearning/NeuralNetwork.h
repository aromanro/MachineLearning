#pragma once

#include <vector>


#include "GradientSolvers.h"
#include "LogisticRegression.h"
#include "NeuralLayer.h"

namespace NeuralNetworks
{

	// the LastSolver is here to be able to provide a different method for the last layer
	// not necessarily some other stochastic gradient descent, but through the solver the activation and cost functions can be specified 
	// and that one can be different for the last layer

	// for the hidden layer, by default use an adam solver with a leaky RELU activation (the cost does not matter, it's computed for the output of the last layer only)
	using HiddenLayerDefault = SGD::AdamSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>>;

	// for the last layer, by default a single neuron with sigmoid function, to be used for classification
	// this should be changed more often, that's why it's the first template argument

	template<class LastSolver = SGD::LogisticRegressionAdamSolver, class Solver = HiddenLayerDefault>
	class MultilayerPerceptron
	{
	public:
		// neurons contains the number of neurons in each layer including the input layer 
		// which is not explicitly represented in the implementation, but the number is used for the number of inputs
		explicit MultilayerPerceptron(const std::vector<int>& neurons, const std::vector<double>& drop = {})
		{
			if (neurons.empty()) return;
			else if (neurons.size() == 1)
			{
				lastLayer = NeuralLayer<LastSolver>(neurons[0]);
				lastLayer.setLastLayer();
				return;
			}

			int inputs = neurons[0];
			for (int i = 1; i < neurons.size() - 1; ++i)
			{
				const int outputs = neurons[i];
				hiddenLayers.emplace_back(NeuralLayer<Solver>(inputs, outputs));
				const int hidInd = i - 1;
				hiddenLayers[hidInd].setLastLayer(false);
				hiddenLayers[hidInd].setFirstLayer(false);

				inputs = outputs;
			}
			if (!hiddenLayers.empty()) hiddenLayers.front().setFirstLayer();

			lastLayer = NeuralLayer<LastSolver>(inputs, neurons.back());
			lastLayer.setLastLayer();
			lastLayer.setFirstLayer(false);

			std::random_device rd;
			rde.seed(rd());

			dropout.resize(hiddenLayers.size() + 1, 0.);
			for (int i = 0; i < std::min(drop.size(), dropout.size()); ++i)
				dropout[i] = drop[i];
		}

		// this assumes that the params for the last and hidden layers are the same, so use the same solver (but the activation and cost functions can be different)
		void setParams(const std::vector<double>& params)
		{
			setParamsLastLayer(params);
			setParamsHiddenLayers(params);
		}

		void setLearnRate(double a)
		{
			lastLayer.setLearnRate(a);
			for (int i = 0; i < hiddenLayers.size(); ++i)
				hiddenLayers[i].setLearnRate(a);
		}

		void setParamsLastLayer(const std::vector<double>& params)
		{
			lastLayer.setParams(params);
		}

		void setParamsHiddenLayers(const std::vector<double>& params)
		{
			for (int i = 0; i < hiddenLayers.size(); ++i)
				hiddenLayers[i].setParams(params);
		}

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			InitializeLastLayer(initializer);
			InitializHiddenLayers(initializer);
		}

		void InitializeLastLayer(Initializers::WeightsInitializerInterface& initializer)
		{
			lastLayer.Initialize(initializer);
		}

		void InitializHiddenLayers(Initializers::WeightsInitializerInterface& initializer)
		{
			for (int i = 0; i < hiddenLayers.size(); ++i)
				hiddenLayers[i].Initialize(initializer);
		}

		Eigen::VectorXd Predict(const Eigen::VectorXd& input)
		{
			Eigen::VectorXd v = input;

			for (int i = 0; i < hiddenLayers.size(); ++i)
				v = hiddenLayers[i].Predict(v);

			return lastLayer.Predict(v);
		}

		double getLoss() const
		{
			return lastLayer.getLoss();
		}

		void ForwardBackwardStep(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target)
		{
			const int batchSize = static_cast<int>(input.cols());

			// forward
			Eigen::MatrixXd inp = input;

			// dropout for input
			if (!dropout.empty() && dropout[0] > 0.)
			{
				const Eigen::RowVectorXd zeroRow = Eigen::RowVectorXd::Zero(inp.cols());
				for (int i = 0; i < inp.rows(); ++i)
					if (distDrop(rde) < dropout[0]) inp.row(i) = zeroRow;

				inp /= (1. - dropout[0]);
			}

			for (int i = 0; i < hiddenLayers.size(); ++i)
			{
				t.resize(hiddenLayers[i].getNrOutputs(), batchSize);
				hiddenLayers[i].AddBatchNoParamsAdjustment(inp, t);
				inp = hiddenLayers[i].getPrediction();

				if (dropout.size() > i + 1 && dropout[i + 1] > 0.)
				{
					const Eigen::RowVectorXd zeroRow = Eigen::RowVectorXd::Zero(inp.cols());
					for (int j = 0; j < inp.rows(); ++j)
						if (distDrop(rde) < dropout[i + 1]) inp.row(j) = zeroRow;

					inp /= (1. - dropout[i + 1]);

					// change the prediction, too, for backpropagation 
					hiddenLayers[i].setPrediction(inp);
				}
			}

			// forward and backward for the last layer and backpropagate the gradient to the last hidden layer
			Eigen::MatrixXd grad = lastLayer.BackpropagateBatch(lastLayer.AddBatch(inp, target));

			// now backpropagate the gradient trough the hidden layers:

			for (int i = static_cast<int>(hiddenLayers.size() - 1); i > 0; --i)
				// now do the adjustments of the parameters as well and backpropagate for each hidden layer
				grad = hiddenLayers[i].BackpropagateBatch(hiddenLayers[i].AddBatch(hiddenLayers[i].getInput(), grad));

			// the first layer does not need to backpropagate gradient to the output layer, that one cannot be adjusted
			if (!hiddenLayers.empty())
				hiddenLayers[0].AddBatch(hiddenLayers[0].getInput(), grad);
		}

	private:
		NeuralLayer<LastSolver> lastLayer;
		std::vector<NeuralLayer<Solver>> hiddenLayers;

		std::vector<double> dropout;

		std::mt19937 rde;
		std::uniform_real_distribution<> distDrop{0., 1.};

		Eigen::MatrixXd t; // bogus, used only for its size during forward-backward step
	};

}

