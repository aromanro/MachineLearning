#pragma once

#include <vector>


#include "GradientSolvers.h"
#include "LogisticRegression.h"
#include "NeuralLayer.h"
#include "Normalizer.h"

namespace NeuralNetworks
{

	// the LastSolver is here to be able to provide a different method for the last layer
	// not necessarily some other stochastic gradient descent, but through the solver the activation and cost functions can be specified 
	// and that one can be different for the last layer

	// for the hidden layer, by default use an adam solver with a leaky RELU activation (the cost does not matter, it's computed for the output of the last layer only)
	using HiddenLayerDefault = SGD::AdamWSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>>;

	// for the last layer, by default a single neuron with sigmoid function, to be used for classification
	// this should be changed more often, that's why it's the first template argument

	template<class LastSolver = SGD::LogisticRegressionAdamWSolver, class Solver = HiddenLayerDefault>
	class MultilayerPerceptron
	{
	public:
		// neurons contains the number of neurons in each layer including the input layer 
		// which is not explicitly represented in the implementation, but the number is used for the number of inputs
		explicit MultilayerPerceptron(const std::vector<int>& neurons, const std::vector<double>& drop = {})
			: batchNormParam(1.), noBatchNormalizationOnInput(true)
		{
			if (neurons.empty()) return;
			else if (neurons.size() == 1)
			{
				lastLayer = NeuralLayerPerceptron<LastSolver>(neurons[0]);
				lastLayer.setLastLayer();
				batchNormMeans.push_back(Eigen::VectorXd::Zero(neurons[0]));
				batchNormInvStds.push_back(Eigen::VectorXd::Ones(neurons[0]));
				return;
			}

			batchNormMeans.reserve(neurons.size());
			batchNormInvStds.reserve(neurons.size());

			int inputs = neurons[0];
			for (int i = 1; i < neurons.size() - 1; ++i)
			{
				const int outputs = neurons[i];
				hiddenLayers.emplace_back(NeuralLayerPerceptron<Solver>(inputs, outputs));
				const int hidInd = i - 1;
				hiddenLayers[hidInd].setLastLayer(false);
				hiddenLayers[hidInd].setFirstLayer(false);

				batchNormMeans.push_back(Eigen::VectorXd::Zero(inputs));
				batchNormInvStds.push_back(Eigen::VectorXd::Ones(inputs));

				inputs = outputs;
			}
			if (!hiddenLayers.empty()) hiddenLayers.front().setFirstLayer();

			lastLayer = NeuralLayerPerceptron<LastSolver>(inputs, neurons.back());
			lastLayer.setLastLayer();
			lastLayer.setFirstLayer(false);

			batchNormMeans.push_back(Eigen::VectorXd::Zero(inputs));
			batchNormInvStds.push_back(Eigen::VectorXd::Ones(inputs));

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

		void setBatchNormalizationParam(double val)
		{
			if (val == 0.) val = 1.;
			batchNormParam = val;
		}

		double getBatchNormalizationParam() const
		{
			return batchNormParam;
		}

		void setNoBatchNormalizationOnInput(bool val = true)
		{
			noBatchNormalizationOnInput = val;
		}

		bool getNoBatchNormalizationOnInput() const
		{
			return noBatchNormalizationOnInput;
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
			{
				if (batchNormParam != 1. && !(i == 0 && noBatchNormalizationOnInput))
					v = (v - batchNormMeans[i]).cwiseProduct(batchNormInvStds[i]);
				
				v = hiddenLayers[i].Predict(v);
			}

			if (batchNormParam != 1. && !(hiddenLayers.empty() && noBatchNormalizationOnInput))
				v = (v - batchNormMeans.back()).cwiseProduct(batchNormInvStds.back());

			return lastLayer.Predict(v);
		}

		double getLoss() const
		{
			return lastLayer.getLoss();
		}

		double getLoss(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& target) const
		{
			return lastLayer.getLoss(prediction, target);
		}

		Eigen::MatrixXd ForwardBackwardStep(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, bool backpropagateToInput = false)
		{
			const int batchSize = static_cast<int>(input.cols());
			std::vector<Eigen::VectorXd> dropoutMasks(dropout.size());

			std::vector<Eigen::VectorXd> avgi;
			std::vector<Eigen::VectorXd> istdi;

			if (batchNormParam != 1.)
			{
				avgi.reserve(batchNormMeans.size());
				istdi.reserve(batchNormInvStds.size());
			}

			// forward
			Eigen::MatrixXd inp = input;

			// dropout for input
			Dropout(0, inp, dropoutMasks);
			
			for (int i = 0; i < hiddenLayers.size(); ++i)
			{
				BatchNormalize(avgi, istdi, i, inp);
				
				t.resize(hiddenLayers[i].getNrOutputs(), batchSize);
				hiddenLayers[i].AddBatchNoParamsAdjustment(inp, t);
				inp = hiddenLayers[i].getPrediction();

				Dropout(i + 1, inp, dropoutMasks);
			}

			BatchNormalize(avgi, istdi, static_cast<int>(hiddenLayers.size()), inp);

			// forward and backward for the last layer and backpropagate the gradient to the last hidden layer
			Eigen::MatrixXd grad = lastLayer.BackpropagateBatch(lastLayer.AddBatchWithParamsAdjusment(inp, target));

			// backward: now backpropagate the gradient through the hidden layers:

			for (int i = static_cast<int>(hiddenLayers.size() - 1); i > 0; --i)
			{
				const int ip1 = i + 1;

				BatchNormalizeGradient(istdi, ip1, grad);

				// zero out the gradient for the dropped out neurons
				DropoutGradient(ip1, grad, dropoutMasks);

				// do the adjustments of the parameters as well and backpropagate for each hidden layer
				grad = hiddenLayers[i].BackpropagateBatch(hiddenLayers[i].AddBatchWithParamsAdjusment(hiddenLayers[i].getInput(), grad));
			}

			// the first layer does not need to backpropagate gradient to the input layer, that one cannot be adjusted
			// TODO: this could be part of a larger network, even as a single 'layer', before it there could be more layers, for example a convolutional network, in such a case the gradient needs to be backpropagated
			if (!hiddenLayers.empty())
			{
				BatchNormalizeGradient(istdi, 1, grad);

				DropoutGradient(1, grad, dropoutMasks);

				hiddenLayers[0].AddBatchWithParamsAdjusment(hiddenLayers[0].getInput(), grad);

				if (backpropagateToInput) grad = hiddenLayers[0].BackpropagateBatch(grad);
			}

			return grad;
		}

		bool saveNetwork(const std::string& name) const
		{
			std::ofstream os(name, std::ios::out | std::ios::trunc);

			return saveNetwork(os);
		}

		bool saveNetwork(std::ofstream& os) const
		{
			if (!os.is_open()) return false;

			os << hiddenLayers.size() << std::endl;

			// save all layers
			for (int i = 0; i < hiddenLayers.size(); ++i)
				hiddenLayers[i].saveLayer(os);

			lastLayer.saveLayer(os);

			// save batch normalization stuff
			os << batchNormParam << std::endl;

			const static Eigen::IOFormat csv(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
			for (int i = 0; i < batchNormMeans.size(); ++i)
			{
				os << batchNormMeans[i].format(csv) << std::endl;
				os << batchNormInvStds[i].format(csv) << std::endl;
			}

			return true;
		}

		bool loadNetwork(const std::string& name)
		{
			std::ifstream is(name, std::ios::in);

			return loadNetwork(is);
		}

		bool loadNetwork(std::ifstream& is)
		{
			if (!is.is_open()) return false;

			int nrLayers;
			is >> nrLayers;

			std::cout << "Loading network with " << nrLayers + 1 << " layers" << std::endl;
			
			hiddenLayers.resize(nrLayers);

			// load all layers
			for (int i = 0; i < hiddenLayers.size(); ++i)
				if (!hiddenLayers[i].loadLayer(is)) {
					std::cout << "Couldn't load layer nr: " << i << std::endl;
					return false;
				}

			if (!lastLayer.loadLayer(is)) return false;

			// load batch normalization stuff
			try
			{
				batchNormParam = 1.;
				is >> batchNormParam;
				is.ignore();
			}
			catch (...)
			{
				batchNormParam = 1.;
			}

			if (batchNormParam == 0)
				batchNormParam = 1.;

			if (batchNormParam == 1.)
			{
				for (int i = 0; i < batchNormMeans.size(); ++i)
				{
					batchNormMeans[i] = Eigen::VectorXd::Zero(batchNormMeans[i].size());
					batchNormInvStds[i] = Eigen::VectorXd::Ones(batchNormInvStds[i].size());
				}

				return true;
			}

			for (int i = 0; i < batchNormMeans.size(); ++i)
			{
				int row = 0;
				std::string field;
				while (getline(is, field))
				{
					batchNormMeans[i](row) = stod(field);

					++row;
					if (row == batchNormMeans[i].size())
						break;
				}

				row = 0;
				while (getline(is, field))
				{
					batchNormInvStds[i](row) = stod(field);

					++row;
					if (row == batchNormInvStds[i].size())
						break;
				}
			}

			return true;
		}

	private:
		void BatchNormalize(std::vector<Eigen::VectorXd>& avgi, std::vector<Eigen::VectorXd>& istdi, int i, Eigen::MatrixXd& inp)
		{
			if (batchNormParam != 1.)
			{
				if (i != 0 || !noBatchNormalizationOnInput)
				{
					const double oneMinusBatchNormParam = 1. - batchNormParam;

					Norm::Normalizer normalizer(static_cast<int>(inp.rows()));
					normalizer.AddBatch(inp);

					avgi.emplace_back(normalizer.getAverage());
					const Eigen::VectorXd eps = Eigen::VectorXd::Constant(avgi.back().size(), 1E-10);
					istdi.emplace_back((normalizer.getVariance() + eps).cwiseSqrt().cwiseInverse());

					inp = inp.colwise() - avgi.back();
					inp = inp.array().colwise() * istdi.back().array();

					batchNormMeans[i] = batchNormParam * batchNormMeans[i] + oneMinusBatchNormParam * avgi.back();
					batchNormInvStds[i] = batchNormParam * batchNormInvStds[i] + oneMinusBatchNormParam * istdi.back();
				}
				else
				{
					avgi.emplace_back(Eigen::VectorXd::Zero(inp.rows()));
					istdi.emplace_back(Eigen::VectorXd::Ones(inp.rows()));
				}
			}
		}

		void BatchNormalizeGradient(const std::vector<Eigen::VectorXd>& istdi, int index, Eigen::MatrixXd& grad)
		{
			if (batchNormParam != 1.)
				grad = grad.array().colwise() * istdi[index].array();
		}

		void Dropout(int index, Eigen::MatrixXd& inp, std::vector<Eigen::VectorXd>& dropoutMasks)
		{
			if (dropout.size() > index && dropout[index] > 0.)
			{
				const Eigen::RowVectorXd zeroRow = Eigen::RowVectorXd::Zero(inp.cols());
				dropoutMasks[index] = Eigen::VectorXd::Ones(inp.rows());
				for (int j = 0; j < inp.rows(); ++j)
					if (distDrop(rde) < dropout[index])
					{
						inp.row(j) = zeroRow;
						dropoutMasks[index](j) = 0.;
					}

				inp /= (1. - dropout[index]);

				// change the prediction, too, for backpropagation 
				if (index > 0) hiddenLayers[index - 1].setPrediction(inp);
			}
		}

		void DropoutGradient(int index, Eigen::MatrixXd& grad, const std::vector<Eigen::VectorXd>& dropoutMasks)
		{
			if (dropout.size() > index && dropout[index] > 0.)
			{
				for (int j = 0; j < grad.rows(); ++j)
					grad.row(j) *= dropoutMasks[index](j);
				
				grad /= (1. - dropout[index]);
			}
		}


		NeuralLayerPerceptron<LastSolver> lastLayer;
		std::vector<NeuralLayerPerceptron<Solver>> hiddenLayers;

		std::vector<double> dropout;

		std::mt19937 rde;
		std::uniform_real_distribution<> distDrop{0., 1.};

		Eigen::MatrixXd t; // bogus, used only for its size during forward-backward step

		double batchNormParam;
		bool noBatchNormalizationOnInput;
		std::vector<Eigen::VectorXd> batchNormMeans;
		std::vector<Eigen::VectorXd> batchNormInvStds;
	};

}

