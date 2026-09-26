#pragma once

#include <algorithm>
#include <iomanip>
#include <limits>
#include <vector>
#include <stdexcept>


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
	using HiddenLayerDefault = SGD::AdamWSolver<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, ActivationFunctions::LeakyRELUFunction<>/*ActivationFunctions::SELUFunction<>*/>;

	// for the last layer, by default a single neuron with sigmoid function, to be used for classification
	// this should be changed more often, that's why it's the first template argument

	template<class LastSolver = SGD::LogisticRegressionAdamWSolver, class Solver = HiddenLayerDefault>
	class MultilayerPerceptron
	{
	public:
		// neurons contains the number of neurons in each layer including the input layer 
		// which is not explicitly represented in the implementation, but the number is used for the number of inputs
		explicit MultilayerPerceptron(const std::vector<int>& neurons, const std::vector<double>& drop = {}, const std::vector<double> snoise = {},
			GLM::OptimizerInitialization initialization = GLM::OptimizerInitialization::Eager)
			: lastLayer(1, 1, initialization), optimizerInitialization(initialization),
			configuredDropout(drop), configuredNoise(snoise), batchNormParam(1.), noBatchNormalizationOnInput(true)
		{
			const std::vector<int> sizes = neurons.size() < 2
				? std::vector<int>{neurons.empty() ? 1 : neurons.front(), 1} : neurons;
			if (std::any_of(sizes.begin(), sizes.end(), [](int size) { return size <= 0; }))
				throw std::invalid_argument("Layer dimensions must be positive");

			batchNormMeans.reserve(sizes.size());
			batchNormVariances.reserve(sizes.size());
			hiddenLayers.reserve(sizes.size() - 2);

			int inputs = sizes[0];
			for (int i = 1; i < sizes.size() - 1; ++i)
			{
				const int outputs = sizes[i];
				hiddenLayers.emplace_back(inputs, outputs, initialization);
				const int hidInd = i - 1;
				hiddenLayers[hidInd].setLastLayer(false);
				hiddenLayers[hidInd].setFirstLayer(false);

				batchNormMeans.push_back(Eigen::VectorXd::Zero(inputs));
				batchNormVariances.push_back(Eigen::VectorXd::Ones(inputs));

				inputs = outputs;
			}
			if (!hiddenLayers.empty()) hiddenLayers.front().setFirstLayer();

			lastLayer = NeuralLayerPerceptron<LastSolver>(inputs, sizes.back(), initialization);
			lastLayer.setLastLayer();
			lastLayer.setFirstLayer(hiddenLayers.empty());

			batchNormMeans.push_back(Eigen::VectorXd::Zero(inputs));
			batchNormVariances.push_back(Eigen::VectorXd::Ones(inputs));

			std::random_device rd;
			rde.seed(rd());

			ConfigureRegularization();
		}

		// this assumes that the params for the last and hidden layers are the same, so use the same solver (but the activation and cost functions can be different)
		void setParams(const std::vector<double>& params)
		{
			setParamsLastLayer(params);
			setParamsHiddenLayers(params);
		}

		void setLearnRate(double a)
		{
			if (hiddenLayerParams.empty()) hiddenLayerParams.push_back(a);
			else hiddenLayerParams[0] = a;
			lastLayer.setLearnRate(a);
			for (int i = 0; i < hiddenLayers.size(); ++i)
				hiddenLayers[i].setLearnRate(a);
		}

		// Reproducible dropout/noise; weight initialization has its own generator.
		void setRandomSeed(unsigned int seed)
		{
			rde.seed(seed);
		}

		void setParamsLastLayer(const std::vector<double>& params)
		{
			lastLayer.setParams(params);
		}

		void setParamsHiddenLayers(const std::vector<double>& params)
		{
			// setParams accepts a prefix: retain the rest for layers added by loading.
			if (hiddenLayerParams.size() < params.size()) hiddenLayerParams.resize(params.size());
			std::copy(params.begin(), params.end(), hiddenLayerParams.begin());
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
			InitializeHiddenLayers(initializer);
			for (auto& mean : batchNormMeans) mean.setZero();
			for (auto& variance : batchNormVariances) variance.setOnes();
		}

		void InitializeLastLayer(Initializers::WeightsInitializerInterface& initializer)
		{
			lastLayer.Initialize(initializer);
		}

		void InitializeHiddenLayers(Initializers::WeightsInitializerInterface& initializer)
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
					v = (v - batchNormMeans[i]).cwiseProduct(PredictionInvStd(i));
				
				v = hiddenLayers[i].Predict(v);
			}

			if (batchNormParam != 1. && !(hiddenLayers.empty() && noBatchNormalizationOnInput))
				v = (v - batchNormMeans.back()).cwiseProduct(PredictionInvStd(static_cast<int>(hiddenLayers.size())));

			return lastLayer.Predict(v);
		}

		// Each column is an independent example. This path uses the saved
		// inference normalization statistics and never records training caches.
		Eigen::MatrixXd PredictBatch(const Eigen::MatrixXd& input)
		{
			Eigen::MatrixXd value = input;
			for (int i = 0; i < hiddenLayers.size(); ++i)
			{
				NormalizePredictionBatch(i, value);
				value = hiddenLayers[i].PredictBatch(value);
			}
			NormalizePredictionBatch(static_cast<int>(hiddenLayers.size()), value);
			return lastLayer.PredictBatch(value);
		}

		// True if at least one layer has initialized its optimizer.
		bool isOptimizerInitialized() const
		{
			if (lastLayer.isOptimizerInitialized()) return true;
			for (const auto& layer : hiddenLayers)
				if (layer.isOptimizerInitialized()) return true;
			return false;
		}

		void ClearBatchCache()
		{
			lastLayer.ClearBatchCache();
			for (auto& layer : hiddenLayers) layer.ClearBatchCache();
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
			std::vector<Eigen::VectorXd> dropoutMasks(dropout.size());

			std::vector<Eigen::VectorXd> avgi;
			std::vector<Eigen::VectorXd> istdi;

			if (batchNormParam != 1.)
			{
				avgi.reserve(batchNormMeans.size());
				istdi.reserve(batchNormVariances.size());
			}

			const bool transforms = batchNormParam != 1.
				|| std::any_of(dropout.begin(), dropout.end(), [](double p) { return p > 0.; })
				|| std::any_of(stdsNoise.begin(), stdsNoise.end(), [](double s) { return s > 0.; });
			if (!transforms)
			{
				// Pass borrowed outputs directly to the next layer's owning cache.
				// No temporary input/prediction copies or dummy targets are needed.
				const Eigen::MatrixXd* current = &input;
				for (auto& layer : hiddenLayers)
				{
					layer.AddBatchNoParamsAdjustment(*current);
					current = &layer.getPrediction();
				}
				lastLayer.AddBatchNoParamsAdjustment(*current, target);
			}
			else
			{
				Eigen::MatrixXd inp = input;

				// dropout for input
				Dropout(0, inp, dropoutMasks);

				// Experimental code that allows adding gaussian noise to the outputs of the hidden layers and the input
				// noise for input
				if (stdsNoise[0] > 0.)
				{
					std::normal_distribution<double> dist(0., stdsNoise[0]);

					for (int j = 0; j < inp.rows(); ++j)
						for (int k = 0; k < inp.cols(); ++k)
							inp(j, k) += dist(rde);
				}
				// end of experimental code

				for (int i = 0; i < hiddenLayers.size(); ++i)
				{
					BatchNormalize(avgi, istdi, i, inp);

					hiddenLayers[i].AddBatchNoParamsAdjustment(inp);
					inp = hiddenLayers[i].getPrediction();

					Dropout(i + 1, inp, dropoutMasks);

					// Experimental code that allows adding gaussian noise to the outputs of the hidden layers and the input
					// noise for hidden layers
					if (stdsNoise[i + 1] > 0.)
					{
						std::normal_distribution<double> dist(0., stdsNoise[i + 1]);

						for (int j = 0; j < inp.rows(); ++j)
							for (int k = 0; k < inp.cols(); ++k)
								inp(j, k) += dist(rde);
					}
					// end of experimental code
				}

				BatchNormalize(avgi, istdi, static_cast<int>(hiddenLayers.size()), inp);
				lastLayer.AddBatchNoParamsAdjustment(inp, target);
			}

			// Finish forward without updating any weights. Every input derivative
			// below must use the same weights that produced these predictions.
			const Eigen::MatrixXd lastGradient = lastLayer.ComputeBatchGradient(target);
			Eigen::MatrixXd grad = lastLayer.BackpropagateBatch(lastGradient);
			std::vector<Eigen::MatrixXd> hiddenGradients(hiddenLayers.size());

			for (int i = static_cast<int>(hiddenLayers.size()) - 1; i >= 0; --i)
			{
				const int ip1 = i + 1;

				BatchNormalizeGradient(istdi, ip1, grad);

				// zero out the gradient for the dropped out neurons
				DropoutGradient(ip1, grad, dropoutMasks);

				hiddenGradients[i] = hiddenLayers[i].ComputeBatchGradient(grad);
				if (i > 0 || backpropagateToInput)
					grad = hiddenLayers[i].BackpropagateBatch(hiddenGradients[i]);
			}

			if (backpropagateToInput)
			{
				BatchNormalizeGradient(istdi, 0, grad);
				DropoutGradient(0, grad, dropoutMasks);
			}

			// All backward propagation is complete before the first update.
			lastLayer.ApplyBatchUpdate(lastGradient);
			for (size_t i = 0; i < hiddenLayers.size(); ++i)
				hiddenLayers[i].ApplyBatchUpdate(hiddenGradients[i]);

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
				if (!hiddenLayers[i].saveLayer(os)) return false;

			if (!lastLayer.saveLayer(os)) return false;

			// Keep the published inverse-std layout so older readers preserve inference.
			// Running statistics in memory remain variances.
			const auto precision = os.precision();
			os << std::setprecision(std::numeric_limits<double>::max_digits10) << batchNormParam << std::endl;
			os.precision(precision);

			const static Eigen::IOFormat csv(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
			for (int i = 0; i < batchNormMeans.size(); ++i)
			{
				os << batchNormMeans[i].format(csv) << std::endl;
				const Eigen::VectorXd invStd = (batchNormVariances[i].array() + batchNormEpsilon).sqrt().inverse();
				os << invStd.format(csv) << std::endl;
			}

			return static_cast<bool>(os);
		}

		bool loadNetwork(const std::string& name)
		{
			std::ifstream is(name, std::ios::in);
			if (!is.is_open())
			{
				std::cerr << "Couldn't load network '" << name << "': cannot open file\n";
				return false;
			}

			return loadNetwork(is);
		}

		bool loadNetwork(std::ifstream& is)
		{
			try
			{
				// Parse into a candidate: failed loads preserve parameters, optimizer
				// state, normalization statistics, configuration, and the random seed.
				auto candidate = *this;
				std::string error;
				if (!candidate.LoadNetworkParameters(is, error))
				{
					std::cerr << "Couldn't load network: " << error << '\n';
					return false;
				}
				if (candidate.hiddenLayers.size() > hiddenLayers.size()
					&& (configuredDropout.size() < candidate.dropout.size() || configuredNoise.size() < candidate.stdsNoise.size()))
					std::cerr << "Loaded network: unspecified dropout/noise entries for added stages use 0; "
						"pass complete constructor schedules to configure them.\n";
				*this = std::move(candidate);
				return true;
			}
			catch (const std::exception& exception)
			{
				std::cerr << "Couldn't load network: " << exception.what() << '\n';
				return false;
			}
		}

	private:
		// Entries refer to input (0), then each hidden layer's output (1, 2, ...).
		// Keep the full supplied schedules so a smaller placeholder or a temporary
		// topology reduction does not discard settings for a later load.
		void ConfigureRegularization()
		{
			dropout = configuredDropout;
			stdsNoise = configuredNoise;
			dropout.resize(hiddenLayers.size() + 1, 0.);
			stdsNoise.resize(hiddenLayers.size() + 1, 0.);
		}

		bool LoadNetworkParameters(std::ifstream& is, std::string& error)
		{
			const auto fail = [&](const std::string& reason) { error = reason; return false; };
			if (!is.is_open()) return fail("input file is not open");

			int nrLayers;
			if (!(is >> nrLayers) || nrLayers < 0) return fail("invalid hidden-layer count");

			std::cout << "Loading network with " << static_cast<size_t>(nrLayers) + 1 << " layers" << std::endl;
			
			if (nrLayers < hiddenLayers.size()) hiddenLayers.resize(nrLayers);

			// load all layers
			for (int i = 0; i < nrLayers; ++i)
			{
				if (i == hiddenLayers.size())
				{
					hiddenLayers.emplace_back(1, 1, optimizerInitialization);
					hiddenLayers.back().setParams(hiddenLayerParams);
				}
				std::string layerError;
				if (!hiddenLayers[i].loadLayer(is, &layerError))
					return fail("hidden layer " + std::to_string(i) + ": " + layerError);
				if (i > 0 && hiddenLayers[i].getNrInputs() != hiddenLayers[i - 1].getNrOutputs())
					return fail("hidden layer " + std::to_string(i) + ": input count does not match the previous layer");
			}

			std::string layerError;
			if (!lastLayer.loadLayer(is, &layerError)) return fail("output layer: " + layerError);
			if (!hiddenLayers.empty() && lastLayer.getNrInputs() != hiddenLayers.back().getNrOutputs())
				return fail("output layer: input count does not match the previous layer");

			// Size statistics from the loaded layers, including when topology changes.
			batchNormMeans.resize(hiddenLayers.size() + 1);
			batchNormVariances.resize(hiddenLayers.size() + 1);
			ConfigureRegularization();
			for (int i = 0; i < batchNormMeans.size(); ++i)
			{
				const int inputs = i < hiddenLayers.size() ? hiddenLayers[i].getNrInputs() : lastLayer.getNrInputs();
				batchNormMeans[i] = Eigen::VectorXd::Zero(inputs);
				batchNormVariances[i] = Eigen::VectorXd::Ones(inputs);
				if (i < hiddenLayers.size())
				{
					hiddenLayers[i].setLastLayer(false);
					hiddenLayers[i].setFirstLayer(i == 0);
				}
			}
			lastLayer.setLastLayer();
			lastLayer.setFirstLayer(hiddenLayers.empty());

			std::string format;
			batchNormParam = 1.;
			if (!(is >> format)) // Older files may omit normalization.
				return (is.eof() && !is.bad()) || fail("cannot read normalization header");
			const bool varianceFormat = format == "BatchNormVarianceV1";
			if (varianceFormat)
			{
				if (!(is >> batchNormParam)) return fail("missing normalization coefficient");
			}
			else
			{
				try
				{
					size_t parsed = 0;
					batchNormParam = std::stod(format, &parsed);
					if (parsed != format.size()) return fail("invalid normalization header");
				}
				catch (const std::exception&) { return fail("invalid normalization header"); }
			}
			if (!std::isfinite(batchNormParam) || batchNormParam < 0. || batchNormParam > 1.)
				return fail("normalization coefficient must be finite and between 0 and 1");
			if (batchNormParam == 0.) batchNormParam = 1.;
			// Legacy files may stop at the disabled-normalization flag. If stats
			// are present, validate and retain them even while normalization is off.
			is >> std::ws;
			if (batchNormParam == 1. && !varianceFormat && is.eof()) return !is.bad() || fail("cannot read normalization statistics");

			for (int i = 0; i < batchNormMeans.size(); ++i)
			{
				for (int row = 0; row < batchNormMeans[i].size(); ++row)
				{
					if (!(is >> batchNormMeans[i](row)) || !std::isfinite(batchNormMeans[i](row)))
						return fail("invalid or missing normalization mean for stage " + std::to_string(i));
				}
				for (int row = 0; row < batchNormVariances[i].size(); ++row)
				{
					double value;
					if (!(is >> value) || !std::isfinite(value) || value < 0.)
						return fail("invalid or missing normalization scale for stage " + std::to_string(i));
					if (!varianceFormat)
					{
						if (value == 0.) return fail("zero inverse standard deviation for stage " + std::to_string(i));
						// Recover the effective variance; historical batch variances are unavailable.
						const double stddev = 1. / value;
						value = std::max(0., stddev * stddev - batchNormEpsilon);
						if (!std::isfinite(value)) return fail("normalization variance overflow for stage " + std::to_string(i));
					}
					batchNormVariances[i](row) = value;
				}
			}

			is >> std::ws;
			return (is.eof() && !is.bad()) || fail("unexpected trailing data or stream error");
		}

		static constexpr double batchNormEpsilon = 1E-10;

		Eigen::VectorXd PredictionInvStd(int index) const
		{
			// Noise is a training transform. Preserve the learned normalization
			// scale at inference, including the noise contribution to its variance.
			return (batchNormVariances[index].array() + batchNormEpsilon).sqrt().inverse().matrix();
		}

		void NormalizePredictionBatch(int index, Eigen::MatrixXd& values) const
		{
			if (batchNormParam == 1. || (index == 0 && noBatchNormalizationOnInput)) return;
			const Eigen::VectorXd invStd = PredictionInvStd(index);
			values.colwise() -= batchNormMeans[index];
			values.array().colwise() *= invStd.array();
		}

		void BatchNormalize(std::vector<Eigen::VectorXd>& avgi, std::vector<Eigen::VectorXd>& istdi, int i, Eigen::MatrixXd& inp)
		{
			if (batchNormParam != 1.)
			{
				if (i != 0 || !noBatchNormalizationOnInput)
				{
					const double oneMinusBatchNormParam = 1. - batchNormParam;

					avgi.emplace_back(inp.rowwise().mean());
					inp.colwise() -= avgi.back();
					// Center first to avoid cancellation in E[x^2] - E[x]^2.
					const Eigen::VectorXd variance = inp.array().square().rowwise().mean();
					istdi.emplace_back((variance.array() + batchNormEpsilon).sqrt().inverse().matrix());
					inp = inp.array().colwise() * istdi.back().array();

					batchNormMeans[i] = batchNormParam * batchNormMeans[i] + oneMinusBatchNormParam * avgi.back();
					batchNormVariances[i] = batchNormParam * batchNormVariances[i] + oneMinusBatchNormParam * variance;
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
			if (batchNormParam == 1. || (index == 0 && noBatchNormalizationOnInput)) return;

			// The layer input is the normalized activation saved during forward.
			// Differentiate the batch mean and variance as well as the scaling.
			const Eigen::MatrixXd& normalized = index < hiddenLayers.size()
				? hiddenLayers[index].getInput() : lastLayer.getInput();
			const Eigen::VectorXd meanGradient = grad.rowwise().mean();
			const Eigen::VectorXd projected = (grad.array() * normalized.array()).rowwise().mean();
			grad.colwise() -= meanGradient;
			grad.array() -= normalized.array().colwise() * projected.array();
			grad.array().colwise() *= istdi[index].array();
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

				// Keep the activation cache unchanged. Backward uses the saved
				// preactivation and the explicit mask to differentiate dropout.
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
		GLM::OptimizerInitialization optimizerInitialization;
		std::vector<NeuralLayerPerceptron<Solver>> hiddenLayers;
		std::vector<double> hiddenLayerParams;
		std::vector<double> configuredDropout;
		std::vector<double> configuredNoise;

		std::vector<double> dropout;
		std::vector<double> stdsNoise;

		std::mt19937 rde;
		std::uniform_real_distribution<> distDrop{0., 1.};

		double batchNormParam;
		bool noBatchNormalizationOnInput;
		std::vector<Eigen::VectorXd> batchNormMeans;
		std::vector<Eigen::VectorXd> batchNormVariances;
	};

}

