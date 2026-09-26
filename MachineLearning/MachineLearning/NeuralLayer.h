#pragma once

#include "NeuralSublayer.h"

namespace NeuralNetworks
{

	// to keep things simple, at least for a while it will simply use a single 'neural sublayer'

	template<class Solver = SGD::AdamWSolver<>>
	class NeuralLayerPerceptron
	{
	public:
		NeuralLayerPerceptron(int szi = 1, int szo = 1, GLM::OptimizerInitialization initialization = GLM::OptimizerInitialization::Eager)
			: layer(szi, szo, initialization)
		{
		}

		void setLastLayer(bool last = true)
		{
			layer.getSolver().lastLayer = last;
		}

		bool getLastLayer() const
		{
			return layer.getSolver().lastLayer;
		}

		void setFirstLayer(bool first = true)
		{
			layer.getSolver().firstLayer = first;
		}

		bool getFirstLayer() const
		{
			return layer.getSolver().firstLayer;
		}

		void setParams(const std::vector<double>& params)
		{
			layer.setParams(params);
		}

		void setLearnRate(double a)
		{
			layer.setLearnRate(a);
		}

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			layer.Initialize(initializer);
		}

		Eigen::VectorXd Predict(const Eigen::VectorXd& input)
		{
			return layer.Predict(input);
		}

		Eigen::MatrixXd PredictBatch(const Eigen::MatrixXd& input) { return layer.PredictBatch(input); }
		bool isOptimizerInitialized() const { return layer.isOptimizerInitialized(); }
		void ClearBatchCache() { layer.ClearBatchCache(); }

		double getLoss() const
		{
			return layer.getLoss();
		}

		double getLoss(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& target) const
		{
			return layer.getLoss(prediction, target);
		}

		int getNrOutputs() const
		{
			return layer.getNrOutputs();
		}

		int getNrInputs() const
		{
			return layer.getNrInputs();
		}

		void AddBatchNoParamsAdjustment(const Eigen::MatrixXd& batchInput, const Eigen::MatrixXd& batchOutput)
		{
			layer.AddBatchNoParamsAdjustment(batchInput, batchOutput);
		}

		void AddBatchNoParamsAdjustment(const Eigen::MatrixXd& batchInput)
		{
			layer.AddBatchNoParamsAdjustment(batchInput);
		}

		Eigen::MatrixXd AddBatchWithParamsAdjusment(const Eigen::MatrixXd& batchInput, const Eigen::MatrixXd& batchOutput)
		{
			return layer.AddBatchWithParamsAdjusment(batchInput, batchOutput);
		}

		Eigen::MatrixXd ComputeBatchGradient(const Eigen::MatrixXd& batchOutput)
		{
			return layer.ComputeBatchGradient(batchOutput);
		}

		void ApplyBatchUpdate()
		{
			layer.ApplyBatchUpdate();
		}

		void ApplyBatchUpdate(const Eigen::MatrixXd& gradient)
		{
			layer.ApplyBatchUpdate(gradient);
		}

		// Borrowed views: valid until the next recording forward, cache mutation,
		// ClearBatchCache, or destruction. Copy explicitly when longer lived.
		const Eigen::MatrixXd& getPrediction() const
		{
			return layer.getPrediction();
		}

		void setPrediction(const Eigen::MatrixXd& p)
		{
			layer.setPrediction(p);
		}

		const Eigen::MatrixXd& getInput() const
		{
			return layer.getInput();
		}

		Eigen::MatrixXd BackpropagateBatch(const Eigen::MatrixXd& grad) const
		{
			return layer.BackpropagateBatch(grad);
		}

		bool saveLayer(std::ofstream& os) const
		{
			return layer.saveModel(os);
		}

		bool loadLayer(std::ifstream& is, std::string* error = nullptr)
		{
			return layer.loadModel(is, error);
		}

	private:
		NeuralSublayer<Solver> layer;
	};

}


