#pragma once

#include "NeuralSublayer.h"

namespace NeuralNetworks
{

	// to keep things simple, at least for a while it will simply use a single 'neural sublayer'

	template<class Solver = SGD::AdamSolver<>>
	class NeuralLayer
	{
	public:
		NeuralLayer(int szi = 1, int szo = 1)
			: layer(szi, szo)
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

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			layer.Initialize(initializer);
		}

		Eigen::VectorXd Predict(const Eigen::VectorXd& input)
		{
			return layer.Predict(input);
		}

		double getLoss() const
		{
			return layer.getLoss();
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

		Eigen::MatrixXd AddBatch(const Eigen::MatrixXd& batchInput, const Eigen::MatrixXd& batchOutput)
		{
			return layer.AddBatch(batchInput, batchOutput);
		}

		Eigen::MatrixXd getPrediction() const
		{
			return layer.getPrediction();
		}

		Eigen::MatrixXd getInput() const
		{
			return layer.getInput();
		}

		Eigen::MatrixXd BackpropagateBatch(const Eigen::MatrixXd& grad) const
		{
			return layer.BackpropagateBatch(grad);
		}

	private:
		NeuralSublayer<Solver> layer;
	};

}


