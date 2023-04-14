#pragma once

#include <random>

#include "ActivationFunctions.h"
#include "CostFunctions.h"
#include "WeightsInitializer.h"
#include "GradientSolvers.h"


namespace GLM {

	template<class DerivedClass, typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::AdamSolver<>, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
	class GeneralizedLinearModelBase
	{
	public:
		GeneralizedLinearModelBase(int szi = 1, int szo = 1)
			: inputs(szi), outputs(szo)
		{
		}

		virtual ~GeneralizedLinearModelBase() = default;

		virtual OutputType Predict(const InputType& input)
		{
			return solver.activationFunction(W * input + b);
		}

		virtual BatchOutputType AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			static_cast<DerivedClass*>(this)->AddBatchNoParamsAdjustment(batchInput, batchOutput);

			return solver.getWeightsAndBias(W, b);
		}

		const BatchOutputType& getPrediction() const
		{
			return solver.getPrediction();
		}

		void setPrediction(const BatchOutputType& p)
		{
			solver.setPrediction(p);
		}

		const BatchInputType& getInput() const
		{
			return solver.getInput();
		}

		double getLoss() const
		{
			return solver.getLoss();
		}

		double getLoss(const BatchOutputType& prediction, const BatchOutputType& target) const
		{
			return solver.getLoss(prediction, target);
		}

		int getNrInputs() const
		{
			return inputs;
		}

		int getNrOutputs() const
		{
			return outputs;
		}

		Solver& getSolver()
		{
			return solver;
		}

	protected:
		int inputs;
		int outputs;

		WeightsType W;
		OutputType b;

		Solver solver;
	};



	template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::AdamSolver<>, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
	class GeneralizedLinearModel : public GeneralizedLinearModelBase<GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>, InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
	{
	public:
		using BaseType = GeneralizedLinearModelBase<GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>, InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>;

		GeneralizedLinearModel(int szi = 1, int szo = 1) : BaseType(szi, szo)
		{
			Initialize(szi, szo);
		}

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			for (int j = 0; j < BaseType::W.cols(); ++j)
				for (int i = 0; i < BaseType::W.rows(); ++i)
					BaseType::W(i, j) = initializer.get(BaseType::getNrInputs(), BaseType::getNrOutputs());
		}


		void AddBatchNoParamsAdjustment(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			BaseType::solver.AddBatch(batchInput, batchOutput);

			BatchOutputType pred(batchOutput.rows(), batchOutput.cols());
			BatchOutputType linpred(batchOutput.rows(), batchOutput.cols());

			for (unsigned int i = 0; i < batchInput.cols(); ++i)
			{
				linpred.col(i) = BaseType::W * batchInput.col(i) + BaseType::b;
				pred.col(i) = BaseType::solver.activationFunction(linpred.col(i));
			}

			BaseType::solver.setLinearPrediction(linpred);
			BaseType::solver.setPrediction(pred);
		}

		BatchInputType BackpropagateBatch(const BatchOutputType& grad) const
		{
			InputType firstCol = BackpropagateGradient(grad.col(0));
			BatchInputType res(firstCol.size(), grad.cols());

			res.col(0) = firstCol;
			for (int i = 1; i < grad.cols(); ++i)
				res.col(i) = BackpropagateGradient(grad.col(i));

			return res;
		}

	protected:
		void Initialize(int szi = 1, int szo = 1)
		{
			BaseType::solver.Initialize(szi, szo);

			//W = WeightsType::Random(szo, szi);

			BaseType::W.resize(szo, szi);

			// Eigen has a Random generator (random between -1 and 1 by default), but for now I'll stick with this one:

			std::random_device rd;
			std::mt19937 rde(rd());
			const double x = 1. / sqrt(szi);
			std::uniform_real_distribution<> dist(-x, x);
			for (int i = 0; i < szo; ++i)
				for (int j = 0; j < szi; ++j)
					BaseType::W(i, j) = dist(rde);

			BaseType::b = OutputType::Zero(szo);
		}

		InputType BackpropagateGradient(const OutputType& grad) const
		{
			return BaseType::W.transpose() * grad;
		}
	};

	template<class Solver>
	class GeneralizedLinearModel<double, double, double, Solver, Eigen::RowVectorXd> : public GeneralizedLinearModelBase<GeneralizedLinearModel<double, double, double, Solver, Eigen::RowVectorXd>, double, double, double, Solver, Eigen::RowVectorXd>
	{
	public:
		using BaseType = GeneralizedLinearModelBase<GeneralizedLinearModel<double, double, double, Solver, Eigen::RowVectorXd>, double, double, double, Solver, Eigen::RowVectorXd>;

		GeneralizedLinearModel(int szi = 1, int szo = 1) : BaseType(szi, szo)
		{
			Initialize(szi, szo);
		}

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			BaseType::W = initializer.get();
		}

		void AddBatchNoParamsAdjustment(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
		{
			BaseType::solver.AddBatch(batchInput, batchOutput);

			Eigen::RowVectorXd pred(batchOutput.cols());
			Eigen::RowVectorXd linpred(batchOutput.cols());

			for (unsigned int i = 0; i < batchInput.cols(); ++i)
			{
				linpred(i) = BaseType::W * batchInput(i) + BaseType::b;
				pred(i) = BaseType::solver.activationFunction(linpred(i));
			}

			BaseType::solver.setLinearPrediction(linpred);
			BaseType::solver.setPrediction(pred);
		}

		Eigen::RowVectorXd BackpropagateBatch(const Eigen::RowVectorXd& grad) const
		{
			Eigen::RowVectorXd res(grad.size());

			for (int i = 0; i < grad.size(); ++i)
				res(i) = BackpropagateGradient(grad(i));

			return res;
		}

	protected:
		void Initialize(int szi = 1, int szo = 1)
		{
			BaseType::solver.Initialize(szi, szo);
			BaseType::W = 0;
			BaseType::b = 0;
		}

		double BackpropagateGradient(const double& grad) const
		{
			return BaseType::W * grad;
		}
	};

}
