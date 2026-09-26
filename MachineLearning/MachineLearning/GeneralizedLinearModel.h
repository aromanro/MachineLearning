#pragma once

#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <utility>
#include <limits>
#include <iomanip>

#include "ActivationFunctions.h"
#include "CostFunctions.h"
#include "WeightsInitializer.h"
#include "GradientSolvers.h"


namespace GLM {

	// Lazy models allocate optimizer state only when recording a training batch.
	// Predict/PredictBatch do not record a batch or initialize the optimizer.
	enum class OptimizerInitialization { Eager, Lazy };

	namespace Detail {
		// A row must contain exactly the advertised number of finite values.
		template<class Store>
		bool ReadParameterRow(std::istream& input, int columns, Store store)
		{
			std::string line;
			if (!std::getline(input, line)) return false;
			std::istringstream values(line);
			for (int column = 0; column < columns; ++column)
			{
				double value;
				if (!(values >> value) || !std::isfinite(value)) return false;
				store(column, value);
				if (column + 1 < columns)
				{
					char separator;
					if (!(values >> separator) || separator != ',') return false;
				}
			}
			values >> std::ws;
			return values.eof();
		}

		template<class Solver, class = void> struct HasClearBatchCache : std::false_type {};
		template<class Solver> struct HasClearBatchCache<Solver, std::void_t<
			decltype(std::declval<Solver&>().ClearBatchCache())>> : std::true_type {};

		template<class Solver, class = void> struct HasForwardBuffers : std::false_type {};
		template<class Solver> struct HasForwardBuffers<Solver, std::void_t<
			decltype(std::declval<Solver&>().linearPredictionBuffer()),
			decltype(std::declval<Solver&>().predictionBuffer())>> : std::true_type {};

		template<class Solver, class Weights, class Output, class Gradient, class = void>
		struct HasGradientUpdate : std::false_type {};
		template<class Solver, class Weights, class Output, class Gradient>
		struct HasGradientUpdate<Solver, Weights, Output, Gradient, std::void_t<decltype(
			std::declval<Solver&>().getWeightsAndBias(std::declval<Weights&>(),
				std::declval<Output&>(), std::declval<const Gradient&>()))>> : std::true_type {};
	}

	template<class DerivedClass, typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::AdamWSolver<>, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
	class GeneralizedLinearModelBase
	{
	public:
		GeneralizedLinearModelBase(int szi = 1, int szo = 1, OptimizerInitialization initialization = OptimizerInitialization::Eager)
			: inputs(szi), outputs(szo), optimizerInitialization(initialization)
		{
		}

		virtual ~GeneralizedLinearModelBase() = default;
		GeneralizedLinearModelBase(const GeneralizedLinearModelBase&) = default;
		GeneralizedLinearModelBase& operator=(const GeneralizedLinearModelBase&) = default;
		GeneralizedLinearModelBase(GeneralizedLinearModelBase&&) = default;
		GeneralizedLinearModelBase& operator=(GeneralizedLinearModelBase&&) = default;

		virtual OutputType Predict(const InputType& input)
		{
			return solver.activationFunction(W * input + b);
		}

		virtual BatchOutputType AddBatchWithParamsAdjusment(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			static_cast<DerivedClass*>(this)->AddBatchNoParamsAdjustment(batchInput, batchOutput);

			return solver.getWeightsAndBias(W, b);
		}

		// For composed networks: forward first, compute/backpropagate every
		// derivative with unchanged weights, then apply all parameter updates.
		BatchOutputType ComputeBatchGradient(const BatchOutputType& batchOutput)
		{
			return solver.ComputeGradient(batchOutput);
		}

		void ApplyBatchUpdate()
		{
			// Uses the cached forward values and recorded target/upstream gradient.
			// No forward pass is repeated here.
			solver.getWeightsAndBias(W, b);
		}

		// Reuse ComputeBatchGradient's result from the same recorded batch.
		void ApplyBatchUpdate(const BatchOutputType& gradient)
		{
			if constexpr (Detail::HasGradientUpdate<Solver, WeightsType, OutputType, BatchOutputType>::value)
				solver.getWeightsAndBias(W, b, gradient);
			else
				// Custom solvers with only the original update API remain supported.
				solver.getWeightsAndBias(W, b);
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

		const Solver& getSolver() const { return solver; }

		bool isOptimizerInitialized() const { return optimizerInitialized; }
		OptimizerInitialization getOptimizerInitialization() const { return optimizerInitialization; }

		// Clear the last training batch while preserving parameters and moments.
		// References returned by cache getters must not outlive this call.
		void ClearBatchCache() { solver.ClearBatchCache(); }

		bool saveModel(std::ofstream& os) const
		{
			try
			{
				os << solver.activationFunction.getName() << std::endl;
				os << solver.lossFunction.getName() << std::endl;

				os << inputs << " " << outputs << std::endl;

				const static Eigen::IOFormat csv(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
				if constexpr (std::is_arithmetic_v<WeightsType>)
					os << std::setprecision(std::numeric_limits<double>::max_digits10) << W << '\n' << b << '\n';
				else
				{
					os << W.format(csv) << std::endl;
					os << b.format(csv) << std::endl;
				}
			}
			catch (...)
			{
				return false;
			}

			return static_cast<bool>(os);
		}

		bool loadModel(std::ifstream& is, std::string* error = nullptr)
		{
			if (error) error->clear();
			const auto fail = [&](const std::string& reason) {
				if (error) *error = reason;
				else std::cerr << "Couldn't load model: " << reason << '\n';
				return false;
			};
			try
			{
				if (!is.is_open()) return fail("input file is not open");
				std::string activationName, lossName, remainder;
				int newInputs, newOutputs;
				if (!(is >> activationName >> lossName)) return fail("missing activation or loss name");
				if (activationName != solver.activationFunction.getName())
					return fail("activation mismatch: " + activationName + " vs " + solver.activationFunction.getName());
				if (lossName != solver.lossFunction.getName())
					return fail("loss mismatch: " + lossName + " vs " + solver.lossFunction.getName());
				if (!(is >> newInputs >> newOutputs) || newInputs <= 0 || newOutputs <= 0)
					return fail("invalid layer dimensions");
				if (!std::getline(is, remainder) || remainder.find_first_not_of(" \t\r") != std::string::npos)
					return fail("unexpected text after layer dimensions");

				// Reject impossible dimensions before allocating: even the shortest
				// serialized coefficient consumes at least one byte in this file.
				const auto position = is.tellg();
				if (position == std::streampos(-1)) return fail("cannot read parameter position");
				is.seekg(0, std::ios::end);
				const auto end = is.tellg();
				is.seekg(position);
				const auto coefficients = static_cast<unsigned long long>(newInputs) * newOutputs + newOutputs;
				if (!is || end < position || coefficients > static_cast<unsigned long long>(end - position)
					|| coefficients > static_cast<unsigned long long>(std::numeric_limits<Eigen::Index>::max()) / sizeof(double))
					return fail("parameter data is truncated or dimensions exceed the file size");

				WeightsType newWeights;
				OutputType newBias;
				if constexpr (std::is_arithmetic_v<WeightsType>)
				{
					if (newInputs != 1 || newOutputs != 1) return fail("scalar model requires 1 input and 1 output");
				}
				else
				{
					if constexpr (WeightsType::RowsAtCompileTime != Eigen::Dynamic)
						if (newOutputs != WeightsType::RowsAtCompileTime) return fail("output count does not match fixed weight dimensions");
					if constexpr (WeightsType::ColsAtCompileTime != Eigen::Dynamic)
						if (newInputs != WeightsType::ColsAtCompileTime) return fail("input count does not match fixed weight dimensions");
					if constexpr (OutputType::SizeAtCompileTime != Eigen::Dynamic)
						if (newOutputs != OutputType::SizeAtCompileTime) return fail("output count does not match fixed bias dimensions");
					newWeights.resize(newOutputs, newInputs);
					newBias.resize(newOutputs);
				}
				for (int row = 0; row < newOutputs; ++row)
					if (!Detail::ReadParameterRow(is, newInputs, [&](int column, double value) {
						if constexpr (std::is_arithmetic_v<WeightsType>) newWeights = value;
						else newWeights(row, column) = value;
					})) return fail("invalid or missing weight row " + std::to_string(row));
				for (int row = 0; row < newOutputs; ++row)
					if (!Detail::ReadParameterRow(is, 1, [&](int, double value) {
						if constexpr (std::is_arithmetic_v<OutputType>) newBias = value;
						else newBias(row) = value;
					})) return fail("invalid or missing bias row " + std::to_string(row));

				// Prepare the reset optimizer before publishing any loaded parameters.
				Solver newSolver = solver;
				if constexpr (Detail::HasClearBatchCache<Solver>::value) newSolver.ClearBatchCache();
				const bool initialized = optimizerInitialization == OptimizerInitialization::Eager;
				if (initialized) newSolver.Initialize(newInputs, newOutputs);
				W = std::move(newWeights);
				b = std::move(newBias);
				solver = std::move(newSolver);
				inputs = newInputs;
				outputs = newOutputs;
				optimizerInitialized = initialized;
				return true;
			}
			catch (const std::exception& exception) { return fail(exception.what()); }
		}

	protected:
		void ResetTrainingState()
		{
			if constexpr (Detail::HasClearBatchCache<Solver>::value) solver.ClearBatchCache();
			optimizerInitialized = false;
			if (optimizerInitialization == OptimizerInitialization::Eager) EnsureOptimizerInitialized();
		}

		void EnsureOptimizerInitialized()
		{
			if (!optimizerInitialized)
			{
				solver.Initialize(inputs, outputs);
				optimizerInitialized = true;
			}
		}

		int inputs;
		int outputs;

		WeightsType W;
		OutputType b;

		Solver solver;
		OptimizerInitialization optimizerInitialization;
		bool optimizerInitialized = false;
	};



	template<typename InputType = Eigen::VectorXd, typename OutputType = Eigen::VectorXd, typename WeightsType = Eigen::MatrixXd, class Solver = SGD::AdamWSolver<>, class BatchInputType = Eigen::MatrixXd, class BatchOutputType = BatchInputType>
	class GeneralizedLinearModel : public GeneralizedLinearModelBase<GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>, InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>
	{
	public:
		using BaseType = GeneralizedLinearModelBase<GeneralizedLinearModel<InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>, InputType, OutputType, WeightsType, Solver, BatchInputType, BatchOutputType>;

		GeneralizedLinearModel(int szi = 1, int szo = 1, OptimizerInitialization initialization = OptimizerInitialization::Eager) : BaseType(szi, szo, initialization)
		{
			Initialize(szi, szo);
		}

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			for (int j = 0; j < BaseType::W.cols(); ++j)
				for (int i = 0; i < BaseType::W.rows(); ++i)
					BaseType::W(i, j) = initializer.get(BaseType::getNrInputs(), BaseType::getNrOutputs());
			BaseType::b.setZero();
			BaseType::ResetTrainingState();
		}


		void AddBatchNoParamsAdjustment(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			BaseType::EnsureOptimizerInitialized();
			BaseType::solver.AddBatch(batchInput, batchOutput);

			if constexpr (Detail::HasForwardBuffers<Solver>::value)
			{
				RecordForward();
			}
			else
			{
				// Preserve support for non-gradient solvers with the legacy interface.
				BatchOutputType linpred;
				LinearPredictBatch(batchInput, linpred);
				BatchOutputType pred;
				ActivateBatch(linpred, pred);
				BaseType::solver.setLinearPrediction(linpred);
				BaseType::solver.setPrediction(pred);
			}
		}

		void AddBatchNoParamsAdjustment(const BatchInputType& batchInput)
		{
			BaseType::EnsureOptimizerInitialized();
			BaseType::solver.AddBatchInput(batchInput);
			RecordForward();
		}

		// Inference owns only local activation workspaces; training caches survive
		// unchanged if prediction is interleaved with a pending backward pass.
		BatchOutputType PredictBatch(const BatchInputType& batchInput)
		{
			BatchOutputType linear, prediction;
			LinearPredictBatch(batchInput, linear);
			ActivateBatch(linear, prediction);
			return prediction;
		}

		BatchInputType BackpropagateBatch(const BatchOutputType& grad) const
		{
			BatchInputType res(BaseType::W.cols(), grad.cols());
			res.noalias() = BaseType::W.transpose() * grad;
			return res;
		}

	protected:
		void LinearPredictBatch(const BatchInputType& batchInput, BatchOutputType& linear)
		{
			linear.resize(BaseType::W.rows(), batchInput.cols());
			linear.noalias() = BaseType::W * batchInput;
			linear.colwise() += BaseType::b;
		}

		void ActivateBatch(const BatchOutputType& linear, BatchOutputType& prediction)
		{
			prediction.resize(linear.rows(), linear.cols());
			// Activations remain per-column: softmax and parameterized/custom
			// activations must retain their existing vector semantics.
			for (Eigen::Index i = 0; i < linear.cols(); ++i)
				prediction.col(i) = BaseType::solver.activationFunction(linear.col(i));
		}

		void RecordForward()
		{
			auto& linear = BaseType::solver.linearPredictionBuffer();
			LinearPredictBatch(BaseType::solver.getInput(), linear);
			ActivateBatch(linear, BaseType::solver.predictionBuffer());
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			if (BaseType::optimizerInitialization == OptimizerInitialization::Eager)
				BaseType::EnsureOptimizerInitialized();

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

		GeneralizedLinearModel(int szi = 1, int szo = 1, OptimizerInitialization initialization = OptimizerInitialization::Eager) : BaseType(szi, szo, initialization)
		{
			Initialize(szi, szo);
		}

		void Initialize(Initializers::WeightsInitializerInterface& initializer)
		{
			BaseType::W = initializer.get();
			BaseType::b = 0.;
			BaseType::ResetTrainingState();
		}

		void AddBatchNoParamsAdjustment(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
		{
			BaseType::EnsureOptimizerInitialized();
			BaseType::solver.AddBatch(batchInput, batchOutput);

			if constexpr (Detail::HasForwardBuffers<Solver>::value)
			{
				auto& linear = BaseType::solver.linearPredictionBuffer();
				linear = (BaseType::W * BaseType::solver.getInput()).array() + BaseType::b;
				ActivateBatch(linear, BaseType::solver.predictionBuffer());
			}
			else
			{
				const Eigen::RowVectorXd linear = (BaseType::W * batchInput).array() + BaseType::b;
				Eigen::RowVectorXd prediction;
				ActivateBatch(linear, prediction);
				BaseType::solver.setLinearPrediction(linear);
				BaseType::solver.setPrediction(prediction);
			}
		}

		Eigen::RowVectorXd PredictBatch(const Eigen::RowVectorXd& batchInput)
		{
			const Eigen::RowVectorXd linear = (BaseType::W * batchInput).array() + BaseType::b;
			Eigen::RowVectorXd prediction;
			ActivateBatch(linear, prediction);
			return prediction;
		}

		Eigen::RowVectorXd BackpropagateBatch(const Eigen::RowVectorXd& grad) const
		{
			return BaseType::W * grad;
		}

	protected:
		void ActivateBatch(const Eigen::RowVectorXd& linear, Eigen::RowVectorXd& prediction)
		{
			prediction.resize(linear.size());
			for (Eigen::Index i = 0; i < linear.size(); ++i)
				prediction(i) = BaseType::solver.activationFunction(linear(i));
		}

		void Initialize(int szi = 1, int szo = 1)
		{
			if (BaseType::optimizerInitialization == OptimizerInitialization::Eager)
				BaseType::EnsureOptimizerInitialized();
			BaseType::W = 0;
			BaseType::b = 0;
		}

		double BackpropagateGradient(const double& grad) const
		{
			return BaseType::W * grad;
		}
	};

}
