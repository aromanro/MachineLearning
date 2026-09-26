#pragma once

#include <Eigen/Eigen>
//#include <unsupported/Eigen/MatrixFunctions>

#include <cassert>
#include <stdexcept>
#include <type_traits>

#include "ActivationFunctions.h"
#include "CostFunctions.h"

#include <iostream>

namespace SGD
{
	namespace Detail
	{
		template<class Activation> struct IsSigmoid : std::false_type {};
		template<class Output, class Weights>
		struct IsSigmoid<ActivationFunctions::SigmoidFunction<Output, Weights>> : std::true_type {};

		template<class Activation, class Loss, class Output>
		inline constexpr bool FusedSigmoidBinaryCrossEntropy = IsSigmoid<Activation>::value
			&& (std::is_same_v<Loss, LossFunctions::BinaryCrossEntropyLoss<Output>>
				|| std::is_same_v<Loss, LossFunctions::LogLoss<Output>>);

		template<class Activation, class Output, class Linear, class Prediction, class Gradient>
		Output BackpropagateActivation(Activation& activation, const Linear& linear,
			const Prediction& prediction, const Gradient& gradient)
		{
			if constexpr (std::is_same_v<Activation, ActivationFunctions::SoftmaxFunction<Output>>)
				return activation.derivativeProductFromOutput(prediction, gradient);
			else if constexpr (IsSigmoid<Activation>::value
				|| std::is_same_v<Activation, ActivationFunctions::TanhFunction<Output>>)
			{
				if constexpr (std::is_arithmetic_v<Output>)
					return activation.derivativeFromOutput(prediction) * gradient;
				else
					return activation.derivativeFromOutput(prediction).cwiseProduct(gradient);
			}
			else
			{
				// Custom activations retain their original preactivation-based API.
				if constexpr (std::is_arithmetic_v<Output>)
					return activation.derivative(linear) * gradient;
				else if (activation.isDerivativeJacobianMatrix())
					return activation.derivative(linear).transpose() * gradient;
				else
					return activation.derivative(linear).cwiseProduct(gradient);
			}
		}
	}

	const double eps = 1E-8;

	namespace Detail
	{
		// Validate while dimensions are still available, before changing biases,
		// moments, or the step counter. Raw coefficient loops cannot check shapes.
		template<class Matrix, class Vector, class Input, class Gradient>
		void ValidateAdaptiveUpdate(const Matrix& weights, const Vector& bias,
			const Input& input, const Gradient& gradient, const Matrix& second, const Vector& biasSecond)
		{
			if (weights.rows() != second.rows() || weights.cols() != second.cols()
				|| bias.size() != biasSecond.size())
				throw std::logic_error("Optimizer state is not initialized for these parameter dimensions");
			if (input.cols() == 0 || weights.rows() != bias.size() || input.rows() != weights.cols()
				|| gradient.rows() != weights.rows() || gradient.cols() != input.cols())
				throw std::invalid_argument("Optimizer update requires a nonempty recorded batch and matching gradient dimensions");
		}

		template<class Matrix, class Vector, class Input, class Gradient>
		void ValidateAdamUpdate(const Matrix& weights, const Vector& bias,
			const Input& input, const Gradient& gradient, const Matrix& first, const Vector& biasFirst,
			const Matrix& second, const Vector& biasSecond)
		{
			ValidateAdaptiveUpdate(weights, bias, input, gradient, second, biasSecond);
			if (weights.rows() != first.rows() || weights.cols() != first.cols() || bias.size() != biasFirst.size())
				throw std::logic_error("Optimizer first moment dimensions do not match the parameters");
		}

		// Separate allocations guarantee that these arrays do not alias. Keeping
		// each coefficient's moment and weight updates together avoids extra passes.
		template<class Scalar>
		void ApplyAdamWeights(Eigen::Index size, Scalar* EIGEN_RESTRICT weights,
			Scalar* EIGEN_RESTRICT first, Scalar* EIGEN_RESTRICT second,
			const Scalar* EIGEN_RESTRICT gradient, double alpha, double beta1, double beta2,
			double correction1, double correction2, double decay)
		{
			for (Eigen::Index i = 0; i < size; ++i)
			{
				const Scalar g = gradient[i];
				first[i] = beta1 * first[i] - (1. - beta1) * g;
				second[i] = beta2 * second[i] + (1. - beta2) * g * g;
				weights[i] += alpha * (correction1 * first[i] / (std::sqrt(correction2 * second[i]) + eps) - decay * weights[i]);
			}
		}

		template<class Scalar>
		void ApplyAdaptiveWeights(Eigen::Index size, Scalar* EIGEN_RESTRICT weights,
			Scalar* EIGEN_RESTRICT second, const Scalar* EIGEN_RESTRICT gradient,
			double alpha, double decay, double accumulation)
		{
			for (Eigen::Index i = 0; i < size; ++i)
			{
				const Scalar g = gradient[i];
				second[i] = decay * second[i] + accumulation * g * g;
				weights[i] -= alpha * g / std::sqrt(second[i] + eps);
			}
		}
	}

	template<class ActivationFunction = ActivationFunctions::IdentityFunction<Eigen::VectorXd>, class LossFunction = LossFunctions::L2Loss<Eigen::VectorXd>>
	class GradientDescentSolverCommonImpl
	{
	public:
		int setParams(const std::vector<double>& p)
		{
			if (p.empty()) return 0;

			alpha = p[0];

			if (p.size() < 2) return 1;

			lim = p[1];

			return 2;
		}

		void setLearnRate(double a)
		{
			alpha = a;
		}

		bool lastLayer = true;
		bool firstLayer = true;

		double alpha = 0.0005;
		double lim = 20.;

		ActivationFunction activationFunction;
		LossFunction lossFunction;
	};


	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// Base class for all
	//*************************************************************************************************************************************************************************************************************************************************************************************************************

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType, 
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
	class GradientDescentSolverBase : public GradientDescentSolverCommonImpl<ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverCommonImpl<ActivationFunction, LossFunction>;

		void AddBatch(const BatchInputType& batchInput, const BatchOutputType& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			input = batchInput;
			target = batchOutput;
		}

		// Hidden layers do not have a target until backward supplies a gradient.
		void AddBatchInput(const BatchInputType& batchInput)
		{
			input = batchInput;
		}

		// Used by the recording forward path to reuse model-owned storage.
		BatchOutputType& linearPredictionBuffer() { return linpred; }
		BatchOutputType& predictionBuffer() { return pred; }

		void ClearBatchCache()
		{
			input.resize(input.rows(), 0);
			target.resize(target.rows(), 0);
			linpred.resize(linpred.rows(), 0);
			pred.resize(pred.rows(), 0);
		}

		void setPrediction(const BatchOutputType& output)
		{
			pred = output;
		}

		const BatchOutputType& getPrediction() const
		{
			return pred;
		}

		const BatchInputType& getInput() const
		{
			return input;
		}

		void setLinearPrediction(const BatchOutputType& output) // before calling the link function
		{
			linpred = output;
		}


		const BatchOutputType& getLinearPrediction() const
		{
			return linpred;
		}

		double getLoss() const
		{
			if constexpr (fusedSoftmaxCrossEntropy)
			{
				double cost = 0.;
				for (int c = 0; c < target.cols(); ++c)
					cost += BaseType::lossFunction.fromLogits(linpred.col(c), target.col(c)).sum();
				return cost;
			}
			else if constexpr (Detail::FusedSigmoidBinaryCrossEntropy<ActivationFunction, LossFunction, OutputType>)
			{
				double cost = 0.;
				for (int c = 0; c < target.cols(); ++c)
				{
					const OutputType logits = linpred.col(c).cwiseProduct(BaseType::activationFunction.getSlope())
						+ BaseType::activationFunction.getOffset();
					cost += BaseType::lossFunction.fromLogits(logits, target.col(c)).sum();
				}
				return cost;
			}
			return getLoss(pred, target);
		}

		// Callers with only probabilities keep the epsilon-protected loss API.
		// The recorded fused cross-entropy training losses above use logits.
		double getLoss(const BatchOutputType& p, const BatchOutputType& t) const
		{
			double cost = 0;

			for (int c = 0; c < t.cols(); ++c)
				cost += BaseType::lossFunction(p.col(c), t.col(c)).sum();

			return cost;
		}


		const BatchOutputType& getTarget() const
		{
			return target;
		}

		// Record the target (last layer) or incoming derivative (hidden layer)
		// without changing parameters or recomputing the saved forward pass.
		BatchOutputType ComputeGradient(const BatchOutputType& batchOutput)
		{
			assert(batchOutput.rows() == pred.rows() && batchOutput.cols() == pred.cols());
			target = batchOutput;
			return getGrad();
		}

	protected:
		inline BatchOutputType getGrad()
		{
			BatchOutputType lossLinkGrad(target.rows(), target.cols());

			// Only the loss layer averages over examples. Incoming hidden-layer
			// derivatives already include that reduction.
			const double norm = BaseType::lastLayer ? 1. / input.cols() : 1.;

			for (int c = 0; c < target.cols(); ++c)
			{
				if constexpr (fusedSoftmaxCrossEntropy)
				{
					if (BaseType::lastLayer)
						// Equals p-y for normalized labels; also supports weighted targets.
						lossLinkGrad.col(c) = norm * (pred.col(c) * target.col(c).sum() - target.col(c));
					else
						lossLinkGrad.col(c) = Detail::BackpropagateActivation<ActivationFunction, OutputType>(
							BaseType::activationFunction, linpred.col(c), pred.col(c), target.col(c));
				}
				else if constexpr (Detail::FusedSigmoidBinaryCrossEntropy<ActivationFunction, LossFunction, OutputType>)
				{
					if (BaseType::lastLayer)
						lossLinkGrad.col(c) = norm * (pred.col(c) - target.col(c)).cwiseProduct(BaseType::activationFunction.getSlope());
					else
						lossLinkGrad.col(c) = Detail::BackpropagateActivation<ActivationFunction, OutputType>(
							BaseType::activationFunction, linpred.col(c), pred.col(c), target.col(c));
				}
				else
				{
					const OutputType lossDerivative = BaseType::lastLayer ? BaseType::lossFunction.derivative(pred.col(c), target.col(c)) : target.col(c);
					lossLinkGrad.col(c) = norm * Detail::BackpropagateActivation<ActivationFunction, OutputType>(
						BaseType::activationFunction, linpred.col(c), pred.col(c), lossDerivative);
				}

				// clip it if necessary
				const double n = sqrt(lossLinkGrad.col(c).cwiseProduct(lossLinkGrad.col(c)).sum());
				if (n > BaseType::lim)
					lossLinkGrad.col(c) *= BaseType::lim / n;
			}

			return lossLinkGrad;
		}


	private:
		static constexpr bool fusedSoftmaxCrossEntropy =
			std::is_same_v<ActivationFunction, ActivationFunctions::SoftmaxFunction<OutputType>>
			&& std::is_same_v<LossFunction, LossFunctions::CrossEntropyLoss<OutputType>>;

		BatchOutputType pred;
		BatchOutputType linpred;

		BatchInputType input;
		BatchOutputType target;
	};


	template<class ActivationFunction, class LossFunction>
	class GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
		: public GradientDescentSolverCommonImpl<ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverCommonImpl<ActivationFunction, LossFunction>;

		void AddBatch(const Eigen::RowVectorXd& batchInput, const Eigen::RowVectorXd& batchOutput)
		{
			assert(batchInput.cols() == batchOutput.cols());

			input = batchInput;
			target = batchOutput;
		}

		void AddBatchInput(const Eigen::RowVectorXd& batchInput) { input = batchInput; }
		Eigen::RowVectorXd& linearPredictionBuffer() { return linpred; }
		Eigen::RowVectorXd& predictionBuffer() { return pred; }

		void ClearBatchCache()
		{
			input.resize(0);
			target.resize(0);
			linpred.resize(0);
			pred.resize(0);
		}

		void setPrediction(const Eigen::RowVectorXd& output)
		{
			pred = output;
		}

		const Eigen::RowVectorXd& getPrediction() const
		{
			return pred;
		}

		void setLinearPrediction(const Eigen::RowVectorXd& output) // before calling the link function
		{
			linpred = output;
		}

		const Eigen::RowVectorXd& getLinearPrediction() const
		{
			return linpred;
		}

		const Eigen::RowVectorXd& getTarget() const
		{
			return target;
		}

		const Eigen::RowVectorXd& getInput() const
		{
			return input;
		}

		double getLoss() const
		{
			if constexpr (Detail::FusedSigmoidBinaryCrossEntropy<ActivationFunction, LossFunction, double>)
			{
				double cost = 0.;
				for (int c = 0; c < target.cols(); ++c)
					cost += BaseType::lossFunction.fromLogits(
						BaseType::activationFunction.getSlope() * linpred(c) + BaseType::activationFunction.getOffset(), target(c));
				return cost;
			}
			return getLoss(pred, target);
		}

		double getLoss(const Eigen::RowVectorXd& p, const Eigen::RowVectorXd& t) const
		{
			double cost = 0;

			for (int c = 0; c < t.cols(); ++c)
				cost += BaseType::lossFunction(p(c), t(c));

			return cost;
		}

		Eigen::RowVectorXd ComputeGradient(const Eigen::RowVectorXd& batchOutput)
		{
			assert(batchOutput.size() == pred.size());
			target = batchOutput;
			return getGrad();
		}

	protected:
		inline Eigen::RowVectorXd getGrad()
		{
			Eigen::RowVectorXd lossLinkGrad;
			lossLinkGrad.resize(target.cols());

			const double norm = BaseType::lastLayer ? 1. / input.cols() : 1.;

			for (int c = 0; c < target.cols(); ++c)
			{
				if constexpr (Detail::FusedSigmoidBinaryCrossEntropy<ActivationFunction, LossFunction, double>)
				{
					if (BaseType::lastLayer)
						lossLinkGrad(c) = norm * BaseType::activationFunction.getSlope() * (pred(c) - target(c));
					else
						lossLinkGrad(c) = Detail::BackpropagateActivation<ActivationFunction, double>(
							BaseType::activationFunction, linpred(c), pred(c), target(c));
				}
				else
				{
					const double lossDerivative = BaseType::lastLayer ? BaseType::lossFunction.derivative(pred(c), target(c)) : target(c);
					lossLinkGrad(c) = norm * Detail::BackpropagateActivation<ActivationFunction, double>(
						BaseType::activationFunction, linpred(c), pred(c), lossDerivative);
				}
				// clip it if necessary
				const double n = sqrt(lossLinkGrad(c) * lossLinkGrad(c));
				if (n > BaseType::lim)
					lossLinkGrad(c) *= BaseType::lim / n;
			}

			return lossLinkGrad;
		}

	private:
		Eigen::RowVectorXd pred;
		Eigen::RowVectorXd linpred;

		Eigen::RowVectorXd input;
		Eigen::RowVectorXd target;
	};


	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// The simplest one
	//*************************************************************************************************************************************************************************************************************************************************************************************************************


	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType, 
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
	class GradientDescentSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			// no need for initialization based on size, it's very simple
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			b -= BaseType::alpha * lossLinkGrad.rowwise().sum();

			const WeightsType wAdj = lossLinkGrad * BaseType::getInput().transpose();
			w -= BaseType::alpha * wAdj;

			BaseType::alpha *= decay; //learning rate could decrease over time
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			decay = p[i];
			++i;

			return i;
		}

		double decay = 1.;
	};

	template<class ActivationFunction, class LossFunction>
	class GradientDescentSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			// no need for initialization based on size, it's very simple
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			b -= BaseType::alpha * lossLinkGrad.sum();

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);
			w -= BaseType::alpha * wAdj;

			BaseType::alpha *= decay; //learning rate could decrease over time
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			decay = p[i];
			++i;

			return i;
		}

		double decay = 1.;
	};


	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// Momentum Base
	//*************************************************************************************************************************************************************************************************************************************************************************************************************

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType,
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
		class MomentumBaseSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			mW = WeightsType::Zero(szo, szi);
			mb = OutputType::Zero(szo);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta = p[i];
			++i;

			return i;
		}

		double beta = 0.5;

	protected:
		WeightsType mW;
		OutputType mb;
	};

	template<class ActivationFunction, class LossFunction>
	class MomentumBaseSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			mW = 0;
			mb = 0;
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta = p[i];
			++i;

			return i;
		}

		double beta = 0.5;

	protected:
		double mW;
		double mb;
	};

	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// Momentum
	//*************************************************************************************************************************************************************************************************************************************************************************************************************

	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType, 
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
	class MomentumSolver : public MomentumBaseSolver<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = MomentumBaseSolver<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.rowwise().sum();

			b += BaseType::mb;

			const WeightsType wAdj = lossLinkGrad * BaseType::BaseType::getInput().transpose();
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;

			w += BaseType::mW;
		}
	};

	template<class ActivationFunction, class LossFunction>
	class MomentumSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public MomentumBaseSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = MomentumBaseSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.sum();
			b += BaseType::mb;

			const double wAdj = (lossLinkGrad * BaseType::BaseType::getInput().transpose())(0);
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;

			w += BaseType::mW;
		}
	};


	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// Nesterov Momentum
	//*************************************************************************************************************************************************************************************************************************************************************************************************************



	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType,
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
		class NesterovMomentumSolver : public MomentumBaseSolver<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = MomentumBaseSolver<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		
		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			const OutputType mbPrev = BaseType::mb;
			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.rowwise().sum();
			b += -BaseType::beta * mbPrev + (1. + BaseType::beta) * BaseType::mb;

			const WeightsType mWPrev = BaseType::mW;
			const WeightsType wAdj = lossLinkGrad * BaseType::BaseType::getInput().transpose();
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;
			w += -BaseType::beta * mWPrev + (1. + BaseType::beta) * BaseType::mW;
		}
	};

	template<class ActivationFunction, class LossFunction>
	class NesterovMomentumSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public MomentumBaseSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = MomentumBaseSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			const double mbPrev = BaseType::mb;
			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.sum();
			b += -BaseType::beta * mbPrev + (1. + BaseType::beta) * BaseType::mb;

			const double wAdjPrev = BaseType::mW;
			const double wAdj = (lossLinkGrad * BaseType::BaseType::getInput().transpose())(0);
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;
			w += -BaseType::beta * wAdjPrev + (1. + BaseType::beta) * BaseType::mW;
		}
	};


	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// Ada Grad
	//*************************************************************************************************************************************************************************************************************************************************************************************************************



	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType, 
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
class AdaGradSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sW = WeightsType::Zero(szo, szi);
			weightGradient.resize(szo, szi);
			sb = OutputType::Zero(szo);
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			Detail::ValidateAdaptiveUpdate(w, b, BaseType::getInput(), lossLinkGrad, sW, sb);
			const OutputType bAdj = lossLinkGrad.rowwise().sum();
			sb += bAdj.cwiseProduct(bAdj);

			const OutputType sba = sb + OutputType::Constant(sb.size(), eps);
			b -= BaseType::alpha * bAdj.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

			weightGradient.noalias() = lossLinkGrad * BaseType::getInput().transpose();
			Detail::ApplyAdaptiveWeights(w.size(), w.data(), sW.data(), weightGradient.data(), BaseType::alpha, 1., 1.);
		}

	private:
		WeightsType weightGradient;
		WeightsType sW;
		OutputType sb;
	};

	template<class ActivationFunction, class LossFunction>
	class AdaGradSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sW = 0;
			sb = 0;
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			const double bAdj = lossLinkGrad.sum();
			sb += bAdj * bAdj;
			b -= BaseType::alpha * bAdj / sqrt(sb + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);
			sW += wAdj * wAdj;

			w -= BaseType::alpha * wAdj / sqrt(sW + eps);
		}

	private:
		double sW;
		double sb;
	};


	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// RMSProp
	//*************************************************************************************************************************************************************************************************************************************************************************************************************



	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType, 
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
class RMSPropSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sW = WeightsType::Zero(szo, szi);
			weightGradient.resize(szo, szi);
			sb = OutputType::Zero(szo);
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			Detail::ValidateAdaptiveUpdate(w, b, BaseType::getInput(), lossLinkGrad, sW, sb);
			const OutputType bAdj = lossLinkGrad.rowwise().sum();
			sb = beta * sb + (1. - beta) * bAdj.cwiseProduct(bAdj);

			const OutputType sba = sb + OutputType::Constant(sb.size(), eps);
			b -= BaseType::alpha * bAdj.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

			weightGradient.noalias() = lossLinkGrad * BaseType::getInput().transpose();
			Detail::ApplyAdaptiveWeights(w.size(), w.data(), sW.data(), weightGradient.data(), BaseType::alpha, beta, 1. - beta);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta = p[i];
			++i;

			return i;
		}

		double beta = 0.5;

	private:
		WeightsType weightGradient;
		WeightsType sW;
		OutputType sb;
	};

	template<class ActivationFunction, class LossFunction>
	class RMSPropSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sW = 0;
			sb = 0;
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			const double bAdj = lossLinkGrad.sum();
			sb = beta * sb + (1. - beta) * bAdj * bAdj;
			b -= BaseType::alpha * bAdj / sqrt(sb + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);
			sW = beta * sW + (1. - beta) * wAdj * wAdj;

			w -= BaseType::alpha * wAdj / sqrt(sW + eps);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta = p[i];
			++i;

			return i;
		}

		double beta = 0.5;

	private:
		double sW;
		double sb;
	};

	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// Adam
	//*************************************************************************************************************************************************************************************************************************************************************************************************************


	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType, 
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
class AdamSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sb = OutputType::Zero(szo);
			mb = OutputType::Zero(szo);
			sW = WeightsType::Zero(szo, szi);
			weightGradient.resize(szo, szi);
			mW = WeightsType::Zero(szo, szi);
			step = 0;
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			Detail::ValidateAdamUpdate(w, b, BaseType::getInput(), lossLinkGrad, mW, mb, sW, sb);
			++step;

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			const OutputType bAdj = lossLinkGrad.rowwise().sum();
			mb = beta1 * mb - (1. - beta1) * bAdj;
			sb = beta2 * sb + (1. - beta2) * bAdj.cwiseProduct(bAdj);

			// Keep the stored moments uncorrected; correct only the current update.
			const OutputType sba = (div2 * sb).cwiseSqrt() + OutputType::Constant(sb.size(), eps);
			b += BaseType::alpha * div1 * mb.cwiseQuotient(sba);

			weightGradient.noalias() = lossLinkGrad * BaseType::getInput().transpose();
			Detail::ApplyAdamWeights(w.size(), w.data(), mW.data(), sW.data(), weightGradient.data(),
				BaseType::alpha, beta1, beta2, div1, div2, 0.);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta1 = p[i];
			++i;
			if (p.size() <= i) return i;

			beta2 = p[i];
			++i;

			return i;
		}

		double beta1 = 0.9;
		double beta2 = 0.98;

	private:
		int step = 0;

		WeightsType weightGradient;
		WeightsType sW;
		OutputType sb;

		WeightsType mW;
		OutputType mb;
	};

	template<class ActivationFunction, class LossFunction>
	class AdamSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sb = 0;
			mb = 0;
			sW = 0;
			mW = 0;
			step = 0;
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			++step;

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			const double bAdj = lossLinkGrad.sum();
			mb = beta1 * mb - (1. - beta1) * bAdj;
			sb = beta2 * sb + (1. - beta2) * bAdj * bAdj;

			b += BaseType::alpha * div1 * mb / (sqrt(div2 * sb) + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);

			mW = beta1 * mW - (1. - beta1) * wAdj;
			sW = beta2 * sW + (1. - beta2) * wAdj * wAdj;

			w += BaseType::alpha * div1 * mW / (sqrt(div2 * sW) + eps);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta1 = p[i];
			++i;
			if (p.size() <= i) return i;

			beta2 = p[i];
			++i;

			return i;
		}

		double beta1 = 0.9;
		double beta2 = 0.98;

	private:
		int step = 0;

		double sW = 0;
		double sb = 0;

		double mW = 0;
		double mb = 0;
	};




	//*************************************************************************************************************************************************************************************************************************************************************************************************************
	// AdamW
	//*************************************************************************************************************************************************************************************************************************************************************************************************************


	template<typename InputType = Eigen::VectorXd, typename OutputType = InputType, typename WeightsType = Eigen::MatrixXd, typename BatchInputType = Eigen::MatrixXd, typename BatchOutputType = BatchInputType,
		class ActivationFunction = ActivationFunctions::IdentityFunction<OutputType>, class LossFunction = LossFunctions::L2Loss<OutputType>>
class AdamWSolver : public GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<InputType, OutputType, WeightsType, BatchInputType, BatchOutputType, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sb = OutputType::Zero(szo);
			mb = OutputType::Zero(szo);
			sW = WeightsType::Zero(szo, szi);
			weightGradient.resize(szo, szi);
			mW = WeightsType::Zero(szo, szi);
			step = 0;
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(WeightsType& w, OutputType& b, const BatchOutputType& lossLinkGrad)
		{
			Detail::ValidateAdamUpdate(w, b, BaseType::getInput(), lossLinkGrad, mW, mb, sW, sb);
			++step;
			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			const OutputType bAdj = lossLinkGrad.rowwise().sum();
			mb = beta1 * mb - (1. - beta1) * bAdj;
			sb = beta2 * sb + (1. - beta2) * bAdj.cwiseProduct(bAdj);

			// Bias correction must not become part of next step's stored moments.
			const OutputType sba = (div2 * sb).cwiseSqrt() + OutputType::Constant(sb.size(), eps);
			b += BaseType::alpha * div1 * mb.cwiseQuotient(sba);

			weightGradient.noalias() = lossLinkGrad * BaseType::getInput().transpose();
			Detail::ApplyAdamWeights(w.size(), w.data(), mW.data(), sW.data(), weightGradient.data(),
				BaseType::alpha, beta1, beta2, div1, div2, lambda);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta1 = p[i];
			++i;
			if (p.size() <= i) return i;

			beta2 = p[i];
			++i;
			if (p.size() <= i) return i;

			lambda = p[i];
			++i;

			return i;
		}

		double beta1 = 0.9;
		double beta2 = 0.98;
		double lambda = 0.001; // with lambda 0 is the same as Adam

	private:
		int step = 0;

		WeightsType weightGradient;
		WeightsType sW;
		OutputType sb;

		WeightsType mW;
		OutputType mb;
	};

	template<class ActivationFunction, class LossFunction>
	class AdamWSolver<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction> : public GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>
	{
	public:
		using BaseType = GradientDescentSolverBase<double, double, double, Eigen::RowVectorXd, Eigen::RowVectorXd, ActivationFunction, LossFunction>;

		void Initialize(int szi = 1, int szo = 1)
		{
			sb = 0;
			mb = 0;
			sW = 0;
			mW = 0;
			step = 0;
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();
			getWeightsAndBias(w, b, lossLinkGrad);
			return lossLinkGrad;
		}

		// Apply the supplied local gradient using the recorded batch input.
		void getWeightsAndBias(double& w, double& b, const Eigen::RowVectorXd& lossLinkGrad)
		{
			++step;

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			const double bAdj = lossLinkGrad.sum();
			mb = beta1 * mb - (1. - beta1) * bAdj;
			sb = beta2 * sb + (1. - beta2) * bAdj * bAdj;

			b += BaseType::alpha * div1 * mb / (sqrt(div2 * sb) + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);

			mW = beta1 * mW - (1. - beta1) * wAdj;
			sW = beta2 * sW + (1. - beta2) * wAdj * wAdj;

			w += BaseType::alpha * (div1 * mW / (sqrt(div2 * sW) + eps) - lambda * w);
		}

		int setParams(const std::vector<double>& p)
		{
			int i = BaseType::setParams(p);

			if (p.size() <= i) return i;

			beta1 = p[i];
			++i;
			if (p.size() <= i) return i;

			beta2 = p[i];
			++i;
			if (p.size() <= i) return i;

			lambda = p[i];
			++i;

			return i;
		}

		double beta1 = 0.9;
		double beta2 = 0.98;
		double lambda = 0.001; // with lambda 0 is the same as Adam

	private:
		int step = 0;

		double sW = 0;
		double sb = 0;

		double mW = 0;
		double mb = 0;
	};
}
