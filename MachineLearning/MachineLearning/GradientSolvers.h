#pragma once

#include <Eigen/Eigen>
//#include <unsupported/Eigen/MatrixFunctions>

#include "ActivationFunctions.h"
#include "CostFunctions.h"

#include <iostream>

namespace SGD
{

	const double eps = 1E-8;

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

		double alpha = 0.001;
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
			return getLoss(pred, target);
		}

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

	protected:
		inline BatchOutputType getGrad()
		{
			BatchOutputType lossLinkGrad(target.rows(), target.cols());

			const double norm = 1. / input.cols();

			for (int c = 0; c < target.cols(); ++c)
			{
				const OutputType lossDerivative = BaseType::lastLayer ? BaseType::lossFunction.derivative(pred.col(c), target.col(c)) : target.col(c);

				if (BaseType::activationFunction.isDerivativeJacobianMatrix())
					lossLinkGrad.col(c) = norm * BaseType::activationFunction.derivative(linpred.col(c)).transpose() * lossDerivative;
				else
					lossLinkGrad.col(c) = norm * BaseType::activationFunction.derivative(linpred.col(c)).cwiseProduct(lossDerivative);

				// clip it if necessary
				const double n = sqrt(lossLinkGrad.col(c).cwiseProduct(lossLinkGrad.col(c)).sum());
				if (n > BaseType::lim)
					lossLinkGrad.col(c) *= BaseType::lim / n;
			}

			return lossLinkGrad;
		}


	private:
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
			return getLoss(pred, target);
		}

		double getLoss(const Eigen::RowVectorXd& p, const Eigen::RowVectorXd& t) const
		{
			double cost = 0;

			for (int c = 0; c < t.cols(); ++c)
				cost += BaseType::lossFunction(p(c), t(c));

			return cost;
		}

	protected:
		inline Eigen::RowVectorXd getGrad()
		{
			Eigen::RowVectorXd lossLinkGrad;
			lossLinkGrad.resize(target.cols());

			const double norm = 1. / input.cols();

			for (int c = 0; c < target.cols(); ++c)
			{
				lossLinkGrad(c) = norm * BaseType::activationFunction.derivative(linpred(c)) * (BaseType::lastLayer ? BaseType::lossFunction.derivative(pred(c), target(c)) : target(c));
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

			b -= BaseType::alpha * lossLinkGrad.rowwise().sum();

			const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
			w -= BaseType::alpha * wAdj;

			BaseType::alpha *= decay; //learning rate could decrease over time

			return lossLinkGrad;
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

		double getWeightsAndBias(double& w, double& b)
		{
			const double lossLinkGrad = BaseType::getGrad();

			b -= BaseType::alpha * lossLinkGrad;

			const double wAdj = lossLinkGrad * BaseType::input.transpose();
			w -= BaseType::alpha * wAdj;

			BaseType::alpha *= decay; //learning rate could decrease over time

			return lossLinkGrad;
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

			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.rowwise().sum();

			b += BaseType::mb;

			const WeightsType wAdj = lossLinkGrad * BaseType::BaseType::input.transpose();
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;

			w += BaseType::mW;

			return lossLinkGrad;
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

			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.sum();
			b += BaseType::mb;

			const double wAdj = (lossLinkGrad * BaseType::BaseType::input.transpose())(0);
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;

			w += BaseType::mW;

			return lossLinkGrad;
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

			const OutputType mbPrev = BaseType::mb;
			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.rowwise().sum();
			b += -BaseType::beta * mbPrev + (1. + BaseType::beta) * BaseType::mb;

			const WeightsType mWPrev = BaseType::mW;
			const WeightsType wAdj = lossLinkGrad * BaseType::BaseType::input.transpose();
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;
			w += -BaseType::beta * mWPrev + (1. + BaseType::beta) * BaseType::mW;

			return lossLinkGrad;
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

			const double mbPrev = BaseType::mb;
			BaseType::mb = BaseType::beta * BaseType::mb - BaseType::BaseType::alpha * lossLinkGrad.sum();
			b += -BaseType::beta * mbPrev + (1. + BaseType::beta) * BaseType::mb;

			const double wAdjPrev = BaseType::mW;
			const double wAdj = (lossLinkGrad * BaseType::BaseType::input.transpose())(0);
			BaseType::mW = BaseType::beta * BaseType::mW - BaseType::BaseType::alpha * wAdj;
			w += -BaseType::beta * wAdjPrev + (1. + BaseType::beta) * BaseType::mW;

			return lossLinkGrad;
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
			sb = OutputType::Zero(szo);
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();

			sb += lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();

			const OutputType sba = sb + OutputType::Constant(sb.size(), eps);
			b -= BaseType::alpha * lossLinkGrad.rowwise().sum().cwiseProduct(sba.cwiseSqrt().cwiseInverse());

			const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
			sW += wAdj.cwiseProduct(wAdj);

			const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
			w -= BaseType::alpha * wAdj.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());

			return lossLinkGrad;
		}

	private:
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

			sb += lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
			b -= BaseType::alpha * lossLinkGrad.sum() / sqrt(sb + eps);

			const double wAdj = (lossLinkGrad * BaseType::input.transpose())(0);
			sW += wAdj * wAdj;

			w -= BaseType::alpha * wAdj / sqrt(sW + eps);

			return lossLinkGrad;
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
			sb = OutputType::Zero(szo);
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			const BatchOutputType lossLinkGrad = BaseType::getGrad();

			sb = beta * sb + (1. - beta) * lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();

			const OutputType sba = sb + OutputType::Constant(sb.size(), eps);
			b -= BaseType::alpha * lossLinkGrad.rowwise().sum().cwiseProduct(sba.cwiseSqrt().cwiseInverse());

			const WeightsType wAdj = lossLinkGrad * BaseType::input.transpose();
			sW = beta * sW + (1. - beta) * wAdj.cwiseProduct(wAdj);

			const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
			w -= BaseType::alpha * wAdj.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());

			return lossLinkGrad;
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

			sb = beta * sb + (1. - beta) * lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
			b -= BaseType::alpha * lossLinkGrad.sum() / sqrt(sb + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput.transpose())(0);
			sW = beta * sW + (1. - beta) * wAdj * wAdj;

			w -= BaseType::alpha * wAdj / sqrt(sW + eps);

			return lossLinkGrad;
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
			sW = WeightsType::Zero(szo, szi);
			sb = OutputType::Zero(szo);
			mW = WeightsType::Zero(szo, szi);
			mb = OutputType::Zero(szo);
			step = 0;
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			++step;

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			const BatchOutputType lossLinkGrad = BaseType::getGrad();

			mb = beta1 * mb - (1. - beta1) * lossLinkGrad.rowwise().sum();
			sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();
						
			mb *= div1;
			sb *= div2;

			const OutputType sba = sb + OutputType::Constant(sb.size(), eps);
			b += BaseType::alpha * mb.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

			const WeightsType wAdj = lossLinkGrad * BaseType::getInput().transpose();

			mW = beta1 * mW - (1. - beta1) * wAdj;
			mW *= div1;
			sW = beta2 * sW + (1. - beta2) * wAdj.cwiseProduct(wAdj);
			sW *= div2;

			const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
			w += BaseType::alpha * mW.cwiseProduct(sWa.cwiseSqrt().cwiseInverse());

			return lossLinkGrad;
		}

		int setParams(const std::vector<double>& p)
		{
			step = 0;

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
		double beta2 = 0.995;

	private:
		int step = 0;

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
			sW = 0;
			sb = 0;
			mb = 0;
			mW = 0;
			step = 0;
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			++step;
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			mb = beta1 * mb - (1. - beta1) * lossLinkGrad.sum();
			mb *= div1;
			sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
			sb *= div2;

			b += BaseType::alpha * mb / sqrt(sb + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);

			mW = beta1 * mW - (1. - beta1) * wAdj;
			mW *= div1;
			sW = beta2 * sW + (1. - beta2) * wAdj * wAdj;
			sW *= div2;

			w += BaseType::alpha * mW / sqrt(sW + eps);

			return lossLinkGrad;
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
		double beta2 = 0.995;

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
			sW = WeightsType::Zero(szo, szi);
			sb = OutputType::Zero(szo);
			mW = WeightsType::Zero(szo, szi);
			mb = OutputType::Zero(szo);
			step = 0;
		}

		BatchOutputType getWeightsAndBias(WeightsType& w, OutputType& b)
		{
			++step;

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			const BatchOutputType lossLinkGrad = BaseType::getGrad();

			mb = beta1 * mb - (1. - beta1) * lossLinkGrad.rowwise().sum();
			sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad).rowwise().sum();

			mb *= div1;
			sb *= div2;

			const OutputType sba = sb + OutputType::Constant(sb.size(), eps);
			b += BaseType::alpha * mb.cwiseProduct(sba.cwiseSqrt().cwiseInverse());

			const WeightsType wAdj = lossLinkGrad * BaseType::getInput().transpose();

			mW = beta1 * mW - (1. - beta1) * wAdj;
			mW *= div1;
			sW = beta2 * sW + (1. - beta2) * wAdj.cwiseProduct(wAdj);
			sW *= div2;

			const WeightsType sWa = sW + WeightsType::Constant(sW.rows(), sW.cols(), eps);
			w += BaseType::alpha * (mW.cwiseProduct(sWa.cwiseSqrt().cwiseInverse()) - lambda * w);

			return lossLinkGrad;
		}

		int setParams(const std::vector<double>& p)
		{
			step = 0;

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
		double beta2 = 0.995;
		double lambda = 0.001; // wuth lambda 0 is the same as Adam

	private:
		int step = 0;

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
			sW = 0;
			sb = 0;
			mb = 0;
			mW = 0;
			step = 0;
		}

		Eigen::RowVectorXd getWeightsAndBias(double& w, double& b)
		{
			++step;
			const Eigen::RowVectorXd lossLinkGrad = BaseType::getGrad();

			const double div1 = 1. / (1. - pow(beta1, step));
			const double div2 = 1. / (1. - pow(beta2, step));

			mb = beta1 * mb - (1. - beta1) * lossLinkGrad.sum();
			mb *= div1;
			sb = beta2 * sb + (1. - beta2) * lossLinkGrad.cwiseProduct(lossLinkGrad).sum();
			sb *= div2;

			b += BaseType::alpha * mb / sqrt(sb + eps);

			const double wAdj = (lossLinkGrad * BaseType::getInput().transpose())(0);

			mW = beta1 * mW - (1. - beta1) * wAdj;
			mW *= div1;
			sW = beta2 * sW + (1. - beta2) * wAdj * wAdj;
			sW *= div2;

			w += BaseType::alpha * (mW / sqrt(sW + eps) - lambda * w);

			return lossLinkGrad;
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
		double beta2 = 0.995;
		double lambda = 0.001; // wuth lambda 0 is the same as Adam

	private:
		int step = 0;

		double sW = 0;
		double sb = 0;

		double mW = 0;
		double mb = 0;
	};
}
