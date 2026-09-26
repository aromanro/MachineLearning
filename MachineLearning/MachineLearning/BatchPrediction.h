#pragma once

#include <Eigen/Eigen>
#include <type_traits>
#include <utility>

namespace Prediction
{
	inline constexpr Eigen::Index DefaultBatchSize = 128;

	template<class Model, class = void> struct HasBatchPrediction : std::false_type {};
	template<class Model> struct HasBatchPrediction<Model, std::void_t<decltype(
		std::declval<Model&>().PredictBatch(std::declval<const Eigen::MatrixXd&>()))>> : std::true_type {};

	// Existing custom models can continue to expose only Predict(vector).
	template<class Model>
	Eigen::MatrixXd PredictBatch(Model& model, const Eigen::MatrixXd& input)
	{
		if constexpr (HasBatchPrediction<Model>::value) return model.PredictBatch(input);
		else
		{
			if (input.cols() == 0) return Eigen::MatrixXd(0, 0);
			const Eigen::VectorXd first = model.Predict(input.col(0));
			Eigen::MatrixXd result(first.size(), input.cols());
			result.col(0) = first;
			for (Eigen::Index column = 1; column < input.cols(); ++column)
				result.col(column) = model.Predict(input.col(column));
			return result;
		}
	}
}
