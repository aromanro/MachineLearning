#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace RuntimeSettings
{
	inline int DefaultEigenThreads()
	{
		// Retain the measured eight-thread cap without requesting more than the
		// reported hardware supports. An unknown hardware count falls back to one.
		return static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
	}

	inline int ParseEigenThreads(std::string_view value)
	{
		if (value.empty()) return DefaultEigenThreads();
		int threads;
		const auto parsed = std::from_chars(value.data(), value.data() + value.size(), threads);
		if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || threads < 0)
			throw std::invalid_argument("ML_EIGEN_THREADS must be a nonnegative integer (0 selects the OpenMP default)");
		return threads;
	}

	inline int ConfigureEigenThreads()
	{
		std::string value;
#ifdef _MSC_VER
		char* environment = nullptr;
		size_t length = 0;
		if (_dupenv_s(&environment, &length, "ML_EIGEN_THREADS") == 0 && environment) value = environment;
		std::free(environment);
#else
		if (const char* environment = std::getenv("ML_EIGEN_THREADS")) value = environment;
#endif
		int threads = DefaultEigenThreads();
		try { threads = ParseEigenThreads(value); }
		catch (const std::invalid_argument& error) { std::cerr << error.what() << "; using " << threads << '\n'; }
		Eigen::setNbThreads(threads);
		return Eigen::nbThreads();
	}
}
