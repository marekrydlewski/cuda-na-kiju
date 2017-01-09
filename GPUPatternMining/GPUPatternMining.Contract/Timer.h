#pragma once

#include <iostream>
#include <chrono>

#define fw(what) std::forward<decltype(what)>(what)

namespace tim
{
	/**
	* @ class detailed_measure
	* @ brief Class to measure the execution time of a callable
	*/
	template <
		typename TimeT = std::chrono::milliseconds, class ClockT = std::chrono::system_clock
	>
		struct detailed_measure
	{
		/**
		* @ fn    execution
		* @ brief Returns the quantity (count) of the elapsed time as TimeT units
		*/
		template<typename F, typename ...Args>
		static typename TimeT::rep execution(F&& func, Args&&... args)
		{
			auto start = ClockT::now();

			fw(func)(std::forward<Args>(args)...);

			auto duration = std::chrono::duration_cast<TimeT>(ClockT::now() - start);

			return duration.count();
		}

		/**
		* @ fn    duration
		* @ brief Returns the duration (in chrono's type system) of the elapsed time
		*/
		template<typename F, typename... Args>
		static TimeT duration(F&& func, Args&&... args)
		{
			auto start = ClockT::now();

			fw(func)(std::forward<Args>(args)...);

			return std::chrono::duration_cast<TimeT>(ClockT::now() - start);
		}
	};
}