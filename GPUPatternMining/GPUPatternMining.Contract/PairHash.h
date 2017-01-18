#pragma once

#include <functional>
#include <utility>


template<typename T>
void hash_combine(std::size_t &seed, T const &key) {
	std::hash<T> hasher;
	seed ^= hasher(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
};


struct pair_hash {
	template <class T1, class T2>
	std::size_t operator () (const std::pair<T1, T2> &p) const {
		std::size_t seed1(0);
		::hash_combine(seed1, p.first);
		::hash_combine(seed1, p.second);

		std::size_t seed2(0);
		::hash_combine(seed2, p.second);
		::hash_combine(seed2, p.first);

		return std::min(seed1, seed2);
	}
};


struct pair_hash_naive {
	template <class T1, class T2>
	std::size_t operator () (const std::pair<T1, T2> &p) const {
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);

		return h1 ^ h2;
	}
};
