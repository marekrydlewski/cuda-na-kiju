#include "catch.hpp"

#include <map> 
#include <thrust/device_vector.h>
#include <thrust/unique.h>
//--------------------------------------------------------------


#define TEST_CUDA_CHECK_RETURN
//--------------------------------------------------------------

#include "BaseCudaTestHandler.h"

#include "../GPUPatternMining/HashMap/gpuhashmapper.h"
#include "../GPUPatternMining/Prevalence/BitmapPairPrevalence.h"


#include <thrust/execution_policy.h>
#include "../GPUPatternMining/Entities/TypeCount.h"
//--------------------------------------------------------------


/*
	Test for graph

	A0-B0-C0-B1-A1-C1
*/
/*
TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | init")
{
	// creating vector with available types
	std::vector<TypeCount> availableTypes = { 
		TypeCount(0x000A, 2 )
		, TypeCount( 0x000B, 2)
		, TypeCount( 0x000C, 2)
	};

	// creating hashMap with types count
	GPUKeyProcessor<unsigned int> *intKeyProcessor = new GPUKeyProcessor<unsigned int>();

	unsigned int hashSize = 5;
	GPUHashMapper<unsigned int, unsigned short, GPUKeyProcessor<unsigned int>> mapper(hashSize, intKeyProcessor);

	unsigned int h_keys[] = { 0x000A, 0x000B, 0x000C };
	unsigned short h_values[] = { 2, 2, 2 };

	unsigned int* c_keys;
	unsigned short* c_values;

	cudaMalloc(reinterpret_cast<void**>(&c_keys), (sizeof(unsigned int) * 3));
	cudaMalloc(reinterpret_cast<void**>(&c_values), (sizeof(unsigned short) * 3));

	cudaMemcpy(c_keys, h_keys, (sizeof(unsigned int) * 3), cudaMemcpyHostToDevice);
	cudaMemcpy(c_values, h_values, (sizeof(unsigned short) * 3), cudaMemcpyHostToDevice);

	mapper.insertKeyValuePairs(c_keys, c_values, 3);

	// init test

	//Prevalence::Bitmap::BitmapPairPrevalenceCounter(availableTypes, &mapper);
}
*/

/*
Test for graph

A0-B0-C0-B1-A1-C1
*/
/*
TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | simple prevalence")
{
	// creating vector with available types
	std::vector<TypeCount> availableTypes = {
		TypeCount(0x000A, 2)
		, TypeCount(0x000B, 2)
		, TypeCount(0x000C, 2)
	};

	// creating hashMap with types count
	GPUKeyProcessor<unsigned int> *intKeyProcessor = new GPUKeyProcessor<unsigned int>();

	unsigned int hashSize = 5;
	GPUHashMapper<unsigned int, unsigned short, GPUKeyProcessor<unsigned int>> mapper(hashSize, intKeyProcessor);

	unsigned int h_keys[] = { 0x000A, 0x000B, 0x000C };
	unsigned short h_values[] = { 2, 2, 2 };

	unsigned int* c_keys;
	unsigned short* c_values;

	cudaMalloc(reinterpret_cast<void**>(&c_keys), (sizeof(unsigned int) * 3));
	cudaMalloc(reinterpret_cast<void**>(&c_values), (sizeof(unsigned short) * 3));

	cudaMemcpy(c_keys, h_keys, (sizeof(unsigned int) * 3), cudaMemcpyHostToDevice);
	cudaMemcpy(c_values, h_values, (sizeof(unsigned short) * 3), cudaMemcpyHostToDevice);

	mapper.insertKeyValuePairs(c_keys, c_values, 3);

	// init test

	//Prevalence::Bitmap::BitmapPairPrevalenceCounter(availableTypes, &mapper);
}
*/
/*
struct uniqueCountFunctor
{
	thrust::device_ptr<int> data;
	thrust::device_ptr<unsigned int> begins;
	thrust::device_ptr<unsigned int> counts;
	
	thrust::device_ptr<signed int> results;

	__host__ __device__
	void operator()(unsigned int idx)
	{
		results[idx] = thrust::distance(
			data + begins[idx]
			, thrust::unique(
				thrust::device,
				data + begins[idx],
				data + begins[idx] + counts[idx]
			) 
		);
	}
};

TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | simple functor mechanism test")
{
	thrust::device_vector<int> data;
	{
			std::vector<int> hData = {
			1, 1, 2, 2, // 0, 4
			2, 3, 3,	// 4, 3
			1, 2, 3, 4, // 7, 4
		};

		data = hData;
	}

	thrust::device_vector<unsigned int> begins;
	{
		thrust::host_vector<unsigned int> hbegins;

		hbegins.push_back(0);
		hbegins.push_back(4);
		hbegins.push_back(7);

		begins = hbegins;
	}

	thrust::device_vector<unsigned int> counts;
	{
		thrust::host_vector<unsigned int> hCounts;

		hCounts.push_back(4);
		hCounts.push_back(3);
		hCounts.push_back(4);

		counts = hCounts;
	}
	
	thrust::device_vector<unsigned int> idxs(3);
	thrust::sequence(idxs.begin(), idxs.end());

	thrust::device_vector<signed int> results(3);


	uniqueCountFunctor f_uq = { data.data(), begins.data(), counts.data(), results.data() };

	thrust::for_each(idxs.begin(), idxs.end(), f_uq);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::vector<unsigned int> expected = { 2, 2, 4 };
	
	thrust::host_vector<signed int> hResult = results;

	REQUIRE(std::equal(expected.begin(), expected.end(), hResult.begin()) == true);
}


TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | flag setter")
{
	thrust::device_vector<float> resultA;
	{
		std::vector<float> resa(33, 0.1f);
		resa.push_back(0.5f);
		resa[15] = 0.6;

		resultA = resa;
	}

	thrust::device_vector<float> resultB;
	{
		std::vector<float> resa(33, 0.1f);
		resa.push_back(0.5f);
		resa[15] = 0.6;
		resa[17] = 0.6;

		resultB = resa;
	}

	thrust::device_vector<bool> flags(34);
	thrust::device_vector<unsigned int> writePos(34);
	
	dim3 grid;

	findSmallest2D(34, 256, grid.x, grid.y);
	
	Prevalence::Bitmap::setPrevalentFlag << < grid, 256 >> > (
		0.5f
		, 34u
		, resultA.data()
		, resultB.data()
		, flags.data()
		, writePos.data()
		);
		
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	thrust::host_vector<bool> gained = flags;

	std::vector<bool> expected(34, false);
	expected[15] = true;
	expected[33] = true;

	REQUIRE(std::equal(expected.begin(), expected.end(), gained.begin()) == true);
}
// -----------------------------------------------------------------
*/

/*
	uniques = { { A1-B1}, {B1-C1}, {C1-D1} ... }
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | tranform throught mask")
{
	const unsigned int uniquesCount = 34;

	thrust::device_vector<thrust::tuple<FeatureInstance, FeatureInstance>> uniques;
	{
		thrust::host_vector<thrust::tuple<FeatureInstance, FeatureInstance>> huniques;

		for (int i = 0; i < uniquesCount; ++i)
		{
			FeatureInstance a;
			a.field = (i << 16) | (1);

			FeatureInstance b;
			b.field = ((i + 1) << 16) | (1);

			huniques.push_back(thrust::make_tuple(a, b));
		}

		uniques = huniques;
	}


	thrust::device_vector<bool> mask;
	{
		std::vector<bool> hmask(uniquesCount, false);
		{
			hmask[15] = true;
			hmask[33] = true;
		}

		mask = hmask;
	}


	thrust::device_vector<unsigned int> writePos;
	{
		thrust::host_vector<unsigned int> hwritePos(uniquesCount);

		std::fill(hwritePos.begin(), hwritePos.begin() + 16, 0);
		std::fill(hwritePos.begin() + 16, hwritePos.begin() + uniquesCount, 1);
	}

	thrust::device_vector<FeatureTypePair> result(2);

	std::vector<FeatureTypePair> expected;
	{
		expected.push_back({ 0x000F0010});
		expected.push_back({ 0x00210022 });
	}

	dim3 grid;
	findSmallest2D(uniquesCount, 256, grid.x, grid.y);

	Prevalence::Bitmap::writeThroughtMask<<< grid, 256 >>>(
		uniquesCount
		, thrust::raw_pointer_cast(uniques.data())
		, mask.data()
		, writePos.data()
		, result.data()
	);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<FeatureTypePair> gained = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), gained.begin()));
}
// -----------------------------------------------------------------


