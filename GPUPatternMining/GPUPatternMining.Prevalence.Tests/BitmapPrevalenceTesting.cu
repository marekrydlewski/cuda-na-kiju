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
TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | simple")
{
	printf("<Bitmap prevalence | simple>\n");

	std::vector<TypeCount> counts = { { 0xA, 2 },{ 0xB, 2 },{ 0xC, 2 } };
	


	Prevalence::Bitmap::BitmapPairPrevalenceCounter bppc(counts);

	const float minimalPrevalence = 0.6f;

	auto plRes = std::make_shared<PlaneSweepTableInstanceResult>();

	{
		thrust::host_vector<thrust::tuple<FeatureInstance, FeatureInstance>> huniques;

		FeatureInstance a;
		FeatureInstance b;

		a.field = 0x000A0000;
		b.field = 0x000B0000;		
		huniques.push_back(thrust::make_tuple(a, b));

		a.field = 0x000A0001;
		b.field = 0x000C0001;
		huniques.push_back(thrust::make_tuple(a, b));

		a.field = 0x000B0000;
		b.field = 0x000C0000;
		huniques.push_back(thrust::make_tuple(a, b));

		plRes->uniques = huniques;
	}

	{
		thrust::host_vector<FeatureInstance> hPairsA;
		thrust::host_vector<FeatureInstance> hPairsB;

		FeatureInstance a;
		FeatureInstance b;
		
		// A-B
		a.field = 0x000A0000;
		hPairsA.push_back(a);
		b.field = 0x000B0000;
		hPairsB.push_back(b);

		a.field = 0x000A0001;
		hPairsA.push_back(a);
		b.field = 0x000B0001;
		hPairsB.push_back(b);
		
		// A-C
		a.field = 0x000A0001;
		hPairsA.push_back(a);
		b.field = 0x000C0001;
		hPairsB.push_back(b);

		// B-C
		a.field = 0x000B0000;
		hPairsA.push_back(a);
		b.field = 0x000C0000;
		hPairsB.push_back(b);

		a.field = 0x000B0001;
		hPairsA.push_back(a);
		b.field = 0x000C0000;
		hPairsB.push_back(b);

		plRes->pairsA = hPairsA;
		plRes->pairsB = hPairsB;
	}
	
	{
		std::vector<unsigned int> counts = { 2, 1, 2 };
		plRes->counts = counts;
	}

	{
		std::vector<unsigned int> indices = { 0, 2, 3 };
		plRes->indices = indices;
	}
	
	thrust::host_vector<FeatureTypePair> result = bppc.getPrevalentPairConnections(
		minimalPrevalence
		, plRes
	);
	

	printf("pairs\n");
	for (FeatureTypePair& ftp : result)
		printf("0x%8x\n", ftp.combined);
	printf("end pairs\n");

	printf("</Bitmap prevalence | simple>\n");
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
	
	Prevalence::Bitmap::setPrevalentFlag <<< grid, 256 >>> (
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


TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | unique tuple functor")
{
	thrust::device_vector<FeatureInstance> pairsA;
	{
		thrust::host_vector<FeatureInstance> hPairsA;

		FeatureInstance a;

		// A-B
		a.field = 0x000A0000;
		hPairsA.push_back(a);

		a.field = 0x000A0001;
		hPairsA.push_back(a);

		// A-C
		a.field = 0x000A0001;
		hPairsA.push_back(a);

		// B-C
		a.field = 0x000B0000;
		hPairsA.push_back(a);

		a.field = 0x000B0001;
		hPairsA.push_back(a);

		pairsA = hPairsA;
	}

	thrust::device_vector<unsigned int> begins;
	{
		std::vector<unsigned int> hBegins = { 0, 2, 3 };
		begins = hBegins;
	}
	
	thrust::device_vector<unsigned int> counts;
	{
		std::vector<unsigned int> hcounts = { 2, 1, 2 };
		counts = hcounts;
	}


	thrust::device_vector<unsigned int> typesCounts;
	{
		std::vector<unsigned int> hTypesCounts = { 2, 2, 2 };
		typesCounts = hTypesCounts;
	}

	thrust::device_vector<FeatureInstance> uniqueFeatureInstancesInPairType(6);

	thrust::device_vector<float> result(3);

	Prevalence::Bitmap::UniqueTupleCountFunctor f_in;
	{
		f_in.data = pairsA.data();
		f_in.begins = begins.data();
		f_in.count = counts.data();
		f_in.typeCount = typesCounts.data();
		f_in.uniquesOutput = uniqueFeatureInstancesInPairType.data();
		f_in.results = result.data();
	}

	thrust::device_vector<unsigned int> idxs(3);
	thrust::sequence(idxs.begin(), idxs.end());

	thrust::for_each(thrust::device, idxs.begin(), idxs.end(), f_in);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<float> res = result;

	printf("results\n");
	for (float val : res)
		printf("%f\n", val);
	printf("end results\n");
}

TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | unary transfrom")
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

	thrust::device_vector<FeatureTypePair> result(uniquesCount);

	std::vector<FeatureTypePair> expected;
	for (int i = 0; i < uniquesCount; ++i)
	{
		FeatureTypePair ftp;
		ftp.combined = ((i << 16) & 0xFFFF0000) | (i + 1);
		expected.push_back(ftp);
	}

	auto f_trans = Prevalence::Bitmap::FeatureInstancesTupleToFeatureTypePair();

	thrust::transform(
		thrust::device
		, uniques.begin()
		, uniques.end()
		, result.begin()
		, f_trans
	);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<FeatureTypePair> gained = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), gained.begin()));
}

// -----------------------------------------------------------------

/*
	uniques = { { A1-B1}, {B1-C1}, {C1-D1} ... }
*/
TEST_CASE_METHOD(BaseCudaTestHandler, "Bitmap prevalence | write throught mask")
{
	const unsigned int uniquesCount = 34;

	thrust::device_vector<FeatureTypePair> dataFeed;
	{
		std::vector<FeatureTypePair> hdataFeed;
		for (int i = 0; i < uniquesCount; ++i)
		{
			FeatureTypePair ftp;
			ftp.combined = ((i << 16) & 0xFFFF0000) | (i + 1);
			hdataFeed.push_back(ftp);
		}

		dataFeed = hdataFeed;
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

		writePos = hwritePos;
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
		, dataFeed.data()
		, mask.data()
		, writePos.data()
		, result.data()
	);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	thrust::host_vector<FeatureTypePair> gained = result;

	REQUIRE(std::equal(expected.begin(), expected.end(), gained.begin()));
}
// -----------------------------------------------------------------