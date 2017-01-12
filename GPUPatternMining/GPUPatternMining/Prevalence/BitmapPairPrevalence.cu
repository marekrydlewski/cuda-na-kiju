#include "BitmapPairPrevalence.h"

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/tuple.h>



namespace Prevalence
{
	namespace Bitmap
	{
		__global__
			void setPrevalentFlag(
				float minimalPrevalence
				, unsigned int count
				, thrust::device_ptr<float> a
				, thrust::device_ptr<float> b
				, thrust::device_ptr<bool> flag
				, thrust::device_ptr<unsigned int> writePos)
		{
			unsigned int tid = computeLinearAddressFrom2D();

			if (tid < count)
			{
				bool write = min(a[tid], b[tid]) >= minimalPrevalence;
				flag[tid] = write;
				writePos[tid] = write;
			}
		}
		// -------------------------------------------------------------------------------------------------

//		__host__ __device__
//		FeatureTypePair FeatureInstancesTupleToFeatureTypePair::operator()(thrust::tuple<FeatureInstance, FeatureInstance> ftt)


		// -------------------------------------------------------------------------------------------------

		__global__
			void writeThroughtMask(
				unsigned int count
				, thrust::device_ptr<FeatureTypePair> uniques
				, thrust::device_ptr<bool> mask
				, thrust::device_ptr<unsigned int> writePos
				, thrust::device_ptr<FeatureTypePair> prevalentPairs)
		{
			unsigned int tid = computeLinearAddressFrom2D();

			if (tid < count)
				if (mask[tid])
					prevalentPairs[writePos[tid]] = uniques[tid];
				
			
		}
		// -------------------------------------------------------------------------------------------------

		BitmapPairPrevalenceCounter::BitmapPairPrevalenceCounter(
			std::vector<TypeCount>& typesCounts
		) :  typeCountMap(std::map<unsigned int, unsigned short>())
		{
			for (const TypeCount& tc : typesCounts)
			{
				typeCountMap[tc.type] = static_cast<unsigned short>(tc.count);
			}
		}
		// -------------------------------------------------------------------------------------------------

		BitmapPairPrevalenceCounter::~BitmapPairPrevalenceCounter()
		{

		}
		// -------------------------------------------------------------------------------------------------

		

		thrust::device_vector<FeatureTypePair> BitmapPairPrevalenceCounter::getPrevalentPairConnections(
			float minimalPrevalence
			, PlaneSweepTableInstanceResultPtr planeSweepResult
		)
		{
			const unsigned int uniquesCount = planeSweepResult->uniques.size();
			thrust::host_vector<thrust::tuple<FeatureInstance, FeatureInstance>> localUniques = planeSweepResult->uniques;

			std::vector<unsigned int> pending;

			thrust::device_vector<unsigned int> idxs(uniquesCount);
			thrust::sequence(idxs.begin(), idxs.end());

			thrust::device_vector<unsigned int> aCounts(uniquesCount);
			thrust::device_vector<unsigned int> bCounts(uniquesCount);
			{
				thrust::host_vector<unsigned int> haCounts(uniquesCount);
				thrust::host_vector<unsigned int> hbCounts(uniquesCount);

				for (thrust::tuple<FeatureInstance, FeatureInstance> tup : localUniques)
				{
					haCounts.push_back(typeCountMap[tup.get<0>().fields.featureId]);
					hbCounts.push_back(typeCountMap[tup.get<1>().fields.featureId]);
				}

				aCounts = haCounts;
				bCounts = hbCounts;
			}

			thrust::device_vector<float> aResults(uniquesCount);
			thrust::device_vector<float> bResults(uniquesCount);


			auto tempResultA = thrust::device_vector<FeatureInstance>(planeSweepResult->pairsA.size());
			auto tempResultB = thrust::device_vector<FeatureInstance>(planeSweepResult->pairsB.size());

			UniqueTupleCountFunctor aPrev;
			{
				aPrev.data = planeSweepResult->pairsA.data();
				aPrev.begins = planeSweepResult->indices.data();
				aPrev.count = planeSweepResult->counts.data();
				aPrev.typeCount = aCounts.data();
				aPrev.uniquesOutput = tempResultA.data();
				aPrev.results = aResults.data();
			}

			UniqueTupleCountFunctor bPrev;
			{
				bPrev.data = planeSweepResult->pairsB.data();
				bPrev.begins = planeSweepResult->indices.data();
				bPrev.count = planeSweepResult->counts.data();
				bPrev.typeCount = bCounts.data();
				bPrev.uniquesOutput = tempResultB.data();
				bPrev.results = bResults.data();
			}

			thrust::for_each(thrust::device, idxs.begin(), idxs.end(), aPrev);
			thrust::for_each(thrust::device, idxs.begin(), idxs.end(), bPrev);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			
			{
				// testing
				thrust::host_vector<float> a = aResults;
				thrust::host_vector<float> b = bResults;
				for (int i = 0; i < a.size(); ++i)
				{
					printf("%f %f\n", a[i], b[i]);
				}
			}

			thrust::device_vector<bool> flags(uniquesCount);
			thrust::device_vector<unsigned int> writePos(uniquesCount);

			dim3 grid;
			findSmallest2D(uniquesCount, 256, grid.x, grid.y);

			Prevalence::Bitmap::setPrevalentFlag <<< grid, 256 >>> (
				minimalPrevalence
				, uniquesCount
				, aResults.data()
				, bResults.data()
				, flags.data()
				, writePos.data()
				);

			cudaDeviceSynchronize();

			unsigned int prevalentCount;
			{
				unsigned int lastEl = writePos[uniquesCount - 1];
				printf("lastEl %i\n", lastEl);
				thrust::exclusive_scan(thrust::device, writePos.begin(), writePos.end(), writePos.begin());
				prevalentCount = lastEl + writePos[uniquesCount - 1];
			}

			printf("Prevalent count %i\n", prevalentCount);

			thrust::device_vector<FeatureTypePair> transformed(uniquesCount);
			auto f_trans = FeatureInstancesTupleToFeatureTypePair();

			thrust::transform(
				thrust::device
				, planeSweepResult->uniques.begin()
				, planeSweepResult->uniques.end()
				, transformed.begin()
				, f_trans
			);
			
			thrust::device_vector<FeatureTypePair> dResult(prevalentCount);
			
			writeThroughtMask <<< grid, 256 >>> (
				prevalentCount
				, transformed.data()
				, flags.data()
				, writePos.data()
				, dResult.data()
				);
				
			cudaDeviceSynchronize();

			return dResult;
		}
		// -------------------------------------------------------------------------------------------------
	}
}
