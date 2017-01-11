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

		
		__global__
			void writeThroughtMask(
				unsigned int count
				, thrust::device_ptr<FeatureInstance> uniques
				, thrust::device_ptr<bool> mask
				, thrust::device_ptr<unsigned int> writePos
				, thrust::device_ptr<FeatureTypePair> prevalentPairs)
		{
			unsigned int tid = computeLinearAddressFrom2D();

			if (tid < count)
			{
				if (mask[tid])
				{
					FeatureTypePair ftp;

					ftp.combined =
						uniques[tid].fields.featureId & 0xFFFF0000
						| ((uniques[tid].get<1>().fields.featureId >> 16) & 0x0000FFFF);

					prevalentPairs[writePos[tid]] = ftp;
				}
			}
		}
		// -------------------------------------------------------------------------------------------------

		BitmapPairPrevalenceCounter::BitmapPairPrevalenceCounter(
			CountMapPtr countMap
			, InstanceTableMapPtr instanceTableMap
			, unsigned int bitmapSize
			, std::vector<TypeCount>& typesCounts
		) : gpuCountMap(countMap)
			, instanceTableMap(instanceTableMap)
			, bitmapSize(bitmapSize)
			, typeCountMap(std::map<unsigned int, unsigned short>())
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

		struct UniqueTupleCountFunctor
		{
			thrust::device_ptr<FeatureInstance> data;
			thrust::device_ptr<unsigned int> begins;
			thrust::device_ptr<unsigned int> counts;
			thrust::device_ptr<unsigned int> count;

			thrust::device_ptr<FeatureInstance> uniquesOutput;

			thrust::device_ptr<float> results;

			__host__ __device__
				void operator()(unsigned int idx)
			{
				results[idx] = thrust::distance(
					uniquesOutput + begins[idx]
					, thrust::unique_copy
					(
						thrust::device
						, data + begins[idx]
						, data + begins[idx] + counts[idx]
						, uniquesOutput + begins[idx]
					)
				) / static_cast<float>(count[idx]);
			}
		};
		// -------------------------------------------------------------------------------------------------


		thrust::host_vector<FeatureTypePair> BitmapPairPrevalenceCounter::getPrevalentPairConnections(
			float minimalPrevalence
			, PlaneSweepTableInstanceResultPtr planeSweepResult
		)
		{
			const unsigned int uniquesCount = planeSweepResult->uniques.size();
			thrust::host_vector<thrust::tuple<FeatureInstance, FeatureInstance>> localUniques = planeSweepResult->uniques;

			// TODO free uniques

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
				aPrev.counts = aCounts.data();
				aPrev.uniquesOutput = tempResultA.data();
				aPrev.results = aResults.data();
			}

			UniqueTupleCountFunctor bPrev;
			{
				bPrev.data = planeSweepResult->pairsB.data();
				bPrev.begins = planeSweepResult->indices.data();
				bPrev.count = planeSweepResult->counts.data();
				bPrev.counts = bCounts.data();
				bPrev.uniquesOutput = tempResultB.data();
				bPrev.results = bResults.data();
			}

			thrust::for_each(thrust::device, idxs.begin(), idxs.end(), aPrev);
			thrust::for_each(thrust::device, idxs.begin(), idxs.end(), bPrev);

			cudaDeviceSynchronize();

			thrust::device_vector<bool> flags(uniquesCount);
			thrust::device_vector<unsigned int> writePos(uniquesCount);

			dim3 grid;
			findSmallest2D(uniquesCount, 256, grid.x, grid.y);

			Prevalence::Bitmap::setPrevalentFlag << < grid, 256 >> > (
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
				thrust::exclusive_scan(thrust::device, writePos.begin(), writePos.end(), writePos.begin());
				prevalentCount = lastEl + writePos[uniquesCount - 1];
			}

			thrust::device_vector<FeatureTypePair> dResult(prevalentCount);

			writeThroughtMask << < grid, 256 >> > (
				prevalentCount
				, thrust::raw_pointer_cast(planeSweepResult->uniques.data())
				, flags.data()
				, writePos.data()
				, dResult.data()
				);

			cudaDeviceSynchronize();

			thrust::host_vector<FeatureTypePair> result = dResult;

			return result;
		}
		// -------------------------------------------------------------------------------------------------
	}
}
