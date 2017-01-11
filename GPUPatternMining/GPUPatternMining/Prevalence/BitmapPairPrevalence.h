#pragma once

#include <map>

#include <thrust/device_vector.h>

#include "../HashMap/gpuhashmapper.h"

#include "../Entities/InstanceTable.h"
#include "../Entities/TypeCount.h"
#include "../PlaneSweep/PlaneSweepTableInstanceResult.h"
#include "../Entities/FeatureTypePair.h"
//------------------------------------------------------------------

class PlaneSweepTableInstanceResult;

namespace Prevalence
{
	namespace Bitmap
	{

		typedef GPUHashMapper<unsigned int, Entities::InstanceTable, GPUKeyProcessor<unsigned int>> InstanceTableMap;
		typedef std::shared_ptr<InstanceTableMap> InstanceTableMapPtr;

		typedef GPUHashMapper<unsigned int, unsigned short, GPUKeyProcessor<unsigned int>> CountMap;
		typedef std::shared_ptr<CountMap> CountMapPtr;
		//------------------------------------------------------------------

		__global__
		void setPrevalentFlag(
			float minimalPrevalence
			, unsigned int count
			, thrust::device_ptr<float> a
			, thrust::device_ptr<float> b
			, thrust::device_ptr<bool> flag
			, thrust::device_ptr<unsigned int> writePos);
		//------------------------------------------------------------------

		__global__
		void writeThroughtMask(
			unsigned int count
			, thrust::tuple<FeatureInstance, FeatureInstance>* uniques
			, thrust::device_ptr<bool> mask
			, thrust::device_ptr<unsigned int> writePos
			, thrust::device_ptr<FeatureTypePair> prevalentPairs);
		//------------------------------------------------------------------

		class BitmapPairPrevalenceCounter
		{
		public:

			BitmapPairPrevalenceCounter(
				CountMapPtr countMap
				, InstanceTableMapPtr instanceTableMap
				, unsigned int bitmapSize
				, std::vector<TypeCount>& typesCounts
			);

			~BitmapPairPrevalenceCounter();

			/*
			@params

			minimalPrevalence	-	minimal required prevalence
			uniques				-	vector that contains type pairs for prevalence check

			*/
			thrust::host_vector<FeatureTypePair> getPrevalentPairConnections(
				float minimalPrevalence
				, PlaneSweepTableInstanceResultPtr planeSweepResult
			);

		private:

			CountMapPtr gpuCountMap;
			InstanceTableMapPtr instanceTableMap;
			unsigned int bitmapSize;
			std::map<unsigned int, unsigned short> typeCountMap;
		};
	}
}
