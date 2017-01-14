#pragma once
// --------------------------------------------------------------------------------------------------


#include "..\PlaneSweep\PlaneSweepTableInstanceResult.h"
#include "..\InstanceTree/IntanceTablesMapCreator.h"
#include "..\InstanceTree/InstanceTypedNeighboursMapCreator.h"
// --------------------------------------------------------------------------------------------------


namespace InstanceTree
{
	typedef std::vector<unsigned short> CliqueCandidate;
	typedef std::vector<CliqueCandidate> CliquesCandidates;
	// ----------------------------------------------------------------------------------------------
	
	/*
		assuming that candidate cliques are
			ABC
			CDE
		then content of CliquesInstances is as follows 
			
			std::vector
			1:	device_vector a1 a2 a3 c1 c2
			2:  device_vector b1 b1 b2 d1 d2
			3:  device_vector c2 c1 c2 e1 e2

	*/
	typedef std::vector<thrust::device_vector<FeatureInstance>> CliquesInstances;
	// ----------------------------------------------------------------------------------------------

	struct InstanceTreeResult
	{
		CliquesInstances instances;
		thrust::device_vector<unsigned int> begins;
		thrust::device_vector<unsigned int> counts;
	};
	// ----------------------------------------------------------------------------------------------
	
	typedef std::shared_ptr<InstanceTreeResult> InstanceTreeResultPtr;
	// ----------------------------------------------------------------------------------------------

	class InstanceTree
	{
	public:
		InstanceTree(
			PlaneSweepTableInstanceResultPtr planeSweepResult
			, IntanceTablesMapCreator::ITMPackPtr instanceTablePack
			, InstanceTypedNeighboursMapCreator::ITNMPackPtr typedInstanceNeighboursPack
		);

		InstanceTreeResultPtr getInstancesResult(CliquesCandidates cliquesCandidates);

	private:

		PlaneSweepTableInstanceResultPtr planeSweepResult;
		IntanceTablesMapCreator::ITMPackPtr instanceTablePack;
		InstanceTypedNeighboursMapCreator::ITNMPackPtr typedInstanceNeighboursPack;
	};
}