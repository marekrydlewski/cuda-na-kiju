#pragma once
#include "CPUMiningAlgorithmBase.h"

#include "../../GPUPatternMining.Contract/Graph.h"
#include "../../GPUPatternMining.Contract/CinsNode.h"
#include "../../GPUPatternMining.Contract/SubcliquesContainer.h"
#include "../../GPUPatternMining.Contract/CliquesContainer.h"
#include "../../GPUPatternMining.Contract/Enity/DataFeed.h"
#include "../../GPUPatternMining.Contract/Hashers.h"
#include "CPUMiningAlgorithmSeq.h"

#include <unordered_map>
#include <vector>

class CPUMiningAlgorithmSeqV2 :
	public CPUMiningAlgorithmSeq
{
public:

	CPUMiningAlgorithmSeqV2();
	virtual ~CPUMiningAlgorithmSeqV2();

	std::vector<std::vector<unsigned short>> filterMaximalCliques(float prevalence) override;

private:
	std::vector<std::vector<unsigned short>> getPrevalentMaxCliques(
		std::vector<unsigned short>& clique,
		float prevalence,
		std::vector<std::vector<unsigned short>>& cliquesToProcess);

};

