/*
 * utils_gpu.cu
 *
 *  Created on: 27-02-2013
 *      Author: witek
 */

#include "gpuutils.h"
#include <cmath>

void eatDelay() {
	int* d_a;
	int h_a=666;

	d_a=cudaAllocArray(1,&h_a);
	cudaDeallocArray(d_a);

}

std::map<void*, struct AllocInfo > memoryAllocs;

unsigned int intDivCeil(unsigned int x, unsigned int y)
{
	return (x + y - 1) / y;
}

void findSmallest2D(unsigned int size,unsigned int blockSize,unsigned int &x,unsigned int &y) {
	x=intDivCeil(size,blockSize);
	y=intDivCeil(x,65535);
	x=intDivCeil(size,blockSize*y);
}

EqualityCondition::EqualityCondition(unsigned int value) {
	this->value=value;
}

__host__ __device__ unsigned int EqualityCondition::perform(unsigned int x) {
	return x==value;
}

InequalityCondition::InequalityCondition(unsigned int value) {
	this->value=value;
}

__host__ __device__ unsigned int InequalityCondition::perform(unsigned int x) {
	return x!=value;
}



void cudaMemoryReport() {
	std::map<void*, struct AllocInfo >::iterator it;

	if (memoryAllocs.size()==0) {
		printf("Everything freed!\n");
	} else {
		printf("%d allocs left:\n",memoryAllocs.size());
		for (it=memoryAllocs.begin();it!=memoryAllocs.end();it++) {
			printf(" -%dB @ %.8x (%d*%dB) (file %s, line %d)\n",it->second.itemSize*it->second.items,it->first,it->second.items,it->second.itemSize,it->second.filename,it->second.lineNumber);
		}
	}
}

void checkKernelExecution() {
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
}


