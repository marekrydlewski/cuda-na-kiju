#pragma once


/*
* utils_gpu.h
*
*  Created on: 27-02-2013
*      Author: witek
*/

#ifndef UTILS_GPU_H_
#define UTILS_GPU_H_

#include <stdio.h>
#include <map>

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <iostream>
#include <device_launch_parameters.h>

//#define UTILS_DEBUG_OUTPUT

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		std::cin.get();															\
		exit(1);															\
	} }

#define FL __LINE__,__FILE__


void checkKernelExecution();

struct AllocInfo
{
	unsigned int items;
	unsigned int itemSize;
	unsigned int lineNumber;
	char filename[1000];
};

extern std::map<void*, struct AllocInfo > memoryAllocs;

template<class T> T peekValue(T *tab, unsigned int pos)
{
	T res;

	cudaMemcpy(&res, tab + pos, sizeof(T), cudaMemcpyDeviceToHost);

	return res;
}

template<class T> void dumpDeviceArray(T* tab, char* type, char * sep, unsigned int size)
{
	T* buf = new T[size];
	CUDA_CHECK_RETURN(cudaMemcpy(buf, tab, size * sizeof(T), cudaMemcpyDeviceToHost));
	for (int i = 0; i<size; i++)
	{
		printf("%d. ", i);
		printf(type, buf[i]);
		printf(sep);
	}
	delete[]buf;
}


unsigned int intDivCeil(unsigned int x, unsigned int y);

void findSmallest2D(unsigned int size, unsigned int blocksize, unsigned int &x, unsigned int &y);

__device__ __inline__ unsigned int computeLinearAddressFrom2D()
{
	unsigned int globalX = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int globalY = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int globalWidth = gridDim.x*blockDim.x;
	return globalY*globalWidth + globalX;
}

__host__ __device__ __inline__ unsigned int wordNum(unsigned int x)
{
	return x >> 5;
}

__host__ __device__ __inline__ unsigned int bitNum(unsigned int x)
{
	return x & 0x0000001f;
}

__host__ __device__ __inline__ unsigned int discreetLog2(unsigned int v)
{
	unsigned int r, shift;
	r = (v > 0xFFFF) << 4; v >>= r;
	shift = (v > 0xFF) << 3; v >>= shift; r |= shift;
	shift = (v > 0xF) << 2; v >>= shift; r |= shift;
	shift = (v > 0x3) << 1; v >>= shift; r |= shift;
	r |= (v >> 1);
	return r;
}

__host__ __device__ __inline__ unsigned int countBits(unsigned int v)
{
	v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
	return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}



__host__ __device__ __inline__ unsigned int nthBitPosition(unsigned int v, unsigned int r)
{
	unsigned int a, b, c, d; // Intermediate temporaries for bit count.
	unsigned int bits = 16;
	unsigned int pos = 0;
	unsigned int t;

	a = v - ((v >> 1) & ~0U / 3); //na ka?dej dwójce bitów liczba bitów w tej parze
	b = (a & ~0U / 5) + ((a >> 2) & ~0U / 5); //na ka?dej czwórce bitów liczba bitów w tej czwórce. Poniewaz liczba bitow dost?pnych na przechowanie sumy rosnie szybciej niz liczba bitów w s?owie to dalej nie trzeba robic jednego &.
	c = (b + (b >> 4)) & ~0U / 0x11; // na ka?dej ósemce bitów liczba bitów w tej ósemce
	d = (c + (c >> 8)) & ~0U / 0x101; // na ka?dej szesnastce bitów liczba bitów w tej szesnastce

	d = d & 0x0000ffff;
	t = ~(d<r) + 1;
	pos += bits&t;
	r -= d&t;

	bits = bits >> 1;

	c = (c >> pos) & 0x000000ff;
	t = ~(c<r) + 1;
	pos += bits&t;
	r -= c&t;

	bits = bits >> 1;

	b = (b >> pos) & 0x0000000f;
	t = ~(b<r) + 1;
	pos += bits&t;
	r -= b&t;

	bits = bits >> 1;

	a = (a >> pos) & 0x00000003;
	t = ~(a<r) + 1;
	pos += bits&t;
	r -= a&t;

	t = (v >> pos) & 0x00000001;
	pos += (t<r);

	return pos;
}




template<class T> T* cudaAllocArray(unsigned int size, T *initTab, unsigned int line = 0, char* filename = NULL)
{
	T* result;
	if (size == 0)
	{
		printf("Trying to alloc 0 bytes @ %s (%d). Thats not good...\n", filename, line);
		std::cin.ignore();
		exit(0);
	}



#ifdef UTILS_DEBUG_OUTPUT
	if (filename == NULL) filename = "";
	printf("Trying to allocate %d bytes of memory for %d items of size %dB, (file %s, line %d)\n", sizeof(T)*size, size, sizeof(T), filename, line);
#endif
	CUDA_CHECK_RETURN(cudaMalloc((void**)&result, sizeof(T)*size));
	if (initTab != NULL)
	{
		CUDA_CHECK_RETURN(cudaMemcpy(result, initTab, sizeof(T)*size, cudaMemcpyHostToDevice));
	}

	struct AllocInfo ai;
	ai.itemSize = sizeof(T);
	ai.items = size;
	ai.lineNumber = line;
	if (filename != NULL) strcpy(ai.filename, filename);
	else ai.filename[0] = 0;
	memoryAllocs.insert(std::pair<void*, struct AllocInfo >((void*)result, ai));

	return result;
}

template<class T> T* cudaAllocArray(unsigned int size, unsigned int line = 0, char* filename = NULL)
{
	return cudaAllocArray<T>(size, NULL, line, filename);
}

//void cudaDeallocArray(void *&address);
//void cudaDeallocArray(void *&address,unsigned int line,char* file);

template<class T> void cudaDeallocArray(T *&address, unsigned int line, char* file)
{

	std::map<void*, struct AllocInfo >::iterator it;


	it = memoryAllocs.find(address);

	if (it == memoryAllocs.end())
	{
#ifdef UTILS_DEBUG_OUTPUT
		printf("Trying to dealloc adress %.8x but it is not currently allocated\n", address);
#endif
	}
	else
	{
		CUDA_CHECK_RETURN(cudaFree(address));

		memoryAllocs.erase(memoryAllocs.find(address));
		if (file == NULL) file = "";
#ifdef UTILS_DEBUG_OUTPUT
		printf("Deallocated memory at %.8x @ %s (%d).\n", address, file, line);//(%dB=%d*%dB), allocated @ %s (%d), deallocated @ %s (%d).\n",address,it->second.itemSize*it->second.items,it->second.items,it->second.itemSize,it->second.filename,it->second.lineNumber,file,line);
#endif
		address = NULL;
	}
}

template<class T> void cudaDeallocArray(T *&address)
{
	cudaDeallocArray(address, 0, NULL);
}


void cudaMemoryReport();


/*__device__ __inline__ void intraWarpReduce(volatile unsigned int *a, unsigned int steps) {
int threadInWarp=threadIdx.x&0x1f;

if ((threadIdx.x&0x01)==0x01 && steps>=1) a[threadInWarp]+=a[threadInWarp-1];
if ((threadIdx.x&0x03)==0x03 && steps>=2) a[threadInWarp]+=a[threadInWarp-2];
if ((threadIdx.x&0x07)==0x07 && steps>=3) a[threadInWarp]+=a[threadInWarp-4];
if ((threadIdx.x&0x0f)==0x0f && steps>=4) a[threadInWarp]+=a[threadInWarp-8];
}*/

class EqualityCondition
{
public:
	unsigned int value;
	EqualityCondition(unsigned int value);
	__host__ __device__ unsigned int perform(unsigned int x);
};

class InequalityCondition
{
public:
	unsigned int value;
	InequalityCondition(unsigned int value);
	__host__ __device__ unsigned int perform(unsigned int x);
};

template<class F, int steps> __device__ __inline__ void intraWarpReduce(volatile unsigned int *a, F op)
{
	int threadInWarp = threadIdx.x & 0x1f;

	if (((threadIdx.x & 0x01) == 0x01) && (steps >= 1)) a[threadInWarp] = op.perform(a[threadInWarp]) + op.perform(a[threadInWarp - 1]);
	if (((threadIdx.x & 0x03) == 0x03) && (steps >= 2)) a[threadInWarp] += a[threadInWarp - 2];
	if (((threadIdx.x & 0x07) == 0x07) && (steps >= 3)) a[threadInWarp] += a[threadInWarp - 4];
	if (((threadIdx.x & 0x0f) == 0x0f) && (steps >= 4)) a[threadInWarp] += a[threadInWarp - 8];
	if (((threadIdx.x & 0x1f) == 0x1f) && (steps >= 5)) a[threadInWarp] += a[threadInWarp - 16];
}


__device__ __host__ __inline__ unsigned int firstSeqPos(unsigned int seqNum, unsigned int strideNextSequence)
{
	return seqNum*strideNextSequence;
}

__device__ __host__  __inline__ unsigned int seqPos(unsigned int pos, unsigned int strideNextItem)
{
	return pos*strideNextItem;
}


template<int bitmapWordLength> __host__ __device__ __inline__ unsigned int oldestBit(unsigned int *base, unsigned strideNextItem)
{


	//unsigned int word=*base;
	/*for (int i=bitmapWordLength-1;i>=0;i--) {
	unsigned int word=base[seqPos(i,strideNextItem)];
	pos+=(pos)?countBits(word):((word!=0)*(discreetLog2(word)+1));
	}*/

	unsigned int word;
	unsigned int pos = 0;

	for (int i = 0; i<bitmapWordLength; i++)
	{
		word = base[seqPos(i, strideNextItem)];
		pos = (word != 0)*(discreetLog2(word) + (i << 5)) + (word == 0)*pos;
	}


	//return pos-1;
	return pos;
}


template <class T> void assignToDevice(T* value, T* out)
{
	CUDA_CHECK_RETURN(cudaMemcpy(value, out, sizeof(T), cudaMemcpyHostToDevice));
}


template <class T> __global__ void performSequenceArrayCompact(
	T* array,
	T* output,
	unsigned int* flags,
	unsigned int* scannedFlags,
	unsigned int sequenceCount,
	unsigned int sequenceLength,
	unsigned int strideNextSequence,
	unsigned int strideNextItem,
	unsigned int newCount)
{

	unsigned int threadNum = computeLinearAddressFrom2D();

	if (threadNum<sequenceCount)
	{
		for (int i = 0; i<sequenceLength; i++)
		{
			if (flags[threadNum])
			{
				output[scannedFlags[threadNum] + i*newCount] = array[firstSeqPos(threadNum, strideNextSequence) + seqPos(i, strideNextItem)];
			}
		}
	}

}

template <class T> void compactSequenceArray(T* array, T *output, unsigned int *flags, unsigned int strideNextSequence, unsigned int strideNextItem, unsigned int sequenceLength, unsigned int sequenceCount, unsigned int &newCount)
{
	dim3 grid(1, 1, 1);

	unsigned int *scannedFlags = cudaAllocArray<unsigned int>(sequenceCount, FL);

	thrust::device_ptr<unsigned int> dt_flags = thrust::device_pointer_cast(flags);
	thrust::device_ptr<unsigned int> dt_scannedFlags = thrust::device_pointer_cast(scannedFlags);

	findSmallest2D(sequenceCount, 512, grid.x, grid.y);

	thrust::exclusive_scan(dt_flags, dt_flags + sequenceCount, dt_scannedFlags);

	newCount = peekValue(flags, sequenceCount - 1) + peekValue(scannedFlags, sequenceCount - 1);

	performSequenceArrayCompact << <grid, 512 >> >(
		array,
		output,
		flags,
		scannedFlags,
		sequenceCount,
		sequenceLength,
		strideNextSequence,
		strideNextItem,
		newCount);

	cudaDeallocArray(scannedFlags, FL);
}

void eatDelay();

#endif /* UTILS_GPU_H_ */