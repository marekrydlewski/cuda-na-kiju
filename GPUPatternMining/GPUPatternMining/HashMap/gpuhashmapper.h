#pragma once

/*
* gpuhashmapper.h
*
*  Created on: 27-02-2013
*      Author: witek
*/

#ifndef GPUHASHMAPPER_H_
#define GPUHASHMAPPER_H_

#include "gpuutils.h"
#include <stdlib.h>

//#define HASHMAPPER_DEBUG_OUTPUT

template <class K> class GPUKeyProcessor
{
public:
	__device__ __inline__ unsigned int reduceKey(K&);
	__device__ __inline__ bool compareKeys(K&, K&);
	__device__ __inline__ bool isKeyEmpty(K&);
	__device__ __inline__ K emptyKey();
	__device__ __inline__ K inSequence(K& start, unsigned int pos, unsigned int step);
};

class GPUUIntKeyProcessor :public GPUKeyProcessor<unsigned int>
{
public:
	__device__ __inline__ unsigned int reduceKey(unsigned int key)
	{
		return key;
	}
	__device__ __inline__ bool compareKeys(unsigned int key1, unsigned int key2)
	{
		return key1 == key2;
	}
	__device__ __inline__ bool isKeyEmpty(unsigned int key)
	{
		return key == emptyKey();
	}
	__device__ __inline__ unsigned int emptyKey()
	{
		return 0xffffffff;
	}
	__device__ __inline__ unsigned int inSequence(unsigned int  start, unsigned int pos, unsigned int step)
	{
		return start + pos*step;
	}
};

class GPUULongIntKeyProcessor :public GPUKeyProcessor<unsigned long long int>
{
public:
	__device__ __inline__ unsigned int reduceKey(unsigned long long int key)
	{
		return key + (key >> 32);
	}
	__device__ __inline__ bool compareKeys(unsigned long long int key1, unsigned long long int key2)
	{
		return key1 == key2;
	}
	__device__ __inline__ bool isKeyEmpty(unsigned long long int key)
	{
		return key == emptyKey();
	}
	__device__ __inline__ unsigned long long int emptyKey()
	{
		return 0xffffffffffffffffL;
	}
	__device__ __inline__ unsigned long long int inSequence(unsigned long long  start, unsigned int pos, unsigned int step)
	{
		return start + pos*step;
	}
};

template <class K, class V, class KP> class GPUHashMapper;

template <class K, class V, class KP> class HashMapperBean
{
public:
	K *d_hashTable;
	V *d_values;
	unsigned int hashSize;
	KP kp;
	HashMapperBean(K *d_hashTable, V *d_values, unsigned int hashSize, KP kp);
};

template <class K, class V, class KP> HashMapperBean<K, V, KP>::HashMapperBean(
	K *d_hashTable, V *d_values, unsigned int hashSize, KP kp)
{
	this->d_hashTable = d_hashTable;
	this->d_values = d_values;
	this->hashSize = hashSize;
	this->kp = kp;
}


namespace GPUHashMapperProcedures
{
	template <class K, class V, class KP> __device__ __inline__ bool getValue(
		HashMapperBean<K, V, KP> &bean, K key, V &value);
	template <class K, class V, class KP> __device__ __inline__ V* getValuePointer(
		HashMapperBean<K, V, KP> &bean, K key);
	template <class K, class V, class KP> __device__ __inline__ void insertKeyValuePair(
		HashMapperBean<K, V, KP> &bean, K key, V value);
	template <class K, class V, class KP> __device__ __inline__ void insertKey(
		HashMapperBean<K, V, KP> &bean, K key);
	template <class K, class V, class KP> __device__ __inline__ bool containsKey(HashMapperBean<K, V, KP> &bean, K key);

	template <class K, class V, class KP> __global__ void getValuesKernel(
		HashMapperBean<K, V, KP> bean, K* keys, V* values, unsigned int count);
	template <class K, class V, class KP> __global__ void getValuesForKeySequenceKernel(
		HashMapperBean<K, V, KP> bean, K key, unsigned int step, V* values, unsigned int count);
	template <class K, class V, class KP> __global__ void getValuePointersKernel(
		HashMapperBean<K, V, KP> bean, K* keys, V** d_valuesPointers, unsigned int count);
	template <class K, class V, class KP> __global__ void getValuePointersForKeySequenceKernel(
		HashMapperBean<K, V, KP> bean, K key, unsigned int step, V** d_valuesPointers, unsigned int count);
	template <class K, class V, class KP> __global__ void insertKeyValuePairsKernel(
		HashMapperBean<K, V, KP> bean, K* keys, V* values, unsigned int count);
	template <class K, class V, class KP> __global__ void insertValuesForKeySequenceKernel(
		HashMapperBean<K, V, KP> bean, K key, unsigned int keyStep, V values, unsigned int valueStep, unsigned int count);
	template <class K, class V, class KP> __global__ void insertKeysKernel(
		HashMapperBean<K, V, KP> bean, K* keys, unsigned int count);
	template <class K, class V, class KP> __global__ void insertKeySequenceKernel(
		HashMapperBean<K, V, KP> bean, K key, unsigned int step, unsigned int count);

	template <class K, class V, class KP> __global__ void cleanKernel(HashMapperBean<K, V, KP> bean);


	__device__ __inline__ unsigned int hashFunction(unsigned int val, unsigned int hashSize);
	__device__ __inline__ unsigned int jumpFunction(unsigned int pos, unsigned int hashSize);
}




template <class K, class V, class KP> class GPUHashMapper
{
private:
	K *d_hashTable;
	V *d_values;
	unsigned int hashSize;
	KP *kp;

public:
	GPUHashMapper(unsigned int hashSize, KP *kp);
	~GPUHashMapper();

	HashMapperBean<K, V, KP> getBean();
	void setKeyProcessor(KP *kp);
	void insertKeyValuePairs(K* keys, V* values, unsigned int count);
	void insertKeys(K* keys, unsigned int count);
	void insertKeySequence(K key, unsigned int step, unsigned int count);
	void insertValuesForKeySequence(K key, unsigned int keyStep, V values, unsigned int valueStep, unsigned int count);
	void getValues(K* keys, V* values, unsigned int count);
	void getValuesForKeySequence(K key, unsigned int step, V* values, unsigned int count);
	void getValuePointers(K* keys, V** valuePointers, unsigned int count);
	void getValuePointersForKeySequence(K key, unsigned int step, V** valuePointers, unsigned int count);
	void clean();

};

//GPUHashMapperProcedures namespace

template <class K, class V, class KP> __device__ __inline__ bool GPUHashMapperProcedures::getValue(
	HashMapperBean<K, V, KP> &bean, K key, V &value)
{
	int hashIndex = hashFunction(bean.kp.reduceKey(key), bean.hashSize);
	bool cont = false;
	bool result;

	do
	{
		//WARNING K must be either 4B or 8B
		K whatsInside = bean.d_hashTable[hashIndex];

		if (bean.kp.isKeyEmpty(whatsInside))
		{
			//not found;
			cont = false;
			result = false;
		}
		else
		{
			//there is something here, there are two possibilities:
			if (!bean.kp.compareKeys(whatsInside, key))
			{
				//Collision - search key and the key in the hashTable are different, we need to try and find other positions.
				hashIndex = jumpFunction(hashIndex, bean.hashSize); //Jump to another position
				cont = true;
			}
			else
			{
				//Found result;
				value = bean.d_values[hashIndex];
				cont = false;
				result = true;
			}
		}
	} while (cont);
	return result;
}

template <class K, class V, class KP> __device__ __inline__ V* GPUHashMapperProcedures::getValuePointer(
	HashMapperBean<K, V, KP> &bean, K key)
{

	int hashIndex = hashFunction(bean.kp.reduceKey(key), bean.hashSize);
	bool cont = false;
	V* result;

	do
	{
		//WARNING K must be either 4B or 8B
		K whatsInside = bean.d_hashTable[hashIndex];

		if (bean.kp.isKeyEmpty(whatsInside))
		{
			//not found;
			cont = false;
			result = NULL;
		}
		else
		{
			//there is something here, there are two possibilities:
			if (!bean.kp.compareKeys(whatsInside, key))
			{
				//Collision - search key and the key in the hashTable are different, we need to try and find other positions.
				hashIndex = jumpFunction(hashIndex, bean.hashSize); //Jump to another position
				cont = true;
			}
			else
			{
				//Found result;
				cont = false;
				result = bean.d_values + hashIndex;
			}
		}
	} while (cont);
	return result;
}

template <class K, class V, class KP> __device__ __inline__ void GPUHashMapperProcedures::insertKeyValuePair(
	HashMapperBean<K, V, KP> &bean, K key, V value)
{

	unsigned int reducedKey = bean.kp.reduceKey(key);
	int hashIndex = hashFunction(reducedKey, bean.hashSize);
	bool cont = false;


	//try to insert into hashTable
	do
	{
		K whatsInside;

		//WARNING K must be either 4B or 8B
		if (sizeof(K) == 4)
		{
			whatsInside = (K)atomicCAS((unsigned int*)(bean.d_hashTable + hashIndex), (unsigned int)bean.kp.emptyKey(), (unsigned int)key);
		}
		else
		{
			whatsInside = (K)atomicCAS((unsigned long long int*)(bean.d_hashTable + hashIndex), (unsigned long long int)bean.kp.emptyKey(), (unsigned long long int)key);
		}

		if (bean.kp.isKeyEmpty(whatsInside))
		{
			//managed to insert into HashTable. If there is some additional data associated with the key insert it into additional tables at propoer index
			bean.d_values[hashIndex] = value;
			cont = false;
		}
		else
		{
			//didnt manage to insert into HashTable. There are now two possibilities...

			if (!bean.kp.compareKeys(whatsInside, key))
			{
				//Collision - inserted key and the key in the hashTable are different, we need to try and find other positions.
				hashIndex = jumpFunction(hashIndex, bean.hashSize); //Jump to another position
				cont = true;
			}
			else
			{
				//Found correct key
				cont = false;
				bean.d_values[hashIndex] = value;
			}
		}
	} while (cont);
}

template <class K, class V, class KP> __device__ __inline__ void GPUHashMapperProcedures::insertKey(
	HashMapperBean<K, V, KP> &bean, K key)
{

	int hashIndex = hashFunction(bean.kp.reduceKey(key), bean.hashSize);
	bool cont = false;

	//try to insert into hashTable
	do
	{
		K whatsInside;

		//WARNING K must be either 4B or 8B
		if (sizeof(K) == 4)
		{
			whatsInside = (K)atomicCAS((unsigned int*)(bean.d_hashTable + hashIndex), (unsigned int)bean.kp.emptyKey(), (unsigned int)key);
		}
		else
		{
			whatsInside = (K)atomicCAS((unsigned long long int*)(bean.d_hashTable + hashIndex), (unsigned long long int)bean.kp.emptyKey(), (unsigned long long int)key);
		}

		if (bean.kp.isKeyEmpty(whatsInside))
		{
			//managed to insert into HashTable. If there is some additional data associated with the key insert it into additional tables at propoer index
			cont = false;
		}
		else
		{
			//didnt manage to insert into HashTable. There are now two possibilities...

			if (!bean.kp.compareKeys(whatsInside, key))
			{
				//Collision - inserted key and the key in the hashTable are different, we need to try and find other positions.
				hashIndex = jumpFunction(hashIndex, bean.hashSize); //Jump to another position
				cont = true;
			}
			else
			{
				//Found correct key
				cont = false;
			}
		}
	} while (cont);
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::getValuesKernel(
	HashMapperBean<K, V, KP> bean, K* keys, V* values, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		GPUHashMapperProcedures::getValue(bean, keys[pos], values[pos]);
	}
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::getValuesForKeySequenceKernel(
	HashMapperBean<K, V, KP> bean, K key, unsigned int step, V* values, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		GPUHashMapperProcedures::getValue(bean, bean.kp.inSequence(key, pos, step), values[pos]);
	}
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::getValuePointersKernel(
	HashMapperBean<K, V, KP> bean, K* keys, V** d_valuesPointers, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		d_valuesPointers[pos] = GPUHashMapperProcedures::getValuePointer(bean, keys[pos]);
	}
}

template <class K, class V, class KP> __global__ void getValuePointersForKeySequenceKernel(
	HashMapperBean<K, V, KP> bean, K key, unsigned int step, V** d_valuesPointers, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		d_valuesPointers[pos] = GPUHashMapperProcedures::getValuePointer(bean, bean.kp.inSequence(key, pos, step));
	}
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::insertKeyValuePairsKernel(
	HashMapperBean<K, V, KP> bean, K* keys, V* values, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		GPUHashMapperProcedures::insertKeyValuePair(bean, keys[pos], values[pos]);
	}
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::insertKeysKernel(
	HashMapperBean<K, V, KP> bean, K* keys, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		GPUHashMapperProcedures::insertKey(bean, keys[pos]);
	}
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::insertKeySequenceKernel(
	HashMapperBean<K, V, KP> bean, K key, unsigned int step, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		GPUHashMapperProcedures::insertKey(bean, bean.kp.inSequence(key, pos, step));
	}
}


template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::insertValuesForKeySequenceKernel(
	HashMapperBean<K, V, KP> bean, K key, unsigned int keyStep, V values, unsigned int valueStep, unsigned int count)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<count)
	{
		GPUHashMapperProcedures::insertKeyValuePair(bean, bean.kp.inSequence(key, pos, keyStep), values + pos*valueStep);
	}
}

template <class K, class V, class KP> __global__ void GPUHashMapperProcedures::cleanKernel(
	HashMapperBean<K, V, KP> bean)
{
	unsigned int pos = computeLinearAddressFrom2D();

	if (pos<bean.hashSize) bean.d_hashTable[pos] = bean.kp.emptyKey();
}

__device__ __inline__ unsigned int GPUHashMapperProcedures::hashFunction(unsigned int val, unsigned int hashSize)
{
	return ((((val * 4321) + 3052) & 0x7fffffff) % 2147483647) % hashSize;
	//return val%hashSize;
	//return val-(val/hashSize)*hashSize;
}

__device__ __inline__ unsigned int GPUHashMapperProcedures::jumpFunction(unsigned int pos, unsigned int hashSize)
{
	return (pos + 1) % hashSize;
}


//GPUHashMapper class

template <class K, class V, class KP> HashMapperBean<K, V, KP> GPUHashMapper<K, V, KP>::getBean()
{
	return HashMapperBean<K, V, KP>(d_hashTable, d_values, hashSize, *kp);
}


template <class K, class V, class KP> GPUHashMapper<K, V, KP>::GPUHashMapper(unsigned int hashSize, KP *kp)
{
	this->hashSize = hashSize;


#ifdef HASHMAPPER_DEBUG_OUTPUT
	printf("Allocating hashmap keytable\n");
#endif
	d_hashTable = cudaAllocArray<K>(hashSize, FL);
#ifdef HASHMAPPER_DEBUG_OUTPUT
	printf("Allocated a hash map for hashsize %d, keysize %d, value size %d at %.8x\n", hashSize, sizeof(K), sizeof(V), d_hashTable);
	printf("Allocating hashmap valuetable\n");
#endif
	d_values = cudaAllocArray<V>(hashSize, FL);
#ifdef HASHMAPPER_DEBUG_OUTPUT
	printf("Allocated hashmap valuetable\n");
#endif


#ifdef HASHMAPPER_DEBUG_OUTPUT
	printf("Before setting of key processor \n");
#endif
	setKeyProcessor(kp);

#ifdef HASHMAPPER_DEBUG_OUTPUT
	printf("Before cleaning of keytable\n");
#endif
	clean();

#ifdef HASHMAPPER_DEBUG_OUTPUT
	printf("After cleaning of keytable\n");
#endif
}

template <class K, class V, class KP> GPUHashMapper<K, V, KP>::~GPUHashMapper()
{
	cudaDeallocArray(d_hashTable, FL);
	cudaDeallocArray(d_values, FL);
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::setKeyProcessor(KP *kp)
{
	this->kp = kp;
}


template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::getValues(K* keys, V* values, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::getValuesKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), keys, values, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::getValuesForKeySequence(K key, unsigned int step, V* values, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::getValuesForKeySequenceKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), key, step, values, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}


template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::getValuePointers(K* keys, V** valuePointers, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::getValuePointersKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), keys, valuePointers, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::getValuePointersForKeySequence(K key, unsigned int step, V** valuePointers, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::getValuePointersForKeySequenceKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), key, step, valuePointers, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::insertKeyValuePairs(K* keys, V* values, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::insertKeyValuePairsKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), keys, values, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::insertValuesForKeySequence(K key, unsigned int keyStep, V values, unsigned int valueStep, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::insertValuesForKeySequenceKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), key, keyStep, values, valueStep, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::insertKeys(K* keys, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::insertKeysKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), keys, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::insertKeySequence(K key, unsigned int step, unsigned int count)
{
	dim3 gridSize;

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);
	gridSize.z = 1;

	GPUHashMapperProcedures::insertKeySequenceKernel<K, V, KP> << <gridSize, 512 >> >(getBean(), key, step, count);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
}

template <class K, class V, class KP> void GPUHashMapper<K, V, KP>::clean()
{
	dim3 gridSize(1, 1, 1);

	findSmallest2D(hashSize, 512, gridSize.x, gridSize.y);

	GPUHashMapperProcedures::cleanKernel<K, V, KP> << <gridSize, 512 >> >(getBean());
	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());

}

template <class K, class V, class KP> __device__ __inline__ bool GPUHashMapperProcedures::containsKey(
	HashMapperBean<K, V, KP> &bean, K key)
{
	int hashIndex = hashFunction(bean.kp.reduceKey(key), bean.hashSize);
	bool cont = false;
	bool result;

	do
	{
		//WARNING K must be either 4B or 8B
		K whatsInside = bean.d_hashTable[hashIndex];

		if (bean.kp.isKeyEmpty(whatsInside))
		{
			//not found;
			cont = false;
			result = false;
		}
		else
		{
			//there is something here, there are two possibilities:
			if (!bean.kp.compareKeys(whatsInside, key))
			{
				//Collision - search key and the key in the hashTable are different, we need to try and find other positions.
				hashIndex = jumpFunction(hashIndex, bean.hashSize); //Jump to another position
				cont = true;
			}
			else
			{
				//Found result;
				cont = false;
				result = true;
			}
		}
	} while (cont);
	return result;
}





#endif /* GPUHASHMAPPER_H_ */
