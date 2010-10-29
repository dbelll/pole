/*
 *  cuda_utils.cu
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include "cuda_utils.h"

#pragma mark timer functions

void CREATE_TIMER(unsigned int *p_timer){
	cutilCheckError(cutCreateTimer(p_timer)); 
	cutilCheckError(cutResetTimer(*p_timer));
}
void START_TIMER(unsigned int timer){ 
	cutilCheckError(cutResetTimer(timer));
	cutilCheckError(cutStartTimer(timer)); 
}
float STOP_TIMER(unsigned int timer, char *message){
	cutilCheckError(cutStopTimer(timer));
	float elapsed = cutGetTimerValue(timer);
	if (message) printf("%12.3f ms for %s\n", elapsed, message);
	return elapsed;
}
void PAUSE_TIMER(unsigned int timer){	cutilCheckError(cutStopTimer(timer));	}
void RESUME_TIMER(unsigned int timer){ cutilCheckError(cutStartTimer(timer));	}
void RESET_TIMER(unsigned timer){		cutilCheckError(cutResetTimer(timer));	}
void DELETE_TIMER(unsigned int timer){	cutilCheckError(cutDeleteTimer(timer)); }

void PRINT_TIME(float time, char *message)
{
	if (message) printf("%12.3f ms for %s\n", time, message);
}

#pragma mark device memory functions
float *device_copyf(float *data, unsigned count_data)
{
	float *d_data = NULL;
	unsigned size_data = count_data * sizeof(float);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_copyf] float host data at 0x%p count = %d, ", data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size_data));
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("copied to 0x%p\n", d_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
	return d_data;
}

unsigned *device_copyui(unsigned *data, unsigned count_data)
{
	unsigned *d_data = NULL;
	unsigned size_data = count_data * sizeof(unsigned);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_copyui] unsigned data at 0x%p count = %d\n", data, count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, size_data));
	CUDA_SAFE_CALL(cudaMemcpy(d_data, data, size_data, cudaMemcpyHostToDevice));
	return d_data;
}

float *device_allocf(unsigned count_data)
{
	float *d_data;
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_allocf] count = %d\n", count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, count_data * sizeof(float)));
	return d_data;
}

unsigned *device_allocui(unsigned count_data)
{
	unsigned *d_data;
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[device_allocf] count = %d\n", count_data);
	#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, count_data * sizeof(unsigned)));
	return d_data;
}

float *host_copyf(float *d_data, unsigned count_data)
{
	unsigned size_data = count_data * sizeof(float);
	float *data = (float *)malloc(size_data);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[host_copyf] float data at %p count = %d\n", d_data, size_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(data, d_data, size_data, cudaMemcpyDeviceToHost));
	return data;
}

unsigned *host_copyui(unsigned *d_data, unsigned count_data)
{
	unsigned size_data = count_data * sizeof(unsigned);
	unsigned *data = (unsigned *)malloc(size_data);
	#ifdef TRACE_DEVICE_ALLOCATIONS
		printf("[host_copyui] float data at 0x%p count = %d\n", d_data, size_data);
	#endif
	CUDA_SAFE_CALL(cudaMemcpy(data, d_data, size_data, cudaMemcpyDeviceToHost));
	return data;
}


