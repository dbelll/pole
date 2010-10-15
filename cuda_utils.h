/*
 *  cuda_utils.h
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

/*
 *	Utilities for...
 *		Reading command line parameters
 *			GET_PARM(<param>, <default_value>)	// get an integer parameter
 *			GET_PARMF(<param>, <default_value>)	// get a float parameter
 *			PARAM_PRESENT(<param>)				// check for existance of a parameter
 *
 *		
 *
 *		Allocating and transferring data to/from device
 *			device_copyf(<h_data>, <count>)		// copy an array to the device
 *			device_copyui(<h_data>, <count>)	
 *												
 *			device_allocf(<count>)				// allocate device memory for <count> values
 *			device_allocui(<count>)				
 *
 *			host_copyf(<d_data>, <count>)		// allocate host memory and copy from device
 *			host_copyui(<d_data>, <count>)
 *
 *		Timers...
 *			CREAT_TIMER(&timer)
 *			START_TIMER(timer)
 *			STOP_TIMER(timer, "message")
 *			DELETE_TIMER(timer)
 */

#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

// define this symbol to print a message at all device_copyx calls
//#define TRACE_DEVICE_ALLOCATIONS

static int __iTemp;			// used in GET_PARAM macros
static float __fTemp;		// used in GET_PARAM macros

// macros to read command line arguments or use a default value
// This macro assums argc and argv are their normal values found in main()
#define GET_PARAM(str, default) (CUTTrue == cutGetCmdLineArgumenti(argc, argv, (str), &__iTemp)) ? __iTemp : (default)
#define GET_PARAMF(str, default) (CUTTrue == cutGetCmdLineArgumentf(argc, argv, (str), &__fTemp)) ? __fTemp : (default)
#define PARAM_PRESENT(str) (CUTTrue == cutCheckCmdLineFlag(argc, argv, (str)))

// allocate room on the device and copy data from host, returning the device pointer
// returned pointer must be ultimately freed on the device
float *device_copyf(float *data, unsigned count);
unsigned *device_copyui(unsigned *data, unsigned count);

// allocate room on the device, returning the device pointer
// returned pointer must be ultimately freed on the device
float *device_allocf(unsigned count);
unsigned *device_allocui(unsigned count);

// allocate room on the host and copy data from device, returning the host pointer
// returned pointer must be ultimately freed on the host
float *host_copyf(float *d_data, unsigned count);
unsigned *host_copyui(unsigned *d_data, unsigned count);

// Macros for calculating timing values.
// Caller must supply a pointer to unsigned int when creating a timer,
// and the unsigned int for other timer calls.
void CREATE_TIMER(unsigned int *p_timer);
void START_TIMER(unsigned int timer);
float STOP_TIMER(unsigned int timer, char *message);
void DELETE_TIMER(unsigned int timer);
void PAUSE_TIMER(unsigned int timer);
void RESUME_TIMER(unsigned int timer);
void RESET_TIMER(unsigned timer);
void DELETE_TIMER(unsigned int timer);
void PRINT_TIME(float time, char *message);


/*
 *		Use Cuda Events to get precise GPU timings.
 *		These timings will be consistent with the profiler timing values.
 *
 *		First, declare a float variabile to hold the elapsed time
 *		Use CUDA_EVENT_PREPARE before doing any timing to setup variables and create the events
 *		Use CUDA_EVENT_START before launching the kernel.
 *		Use CUDA_EVENT_STOP(t) after launching the kernel, where t is the float used to accumulate time
 *		Time values can be printed in a consistent format by calling PRINT_TIME(t, "timing message");
 *		When all timing is done, use CUDA_EVENT_CLEANUP once all event timing is done.
 */

#define CUDA_EVENT_PREPARE	cudaEvent_t __start, __stop;	\
							float __timeTemp = 0.0f;		\
							cudaEventCreate(&__start);		\
							cudaEventCreate(&__stop);

#define CUDA_EVENT_START	cudaEventRecord(__start, 0);
#define CUDA_EVENT_STOP(t)	cudaEventRecord(__stop, 0);							\
							cudaEventSynchronize(__stop);							\
							cudaEventElapsedTime(&__timeTemp, __start, __stop);	\
							t += __timeTemp;

#define CUDA_EVENT_CLEANUP	cudaEventDestroy(__start);	\
							cudaEventDestroy(__stop);

#endif
