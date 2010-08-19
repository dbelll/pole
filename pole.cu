//
//  pole.cu
//  pole
//
//  Created by Dwight Bell on 8/18/10.
//  Copyright dbelll 2010. All rights reserved.
//

#include <cuda.h>
#include "cutil.h"

#include "cuda_rand.cu"

#include "pole.h"
#include "cuda_utils.h"

#define BLOCK_SIZE 256

static int __iTemp;
static float __fTemp;
PARAMS _p;

// private prototypes
void display_help();

AGENT_DATA *initialize_agentsCPU()
{
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	ag->seeds = (unsigned *)malloc(_p.agents * 4 * sizeof(unsigned));
	ag->theta = (float *)malloc(_p.agents * _p.num_features * sizeof(float));
	ag->e = (float *)malloc(_p.agents * _p.num_features * sizeof(float));
	unsigned rows = _p.agents * ((_p.state_size + 2) * _p.sharing_interval + _p.state_size + 1);
	ag->ep_data = (float *)malloc(rows * sizeof(float));
	ag->s = (float *)malloc(_p.agents * sizeof(float));
	ag->Q = (float *)malloc(_p.agents * _p.num_actions * sizeof(float));
	return ag;
}

void free_agentsCPU(AGENT_DATA *ag)
{
	if (ag) {
		if (ag->seeds) free(ag->seeds);
		if (ag->theta) free(ag->theta);
		if (ag->e) free(ag->e);
		if (ag->ep_data) free(ag->ep_data);
		if (ag->s) free(ag->s);
		if (ag->Q) free(ag->Q);
		free(ag);
	}
}

void free_agentsGPU(AGENT_DATA *ag)
{
	if (ag) {
		if (ag->seeds) cudaFree(ag->seeds);
		if (ag->theta) cudaFree(ag->theta);
		if (ag->e) cudaFree(ag->e);
		if (ag->ep_data) cudaFree(ag->ep_data);
		if (ag->s) cudaFree(ag->s);
		if (ag->Q) cudaFree(ag->Q);
		free(ag);
	}
}

AGENT_DATA *initialize_agentsGPU(AGENT_DATA *agCPU)
{
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	
	return ag;	
}

RESULTS *initialize_results()
{
	RESULTS *r = (RESULTS *)malloc(sizeof(RESULTS));
	r->begun = (float *)malloc(_p.agents * _p.data_lines * sizeof(float));
	r->ended = (float *)malloc(_p.agents * _p.data_lines * sizeof(float));
	r->total_length = (float *)malloc(_p.agents * _p.data_lines * sizeof(float));
	return r;
}

void free_results(RESULTS *r)
{
	if (r) {
		if (r->begun) free(r->begun);
		if (r->ended) free(r->ended);
		if (r->total_length) free(r->total_length);
		free(r);
	}
}

void display_results(const char *str, RESULTS *r)
{
	printf("%s \n", str);
	
}

PARAMS read_params(int argc, const char **argv)
{
	if (argc == 1 || PARAM_PRESENT("HELP")) { display_help(); exit(1); }
	
	_p.trials = GET_PARAM("TRIALS", 1024);
	_p.agent_group_size = GET_PARAM("AGENT_GROUP_SIZE", 32);
	_p.block_sharing = (_p.agent_group_size >= 2);
	_p.agents = _p.trials * _p.agent_group_size;
	_p.time_steps = GET_PARAM("TIME_STEPS", 64);
	_p.sharing_interval = GET_PARAM("SHARING_INTERVAL", 4);
	if (0 != _p.time_steps % _p.sharing_interval){
		printf("Inconsistent arguments: TIME_STEPS=%d, SHARING_INTERVAL=%d\n", 
				_p.time_steps, _p.sharing_interval);
		exit(1);
	}
	_p.num_sharing_intervals = _p.time_steps / _p.sharing_interval;
	_p.data_lines = GET_PARAM("DATA_LINES", 16);
	if (0 != _p.time_steps % _p.data_lines){
		printf("Inconsistent arguments: TIME_STEPS=%d, DATA_LINES=%d\n", 
				_p.time_steps, _p.data_lines);
		exit(1);
	}
	_p.data_interval = _p.time_steps / _p.data_lines;
	_p.epsilon = GET_PARAMF("EPSILON", .10f);
	_p.gamma = GET_PARAMF("GAMMA", .90f);
	_p.lambda = GET_PARAMF("LAMBDA", .90f);
	
	if (0 != _p.time_steps % BLOCK_SIZE){
		printf("Inconsistent argument: TIME_STEPS=%d, not a multiple of BLOCKSIZE which is %d\n", 
				_p.time_steps, BLOCK_SIZE);
		exit(1);
	}
	_p.blocks = _p.time_steps / BLOCK_SIZE;
	_p.run_on_CPU = GET_PARAM("RUN_ON_CPU", 1);
	_p.run_on_GPU = GET_PARAM("RUN_ON_GPU", 1);
	_p.no_print = PARAM_PRESENT("NO_PRINT");
	
	_p.state_size = GET_PARAM("STATE_SIZE", 4);

	printf("[POLE][TRIALS%7d][TIME_STEPS%7d][SHARING_INTERVAL%7d][AGENT_GROUP_SIZE%7d]""[EPSILON%7.4f][DATA_LINES%7d][STATE_SIZE%7d]\n", 
			_p.trials, _p.time_steps, _p.sharing_interval, _p.agent_group_size, _p.epsilon, 
			_p.data_lines, _p.state_size);
	return _p;
}

void display_help()
{
	printf("bandit parameters:\n");
	printf("  --TRIALS              number of trials for averaging reults\n");
	printf("  --AGENT_GROUP_SIZE    size of agent groups that will communicate\n");
	printf("  --TIME_STEPS          total number of time steps for each trial\n");
	printf("  --SHARING_INTERVAL    number of time steps between agent communication\n");
	printf("  --DATA_LINES          number of data samples in the report\n");
	printf("  --EPSILON             float value for epsilon\n");
	printf("  --GAMMA               float value for gamma, the discount factor\n");
	printf("  --LAMBDA              f loat value for lambda, the trace decay factor\n");
	printf("  --RUN_ON_GPU          1 = run on GPU, 0 = do not run on GPU\n");
	printf("  --RUN_ON_CPU          1 = run on CPU, 0 = do not run on CPU\n");
	printf("  --HELP                print this help message\n");
	printf("default values will be used for any parameters not on command line\n");
}

#pragma mark CPU & GPU

// take an action from the current state, s, returning the reward and saving the new state in s_prime
__device__ __host__ float take_action(unsigned a, float *s, float *s_prime, unsigned stride)
{
	return NULL;
}

// determine the best action for the current state, using weights in theta, storing the estimated
// Q value for that action in bestQ
__device__ __host__ float best_action(float *s, float *theta, float *Q, unsigned stride)
{
	return NULL;
}

__device__ __host__ unsigned choose_action(float *s, float *theta, float epsilon, unsigned stride, 
											float *bestQ)
{
	return NULL;
}

#pragma mark CPU

void run_CPU(AGENT_DATA *cv, RESULTS *r)
{
	
}

#pragma mark GPU
__global__ void kernel_operation(int n, float *x)
{
}

void run_GPU(AGENT_DATA *cv, RESULTS *r)
{
	
}

void gpu_operation(int n, float *x)
{
	unsigned int timer;
	CREATE_TIMER(&timer);

	// copy data to device
	START_TIMER(timer);
	float *d_x = NULL;
	int size = n * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
	cudaThreadSynchronize();
	STOP_TIMER(timer, "copy data to device");


	// calculate size of block and grid
	int tot_threads = 32 * (1 + (n-1)/32);	// smallest multiple of 32 >= n
	dim3 blockDim(min(512,tot_threads));
	dim3 gridDim(1 + (tot_threads -1)/512, 1);
	if (gridDim.x > 65535) {
		gridDim.y = 1 + (gridDim.x-1) / 65535;
		gridDim.x = 1 + (gridDim.x-1) / gridDim.y;
	}
	
	// Do the operation on the device
	START_TIMER(timer);	
	kernel_operation<<<gridDim, blockDim>>>(n, d_x);
	cudaThreadSynchronize();
	STOP_TIMER(timer, "GPU operations");

	// Check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
	
	
	// copy results back to host
	START_TIMER(timer);
	CUDA_SAFE_CALL(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));
	cudaThreadSynchronize();
	STOP_TIMER(timer, "copy data back to host");
	
	
}