//
//  main.c
//  pole
//
//  Created by Dwight Bell on 8/18/10.
//  Copyright dbelll 2010. All rights reserved.
//

/*
 *	Entry point and functions for reading command line and displaying help.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "cuda_utils.h"
#include "./common/inc/cutil.h"
#include "pole.h"
#include "main.h"

// print out information on using this program
void display_help()
{
	printf("bandit parameters:\n");
	printf("  --TRIALS              number of trials for averaging reults\n");
	printf("  --TIME_STEPS          total number of time steps for each trial\n");
	printf("  --AGENT_GROUP_SIZE    size of agent groups that will communicate\n");
	printf("  --SHARING_INTERVAL    number of time steps between agent communication\n");
	printf("  --ALPHA               float value for alpha, the learning rate parameter\n");
	printf("  --EPSILON             float value for epsilon, the exploration parameter\n");
	printf("  --GAMMA               float value for gamma, the discount factor\n");
	printf("  --LAMBDA              float value for lambda, the trace decay factor\n");
	printf("  --DIVS_X              number of divisions of x value");
	printf("  --DIVS_DX             number of divisions of x velocity");
	printf("  --DIVS_ALPHA          number of divisions of alpha value");
	printf("  --DIVS_DALPHA         number of divisions of alpha velocity");
	printf("  --TEST_INTERVAL       time steps between testing of agent's learning ability\n");
	printf("  --TEST_REPS			duration of test in time steps\n");
	printf("  --RESTART_INTERVAL    time steps between random restarts\n");
	printf("  --RUN_ON_GPU          1 = run on GPU, 0 = do not run on GPU\n");
	printf("  --RUN_ON_CPU          1 = run on CPU, 0 = do not run on CPU\n");
	printf("  --NO_PRINT			flag to suppress printing out results (only timing values printed)\n");
	printf("  --HELP                print this help message\n");
	printf("default values will be used for any parameters not on command line\n");
}

// read parameters from command line (or use default values)
PARAMS read_params(int argc, const char **argv)
{
#ifdef VERBOSE
	printf("reading parameters...\n");
#endif
	PARAMS p;
	if (argc == 1 || PARAM_PRESENT("HELP")) { display_help(); exit(1); }
	
	p.trials = GET_PARAM("TRIALS", 1024);
	p.agent_group_size = GET_PARAM("AGENT_GROUP_SIZE", 1);
	p.agents = p.trials * p.agent_group_size;
	p.time_steps = GET_PARAM("TIME_STEPS", 64);
	p.sharing_interval = GET_PARAM("SHARING_INTERVAL", 4);
	
	// Total time steps must be an integer number of sharing intervals
	if (p.agent_group_size > 1 && 0 != (p.time_steps % p.sharing_interval)){
		printf("Inconsistent arguments: TIME_STEPS=%d, SHARING_INTERVAL=%d\n", 
			   p.time_steps, p.sharing_interval);
		exit(1);
	}
	p.num_sharing_intervals = p.time_steps / p.sharing_interval;

	p.epsilon = GET_PARAMF("EPSILON", DEFAULT_EPSILON);
	p.gamma = GET_PARAMF("GAMMA", DEFAULT_GAMMA);
	p.lambda = GET_PARAMF("LAMBDA", DEFAULT_LAMBDA);
	p.alpha = GET_PARAMF("ALPHA", DEFAULT_ALPHA);
	
	p.divs_x = GET_PARAM("DIVS_X", X_DIV);
	p.divs_dx = GET_PARAM("DIVS_DX", X_VEL_DIV);
	p.divs_alpha = GET_PARAM("DIVS_ALPHA", ANGLE_DIV);
	p.divs_dalpha = GET_PARAM("DIVS_DALPHA", ANGLE_VEL_DIV);
	p.num_features = p.divs_x * p.divs_dx * p.divs_alpha * p.divs_dalpha;
	
	p.run_on_CPU = GET_PARAM("RUN_ON_CPU", 1);
	p.run_on_GPU = GET_PARAM("RUN_ON_GPU", 1);
	p.no_print = PARAM_PRESENT("NO_PRINT");
	
	p.state_size = NUM_STATE_VALUES;
	p.num_actions = NUM_ACTIONS;
	
	p.test_interval = GET_PARAM("TEST_INTERVAL", p.time_steps);
	if (p.test_interval > p.time_steps || 0 != (p.time_steps % p.test_interval)) {
		printf("Inconsistent arguments: TIME_STEPS=%d, TEST_INTERVAL=%d\n", p.time_steps, 
																			   p.test_interval);
		exit(1);
	}
	p.test_reps = GET_PARAM("TEST_REPS", p.test_interval);
	p.num_tests = p.time_steps / p.test_interval;
	
	p.restart_interval = GET_PARAM("RESTART_INTERVAL", p.test_interval);
	if (p.restart_interval > p.test_interval || 0 != (p.test_interval % p.restart_interval)) {
		printf("Inconsistent arguments: TEST_INTERVAL=%d, RESTART_INTERVAL=%d\n", p.test_interval, 
			   p.restart_interval);
		exit(1);
	}
	p.num_restarts = p.time_steps / p.restart_interval;
	p.restarts_per_test = p.num_restarts / p.num_tests;
	p.restarts_per_share = p.sharing_interval / p.restart_interval;
	if (p.restarts_per_share == 0) p.restarts_per_share = 1;
	
	printf("[POLE][TRIALS%7d][TIME_STEPS%7d][SHARING_INTERVAL%7d][AGENT_GROUP_SIZE%7d][ALPHA%7.4f]"
		   "[EPSILON%7.4f][GAMMA%7.4f][LAMBDA%7.4f][TEST_INTERVAL%7d][TEST_REPS%7d]"
		   "[RESTART_INTERVAL%7d][DIVS%3d%3d%3d%3d]\n", p.trials, p.time_steps, p.sharing_interval, p.agent_group_size, 
		   p.alpha, p.epsilon, p.gamma, p.lambda, p.test_interval, p.test_reps, p.restart_interval, p.divs_x, 
		   p.divs_dx, p.divs_alpha, p.divs_dalpha);
#ifdef VERBOSE
	printf("num_agents = %d, num_features = %d\n", p.agents, p.num_features);
#endif
	return p;
}

int main(int argc, const char **argv)
{
	PARAMS p = read_params(argc, argv);
	set_params(p);
	
	// Initialize agents on CPU and GPU
	AGENT_DATA *agCPU = initialize_agentsCPU();
	if (p.run_on_GPU) initialize_agentsGPU(agCPU);

	// run on CPU & GPU
	RESULTS *rCPU = NULL;
	RESULTS *rGPU = NULL;
	if (p.run_on_CPU){
		rCPU = initialize_results();
		run_CPU(agCPU, rCPU);
		if (!p.no_print) display_results("CPU:", rCPU);
	}
	
	if (p.run_on_GPU) {
		rGPU = initialize_results();
		run_GPU(rGPU);
		if (!p.no_print) display_results("GPU:", rGPU);
	}
	
	// Clean-up
	free_agentsCPU(agCPU);
	free_agentsGPU();
	
	free_results(rGPU);
	free_results(rCPU);

	return 0;
}


