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

// read parameters from command line (or use default values)
PARAMS read_params(int argc, const char **argv)
{
#ifdef VERBOSE
	printf("reading parameters...\n");
#endif
	PARAMS p;
	if (argc == 1 || PARAM_PRESENT("HELP")) { display_help(); exit(1); }
	
	p.trials = GET_PARAM("TRIALS", 1024);
	p.agent_group_size = GET_PARAM("AGENT_GROUP_SIZE", 32);
	p.block_sharing = (p.agent_group_size >= 2);
	p.agents = p.trials * p.agent_group_size;
	p.time_steps = GET_PARAM("TIME_STEPS", 64);
	p.sharing_interval = GET_PARAM("SHARING_INTERVAL", 4);
	if (p.agent_group_size > 1 && 0 != p.time_steps % p.sharing_interval){
		printf("Inconsistent arguments: TIME_STEPS=%d, SHARING_INTERVAL=%d\n", 
			   p.time_steps, p.sharing_interval);
		exit(1);
	}
	p.num_sharing_intervals = p.time_steps / p.sharing_interval;
	p.data_lines = GET_PARAM("DATA_LINES", 16);
	if (0 != p.time_steps % p.data_lines){
		printf("Inconsistent arguments: TIME_STEPS=%d, DATA_LINES=%d\n", 
			   p.time_steps, p.data_lines);
		exit(1);
	}
	p.data_interval = p.time_steps / p.data_lines;
	p.epsilon = GET_PARAMF("EPSILON", DEFAULT_EPSILON);
	p.gamma = GET_PARAMF("GAMMA", DEFAULT_GAMMA);
	p.lambda = GET_PARAMF("LAMBDA", DEFAULT_LAMBDA);
	p.alpha = GET_PARAMF("ALPHA", DEFAULT_ALPHA);
	
	if (0 != p.time_steps % BLOCK_SIZE){
		printf("Inconsistent argument: TIME_STEPS=%d, not a multiple of BLOCKSIZE which is %d\n", 
			   p.time_steps, BLOCK_SIZE);
		exit(1);
	}
	p.blocks = p.time_steps / BLOCK_SIZE;
	p.run_on_CPU = GET_PARAM("RUN_ON_CPU", 1);
	p.run_on_GPU = GET_PARAM("RUN_ON_GPU", 1);
	p.no_print = PARAM_PRESENT("NO_PRINT");
	
	p.state_size = GET_PARAM("STATE_SIZE", 4);
	p.num_actions = NUM_ACTIONS;
	p.num_features = NUM_FEATURES;
	
	p.test_interval = GET_PARAM("TEST_INTERVAL", p.time_steps);
	p.test_reps = GET_PARAM("TEST_REPS", 10000);
	p.num_tests = p.time_steps / p.test_interval;
	
	printf("[POLE][TRIALS%7d][TIME_STEPS%7d][SHARING_INTERVAL%7d][AGENT_GROUP_SIZE%7d][ALPHA%7.4f]"
		   "[EPSILON%7.4f][GAMMA%7.4f][LAMBDA%7.4f][DATA_LINES%7d][STATE_SIZE%7d][TEST_INTERVAL%7d]"
		   "[TEST_REPS%7d]\n", 
		   p.trials, p.time_steps, p.sharing_interval, p.agent_group_size, p.alpha, p.epsilon, 
		   p.gamma, p.lambda, p.data_lines, p.state_size, p.test_interval, p.test_reps);
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
#ifdef VERBOSE
	printf("Initializing agents...CPU...\n");
#endif
	AGENT_DATA *agCPU = initialize_agentsCPU();
#ifdef DUMP_INITIAL_AGENTS
	dump_agents("Initial CPU Agents", agCPU);
#endif
	AGENT_DATA *agGPU = NULL;
	if (p.run_on_GPU) {
#ifdef VERBOSE
		printf("Initializing agents...GPU...\n");
#endif
		agGPU = (p.run_on_GPU) ? initialize_agentsGPU(agCPU) : NULL;
	}

	// run on CPU & GPU
	RESULTS *rCPU = NULL;
	RESULTS *rGPU = NULL;
	if (p.run_on_CPU){
#ifdef VERBOSE
		printf("[CPU]\n");
#endif
		rCPU = initialize_results();
		run_CPU(agCPU, rCPU);
		if (!p.no_print) display_results("CPU:", rCPU);
	}
	
	if (p.run_on_GPU) {
#ifdef VERBOSE
		printf("[GPU]\n");
#endif
		rGPU = initialize_results();
		run_GPU(agGPU, rGPU);
		if (!p.no_print) display_results("GPU:", rGPU);
	}
	
	
	// Clean-up
	free_agentsCPU(agCPU);
	free_agentsGPU(agGPU);
	
	free_results(rGPU);
	free_results(rCPU);

	return 0;
}


