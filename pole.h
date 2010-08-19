//
//  pole.h
//  pole
//
//  Created by Dwight Bell on 8/18/10.
//  Copyright dbelll 2010. All rights reserved.
//

typedef struct {
	unsigned trials;
	unsigned agent_group_size;
	unsigned time_steps;
	unsigned sharing_interval;		// SHARING_INTERVAL command-line parameter
	unsigned block_sharing;
	unsigned data_lines;			// DATA_LINES command-line parameter
	unsigned agents;
	unsigned num_sharing_intervals;
	unsigned data_interval;
	float epsilon;				// exploration factor
	float gamma;				// discount factor
	float lambda;				// eligibility trace decay factor
	unsigned blocks;			// number of blocks with BLOCK_SIZE agents in each block
	unsigned run_on_CPU;
	unsigned run_on_GPU;
	unsigned no_print;
	unsigned num_features;
	unsigned num_actions;
	unsigned state_size;		// a state is this number of floats.
} PARAMS;

typedef struct{
	unsigned device_flag;	// 1 => these are device pointers, 0 => host pointers
	unsigned *seeds;	// seeds for random number generator
	float *theta;		// weights for each of the features
	float *e;			// eligibility trace
	float *ep_data;		// state, action, result, state, action values for this action episode
	float *s;			// current state
	float *Q;			// Q values for each action
} AGENT_DATA;		// may hold either host or device pointers

typedef struct{
	float *begun;		// cumulative number of episodes begun
	float *ended;		// cumulative number of episodes ended
	float *total_length;	// total length for all the ended episodes
} RESULTS;

PARAMS read_params(int argc, const char **argv);
AGENT_DATA *initialize_agentsCPU();
AGENT_DATA *initialize_agentsGPU(AGENT_DATA *agCPU);
void free_agentsCPU(AGENT_DATA *agCPU);
void free_agentsGPU(AGENT_DATA *agGPU);
RESULTS *initialize_results();
void free_results(RESULTS *r);
void run_GPU(AGENT_DATA *cv, RESULTS *r);
void run_CPU(AGENT_DATA *cv, RESULTS *r);
void display_results(const char *str, RESULTS *r);
