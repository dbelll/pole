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
	float epsilon;
	unsigned blocks;			// number of blocks with BLOCK_SIZE agents in each block (except the last block may have less)
	unsigned run_on_CPU;
	unsigned run_on_GPU;
	unsigned no_print;
	
	unsigned state_size;		// a state is this number of floats.
} PARAMS;

typedef struct{
	unsigned *seeds;
	float *agent_states;
	float *last_action;
} COMMON_VALS;

typedef struct{
	float *results;
} RESULTS;

void read_params(int argc, const char **argv);

COMMON_VALS *initialize_common_values();
void free_common_values(COMMON_VALS *cv);
void transfer_to_device(COMMON_VALS *cv);
RESULTS *allocate_result_arrays();
void free_result_arrays(RESULTS *r);
void run_GPU(COMMON_VALS *cv, RESULTS *r);
void run_CPU(COMMON_VALS *cv, RESULTS *r);
