#ifndef __POLE_H__
#define __POLE_H__

//
//  pole.h
//  pole
//
//  Created by Dwight Bell on 8/18/10.
//  Copyright dbelll 2010. All rights reserved.
//
#pragma mark -
#pragma mark Problem Constants

#define BLOCK_SIZE 128

#define MAX_TIME_STEPS_PER_LAUNCH 65536


// range of values for the initial random weights, theta
#define RAND_WGT_MIN -1.0f
#define RAND_WGT_MAX 1.0f

#define DIVS 3

// parameters of the problem and tiling of state space
// see Brownlee. The pole balancing problem: a benchmark control theory problem. hdl.handle.net (2005)
#define ANGLE_MAX .209f
#define ANGLE_MIN (-ANGLE_MAX)
#define ANGLE_DIV DIVS

#define ANGLE_VEL_MAX .05f
#define ANGLE_VEL_MIN (-ANGLE_VEL_MAX)
#define ANGLE_VEL_DIV 2

#define X_MAX 2.400f
#define X_MIN (-X_MAX)
#define X_DIV DIVS

#define X_VEL_MAX .5f
#define X_VEL_MIN (-X_VEL_MAX)
#define X_VEL_DIV 2

// number of standard deviations equal to the maximum value for determining initial state values
//#define SD_FOR_MAX 2.0f

#define NUM_FEATURES (ANGLE_DIV * ANGLE_VEL_DIV * X_DIV *X_VEL_DIV)
#define NUM_STATE_VALUES 4
#define NUM_ACTIONS 2

// various constants
#define GRAV -9.81f			// gravitational acceleration in meters per second squared
#define CART_MASS 1.0f		// mass of cart in kilograms
#define POLE_MASS 0.1f		// mass of the pole in kilograms
#define POLE_LENGTH 0.5f	// distance from pole's center of mass to either end
#define FORCE 1.0f			// force applied to the cart from the aciton in newtons
#define TRACK_LENGTH 2.4f	// distance from center of track to either end, in meters
#define TAU .02f			// length of a time step, in seconds

#define REWARD_FAIL -100.0f
#define REWARD_NON_FAIL 0.0f

#define DEFAULT_EPSILON 0.10f
#define DEFAULT_GAMMA 0.90f
#define DEFAULT_LAMBDA 0.90f
#define DEFAULT_ALPHA 0.20f

#pragma mark -
#pragma mark typedefs

typedef struct {
	unsigned trials;
	unsigned time_steps;
	unsigned agent_group_size;

	unsigned sharing_interval;		// SHARING_INTERVAL command-line parameter
	unsigned block_sharing;

	unsigned agents;			// total number of agents = agents per group * number of trials
	unsigned num_sharing_intervals;
	float epsilon;				// exploration factor
	float gamma;				// discount factor
	float lambda;				// eligibility trace decay factor
	float alpha;				// learning rate

//	unsigned blocks;			// number of blocks with BLOCK_SIZE agents in each block
	unsigned run_on_CPU;		// flag indicating to run on CPU
	unsigned run_on_GPU;		// flag indicating to run on GPU
	unsigned no_print;			// flag to suppress print-out
	unsigned num_features;
	unsigned num_actions;		//
	unsigned state_size;		// a state is this number of floats.
	unsigned test_interval;		// number of time steps between testing
	unsigned test_reps;			// number of reps in each test
	unsigned num_tests;			// = time_steps / test_interval
} PARAMS;

typedef struct{
	unsigned device_flag;	// 1 => these are device pointers, 0 => host pointers
	unsigned *seeds;	// seeds for random number generator
	float *theta;		// weights for each of the features / actions
	float *e;			// eligibility trace (num_features * num_actions for each agent)
	float *ep_data;		// state, action, result, state, action values for this action episode
	float *s;			// current state (angle, angular velocity, cart position, cart velocity)
	float *Q;			// Q values for each action, filled when determining best action
	unsigned *action;	// temp storage for action to be taken and the next action
//	unsigned *prev_action;		// storage place for previous action
//	unsigned *f_prev_state;		// feature value for previous state
} AGENT_DATA;		// may hold either host or device pointers

typedef struct{
	float *avg_fail;	// average number of failures per test
} RESULTS;

#pragma mark -
#pragma mark prototypes

void set_params(PARAMS p);
void dump_agents(const char *str, AGENT_DATA *ag);

AGENT_DATA *initialize_agentsGPU(AGENT_DATA *agCPU);
void free_agentsGPU(AGENT_DATA *agGPU);
void run_GPU(AGENT_DATA *cv, RESULTS *r);

AGENT_DATA *initialize_agentsCPU();
void free_agentsCPU(AGENT_DATA *agCPU);
void run_CPU(AGENT_DATA *cv, RESULTS *r);

RESULTS *initialize_results();
void free_results(RESULTS *r);
void display_results(const char *str, RESULTS *r);

#endif
