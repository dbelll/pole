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
#include "main.h"
#include "cuda_row_reduction.h"

// paramaters in constant memory on the device
__constant__ unsigned dc_agents;
__constant__ unsigned dc_time_steps;

__constant__ float dc_epsilon;
__constant__ float dc_gamma;
__constant__ float dc_lambda;
__constant__ float dc_alpha;

__constant__ unsigned dc_num_actions;
__constant__ unsigned dc_num_features;

__constant__ unsigned dc_test_interval;
__constant__ unsigned dc_test_reps;

__constant__ unsigned dc_start_time;
__constant__ unsigned dc_end_time;


// fixed pointers in constant memory on the device
__constant__ unsigned *dc_seeds;
__constant__ float *dc_theta;
__constant__ float *dc_e;
__constant__ float *dc_s;
__constant__ float *dc_Q;
__constant__ unsigned *dc_action;

static AGENT_DATA *last_CPU_agent_dump;
static unsigned *d_seeds;
static float *d_theta;
static float *d_e;
static float *d_s;
static float *d_Q;
static unsigned *d_action;

void set_start_end_times(unsigned start, unsigned end)
{
//	printf("pole_kernel start=%d, end=%d\n", start, end);
//	fflush(NULL);

	cudaMemcpyToSymbol("dc_start_time", &start, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_end_time", &end, sizeof(unsigned));
}

// copy parameter values to constant memory on device
void set_constant_params(PARAMS p)
{
	cudaMemcpyToSymbol("dc_agents", &p.agents, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_time_steps", &p.time_steps, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_epsilon", &p.epsilon, sizeof(float));
	cudaMemcpyToSymbol("dc_gamma", &p.gamma, sizeof(float));
	cudaMemcpyToSymbol("dc_lambda", &p.lambda, sizeof(float));
	cudaMemcpyToSymbol("dc_alpha", &p.alpha, sizeof(float));
	cudaMemcpyToSymbol("dc_num_actions", &p.num_actions, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_num_features", &p.num_features, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_test_interval", &p.test_interval, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_test_reps", &p.test_reps, sizeof(unsigned));
}

// copy agent data pointers (device pointers) to constant memory on device
void set_constant_pointers(AGENT_DATA *ag)
{
	cudaMemcpyToSymbol("dc_seeds", &ag->seeds, sizeof(unsigned *));
	cudaMemcpyToSymbol("dc_theta", &ag->theta, sizeof(float *));
	cudaMemcpyToSymbol("dc_e", &ag->e, sizeof(float *));
	cudaMemcpyToSymbol("dc_s", &ag->s, sizeof(float *));
	cudaMemcpyToSymbol("dc_Q", &ag->Q, sizeof(float *));
	cudaMemcpyToSymbol("dc_action", &ag->action, sizeof(unsigned *));
}


/*
	Procedures for setting up and running the pole balancing experiements on CPU and GPU
 */

static PARAMS _p;
static unsigned g_seeds[4] = {2784565659u, 1491908209u, 3415062841u, 3293636241u};

#pragma mark CPU & GPU

// random number in an interval from -max to +max using random uniform distribution
__host__ __device__ float random_interval(unsigned *seeds, unsigned stride, float max)
{
#pragma unused(sd)
	float r = (-max) + 2 * max * RandUniform(seeds, stride);
	// keep generating values until one is within -max to +max
//	do {
//		r = RandNorm(seeds, stride) * sd;
//	} while (r < -max || r > max);
	return r;
}

// randomize the state
__host__ __device__ void randomize_state(float *s, unsigned *seeds, unsigned stride)
{
	s[0] = random_interval(seeds, stride, ANGLE_MAX);
//	s[stride] = random_interval(seeds, stride, ANGLE_VEL_MAX);
	s[stride] = 0.0f;
	s[2*stride] = random_interval(seeds, stride, X_MAX);
//	s[3*stride] = random_interval(seeds, stride, X_VEL_MAX);
	s[3*stride] = 0.0f;
}

// reset eligibility traces to 0.0f
__host__ __device__ void reset_trace(float *e, unsigned num_features, unsigned num_actions, 
										unsigned stride)
{
	for (int f = 0; f < num_features; f++) {
		for (int a = 0; a < num_actions; a++) {
			e[(a + f * num_actions) * stride] = 0.0f;
		}
	}
}

__device__ void reset_traceGPU(float *e)
{
	for (int f = 0; f < dc_num_features; f++) {
		for (int a = 0; a < dc_num_actions; a++) {
			e[(a + f * dc_num_actions) * dc_agents] = 0.0f;
		}
	}
}

__device__ __host__ unsigned terminal_state(float *s, unsigned stride)
{
	return s[2*stride] < X_MIN || s[2*stride] > X_MAX || 
			s[0] < ANGLE_MIN || s[0] > ANGLE_MAX;
}


// take an action from the current state, s, returning the reward and saving the new state in s_prime
__device__ __host__ float take_action(unsigned a, float *s, float *s_prime, unsigned stride)
{
	// formulas are from: Brownlee. The pole balancing problem: a benchmark control theory problem. hdl.handle.net (2005)
	
	// determine force from the action
	float F = a ? FORCE : -FORCE;

	float ang = s[0];
	float ang_vel = s[stride];
	float cos_a = cos(ang);
	float sin_a = sin(ang);
	
	// calculate angular acceleration
	float ang_accel = GRAV * sin_a;
	ang_accel += cos_a * (-F - POLE_MASS * POLE_LENGTH * ang_vel * ang_vel * sin_a) / 
							(CART_MASS + POLE_MASS);
	ang_accel /= POLE_LENGTH * (4.0f/3.0f - POLE_MASS * cos_a * cos_a / (CART_MASS + POLE_MASS));
	
	float x = s[2*stride];
	float x_vel = s[3*stride];

	// calculate x acceleration
	float x_accel = F + POLE_MASS * POLE_LENGTH * (ang_vel * ang_vel * sin_a - ang_accel * cos_a);
	x_accel /= (CART_MASS + POLE_MASS);
	
	// update ang, ang_vel and x, x_vel
	s_prime[0] = ang + TAU * ang_vel;
	s_prime[stride] = ang_vel + TAU * ang_accel;
	s_prime[2*stride] = x + TAU * x_vel;
	s_prime[3*stride] = x_vel + TAU * x_accel;
	
	// determine the reward
	float reward = terminal_state(s_prime, stride) ? REWARD_FAIL : REWARD_NON_FAIL;
	
	return reward;
}

//float take_action_debug(unsigned a, float *s, float *s_prime, unsigned stride)
//{
//	// formulas are from: Brownlee. The pole balancing problem: a benchmark control theory problem. hdl.handle.net (2005)
//	// determine force from the action
//	float F = a ? FORCE : -FORCE;
//
//	float ang = s[0];
//	float ang_vel = s[stride];
//	float x = s[2*stride];
//	float x_vel = s[3*stride];
//
//	float cos_a = cos_a(ang);
//	float sin_a = sin_a(ang);
//	
//	printf("[tack_action_debug] action=%d, angle=%7.4f, angle_vel=%7.4f\n", a, 
//
//	// calculate angular acceleration
//	float ang_accel = GRAV * sin_a;
//	ang_accel += cos_a * (-F - POLE_MASS * POLE_LENGTH * ang_vel * ang_vel * sin_a) / 
//							(CART_MASS + POLE_MASS);
//	ang_accel /= POLE_LENGTH * (4.0f/3.0f - POLE_MASS * cos_a * cos_a / (CART_MASS + POLE_MASS));
//	
//	// calculate x acceleration
//	float x_accel = F + POLE_MASS * POLE_LENGTH * (ang_vel * ang_vel * sin_a - ang_accel * cos_a);
//	x_accel /= (CART_MASS + POLE_MASS);
//	
//	// update ang, ang_vel and x, x_vel
//	s_prime[0] = ang + TAU * ang_vel;
//	s_prime[1] = ang_vel + TAU * ang_accel;
//	s_prime[2] = x + TAU * x_vel;
//	s_prime[3] = x_vel + TAU * x_accel;
//	
//	// determine the reward
//	float reward = REWARD_NON_FAIL;
//	if (s_prime[2] < X_MIN || s_prime[2] > X_MAX || 
//		s_prime[0] < ANGLE_MIN || s_prime[0] > ANGLE_MAX) 
//	{
//		reward = REWARD_FAIL;
//	}
//	
//	return reward;
//}

// Calculate which feature division the state value falls into, based on the min, max,
// and number of divisions.
__device__ __host__ unsigned feature_val_for_state_val(float s, float minv, float maxv, 
														unsigned div)
{
	return max(0, min(div-1, (unsigned)((s-minv)/(maxv-minv) * (float)div)));
}

// Determine which feature corresponds to the given state
__device__ __host__ unsigned feature_for_state(float *s, unsigned stride)
{
	unsigned feature = feature_val_for_state_val(s[0], ANGLE_MIN, ANGLE_MAX, ANGLE_DIV);
	feature += (ANGLE_DIV) * 
				feature_val_for_state_val(s[stride], ANGLE_VEL_MIN, ANGLE_VEL_MAX, ANGLE_VEL_DIV);
	feature += (ANGLE_DIV * ANGLE_VEL_DIV) * 
				feature_val_for_state_val(s[2 * stride], X_MIN, X_MAX, X_DIV);
	feature += (ANGLE_DIV * ANGLE_VEL_DIV * X_DIV) * 
				feature_val_for_state_val(s[3 * stride], X_VEL_MIN, X_VEL_MAX, X_VEL_DIV);
	return feature;
}

__device__ __host__ const char * failure_type(float *s, unsigned stride)
{
	if (s[0] < ANGLE_MIN) return "Angle < MIN";
	if (s[0] > ANGLE_MAX) return "Angle > MAX";
	if (s[2] < X_MIN) return "X < MIN";
	if (s[2] > X_MAX) return "X > MAX";
	return "";
}

// Calculate a number with the division for each state variable
__device__ __host__ unsigned divs_for_feature(unsigned feature)
{
	unsigned divs = feature % ANGLE_DIV;
	feature /= ANGLE_DIV;
	divs += 16 * (feature % ANGLE_VEL_DIV);
	feature /= ANGLE_VEL_DIV;
	divs += 256 * (feature % X_DIV);
	feature /= X_DIV;
	divs += 4096 * feature;
	return divs;
}

// calculate the Q value for an action from a state
__device__ __host__ float calc_Q(float *s, unsigned a, float *theta, unsigned stride, 
																			unsigned num_actions)
{
	// only one feature corresponds with any given state
	unsigned feature = feature_for_state(s, stride);
	float Q = theta[(a + feature * num_actions) * stride];
	return Q;
}

/*
	Calculate the Q value for a given state/action on GPU
	Stride is assumed to be BLOCK_SIZE for s and a and dc_agents for theta
*/
__device__ float calc_QGPU(float *s, unsigned a, float *theta)
{
	// only one feature corresponds with any given state
	unsigned feature = feature_for_state(s, BLOCK_SIZE);
	float Q = theta[(a + feature * dc_num_actions) * dc_agents];
	return Q;
}

__device__ __host__ void update_stored_Q(float *Q, float *s, float *theta, unsigned stride, 
																			unsigned num_actions)
{
	for (int a = 0; a < num_actions; a++) {
		Q[a * stride] = calc_Q(s, a, theta, stride, num_actions);
	}
}

__device__ void update_stored_QGPU(float *Q, float *s, float *theta)
{
	for (int a = 0; a < dc_num_actions; a++) {
		Q[a * BLOCK_SIZE] = calc_QGPU(s, a, theta);
	}
}

// Calculate the Q value for each action from the given state, storing the values in Q
// Return the action with the highest Q value
__device__ __host__ unsigned best_action(float *s, float *theta, float *Q, unsigned stride,
										 unsigned num_actions)
{
	// calculate the Q value for each action
	Q[0] = calc_Q(s, 0, theta, stride, num_actions);
	unsigned best_action = 0;
	float bestQ = Q[0];

	for (int a = 1; a < num_actions; a++) {
		Q[a * stride] = calc_Q(s, a, theta, stride, num_actions);
		if (Q[a * stride] > bestQ) {
			bestQ = Q[a * stride];
			best_action = a;
		}
	}
	return best_action;
}

/*
	Choose the best action on the GPU
	Strides are assumed to be BLOCK_SIZE for s and Q and dc_agents for theta
*/
__device__ unsigned best_actionGPU(float *s, float *theta, float *Q)
{
	// calculate the Q value for each action
	Q[0] = calc_QGPU(s, 0, theta);
	unsigned best_action = 0;
	float bestQ = Q[0];

	for (int a = 1; a < dc_num_actions; a++) {
		Q[a * BLOCK_SIZE] = calc_QGPU(s, a, theta);
		if (Q[a * BLOCK_SIZE] > bestQ) {
			bestQ = Q[a * BLOCK_SIZE];
			best_action = a;
		}
	}
	return best_action;
}

// choose action from current state, storing Q values for each possible action in Q
__device__ __host__ unsigned choose_action(float *s, float *theta, float epsilon, unsigned stride, 
											float *Q, unsigned num_actions, unsigned *seeds)
{
	// always calcualte the best action and store all the Q values for each action
	unsigned a = best_action(s, theta, Q, stride, num_actions);
	if (epsilon > 0.0f && RandUniform(seeds, stride) < epsilon){
		// choose random action
		float r = RandUniform(seeds, stride);
		a = r * num_actions;
	}
	return a;
}

/*
	choose action on GPU
	strides are assumed to be BLOCK_SIZE for s, Q, and seeds and dc_agents theta
*/
__device__ unsigned choose_actionGPU(float *s, float *theta, float *Q, unsigned *seeds)
{
	// always calcualte the best action and store all the Q values for each action
	unsigned a = best_actionGPU(s, theta, Q);
	if (dc_epsilon > 0.0f && RandUniform(seeds, BLOCK_SIZE) < dc_epsilon){
		// choose random action
		float r = RandUniform(seeds, BLOCK_SIZE);
		a = r * dc_num_actions;
	}
	return a;
}

// Update eligibility traces based on action and state
__host__ __device__ void update_trace(unsigned action, float *s, float *e, unsigned num_features,
										unsigned num_actions, unsigned stride, float gamma, float lambda)
{
	unsigned feature = feature_for_state(s, stride);
	for (int f = 0; f < num_features; f++) {
		for (int a = 0; a < num_actions; a++) {
			unsigned index = (a + f * num_actions) * stride;
			// Replacing trace with optional block
			if (f == feature) {
				// set to 1.0 for action selected from current state,
				// set to 0.0 for actions not taken from current state
				e[index] = (a == action) ? 1.0f : 0.0f;
			}else {
				// decay all other values
				e[index] *= gamma * lambda;
			}
		}
	}
}

/*
	Update the eligibility trace values on the GPU
	Stride is assumed to be BLOCK_SIZE for s and dc_agents for e
*/
__device__ void update_traceGPU(unsigned action, float *s, float *e)
{
	unsigned feature = feature_for_state(s, BLOCK_SIZE);
	for (int f = 0; f < dc_num_features; f++) {
		for (int a = 0; a < dc_num_actions; a++) {
			unsigned index = (a + f * dc_num_actions) * dc_agents;
			// Replacing trace with optional block
			if (f == feature) {
				// set to 1.0 for action selected from current state,
				// set to 0.0 for actions not taken from current state
				e[index] = (a == action) ? 1.0f : 0.0f;
			}else {
				// decay all other values
				e[index] *= dc_gamma * dc_lambda;
			}
		}
	}
}



// Update theta values for one agent
// theta = theta + alpha * delta * eligibility trace
__device__ __host__ void update_thetas(float *theta, float *e, float alpha, float delta, unsigned num_features, unsigned stride, unsigned num_actions)
{
	if (alpha == 0.0f || delta == 0.0f) return;
//#ifdef DUMP_THETA_UPDATE_CALCULATIONS
//	printf("updating thetas for alpha = %9.6f, delta = %9.6f\n", alpha, delta);
//#endif
	for (int fa = 0; fa < num_features * num_actions; fa++) {
		if (e[fa*stride] > 0.001f) {
//#ifdef DUMP_THETA_UPDATE_CALCULATIONS
//			printf("   feature-action %5d(%4x) %3d with trace %9.6f changed from %9.6f", (fa/num_actions), divs_for_feature(fa/num_actions), (fa%num_actions), e[fa*stride], theta[fa*stride]);
//#endif
			theta[fa * stride] += alpha * delta * e[fa * stride];
//#ifdef DUMP_THETA_UPDATE_CALCULATIONS
//			printf(" to %9.6f\n", theta[fa*stride]);
//#endif
		}
	}
}

/*
	Update thetas on GPU
	Stride is assumed to be dc_agents for theta and e
*/
__device__ void update_thetasGPU(float *theta, float *e, float delta)
{
	if (dc_alpha == 0.0f || delta == 0.0f) return;
	for (int fa = 0; fa < dc_num_features * dc_num_actions; fa++) {
	//**TODO** see if this test for size of e makes any difference, it was put here for printing purposes
		if (e[fa*dc_agents] > 0.001f) {
			theta[fa * dc_agents] += dc_alpha * delta * e[fa * dc_agents];
		}
	}
}

#pragma mark -
#pragma mark CPU

void set_params(PARAMS p){ _p = p;}

void dump_agent(AGENT_DATA *ag, unsigned agent)
{
	printf("[agent %d]: ", agent);
	printf("   seeds = %u, %u, %u, %u\n", ag->seeds[agent], ag->seeds[agent + _p.agents], 
									   ag->seeds[agent + 2*_p.agents], ag->seeds[agent + 3*_p.agents]);
#ifdef AGENT_DUMP_INCLUDE_THETA_E
	printf("FEATURE       ACTION    THETA       E  \n");
	for (int f = 0; f < _p.num_features; f++) {
		for (int action = 0; action < _p.num_actions; action++) {
			printf("%7d %4x %7d %9.6f %9.6f\n", f, divs_for_feature(f), action, 
				   ag->theta[agent + (action + f * _p.num_actions) * _p.agents], 
				   ag->e[agent + (action + f * _p.num_actions) * _p.agents]);
		}
	}
#endif
	printf("   angle    angleV       x         xV        Q0        Q1   feature\n");
	unsigned feature = feature_for_state(ag->s + agent, _p.agents);
	printf("%9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %7d(%4x)\n", ag->s[agent], ag->s[agent + _p.agents], ag->s[agent + 2*_p.agents], ag->s[agent + 3*_p.agents], ag->Q[agent], ag->Q[agent + _p.agents],
		feature, divs_for_feature(feature));

	printf("chosen action is %d\n", ag->action[agent]);
	
	printf("ACTION  Q-value\n");
//		printf("number of actions is %d\n", p.num_actions);
	for (int action = 0; action < _p.num_actions; action++) {
		(action == ag->action[agent]) ? printf("-->") : printf("   ");
		printf("%3d  %9.6f\n", action, ag->Q[agent + action * _p.agents]);
	}
	printf("\n");
}

void dump_agents(const char *str, AGENT_DATA *ag)
{
	last_CPU_agent_dump = ag;
	printf("%s\n", str);
	for (int agent = 0; agent < _p.agents; agent++) {
		dump_agent(ag, agent);
	}
}

// generate random seeds for the sepecified number of agents
unsigned *create_seeds(unsigned num_agents)
{
	unsigned *seeds = (unsigned *)malloc(num_agents * 4 * sizeof(unsigned));
	for (int i = 0; i < num_agents * 4; i++) {
		seeds[i] = RandUniformui(g_seeds, 1);
	}
	return seeds;
}

// create wgts set initially to random values between RAND_WGT_MIN and RAND_WGT_MAX
float *create_theta(unsigned num_agents, unsigned num_features, unsigned num_actions)
{
#ifdef VERBOSE
	printf("create_theta for %d agents and %d features\n", num_agents, num_features);
#endif
	float *theta = (float *)malloc(num_agents * num_features * num_actions * sizeof(float));
	for (int i = 0; i < num_agents * num_features * num_actions; i++) {
//		float r = RandUniform(g_seeds, 1);
//		theta[i] = (RAND_WGT_MAX - RAND_WGT_MIN) * r + RAND_WGT_MIN;
//		printf("randome = %7.4f, theta = %7.4f\n", r, theta[i]);
		theta[i] = (RAND_WGT_MAX - RAND_WGT_MIN) * RandUniform(g_seeds, 1) + RAND_WGT_MIN;
	}
	return theta;
}

// initial eligibility traces to 0.0f
float *create_e(unsigned num_agents, unsigned num_features, unsigned num_actions)
{
#ifdef VERBOSE
	printf("create_e for %d agents and %d features and %d actions\n", num_agents, num_features, num_actions);
#endif
	float *e = (float *)malloc(num_agents * num_features * num_actions * sizeof(float));
	for (int i = 0; i < num_agents * num_features * num_actions; i++) {
		e[i] = 0.0f;
	}
	return e;
}

// initial random states
float *create_states(unsigned num_agents, unsigned *seeds)
{
	float *states = (float *)malloc(num_agents * _p.state_size * sizeof(float));
	for (int i = 0; i < num_agents; i++) {
//		states[i] = random_interval(seeds + i, num_agents, ANGLE_MAX, STATE_SD);
//		states[i + num_agents] = random_interval(seeds+i, num_agents, ANGLE_VEL_MAX, STATE_SD);
//		states[i + 2 * num_agents] = random_interval(seeds+i, num_agents, X_MAX, STATE_SD);
//		states[i + 3 * num_agents] = random_interval(seeds+i, num_agents, X_VEL_MAX, STATE_SD);
		randomize_state(states + i, seeds + i, num_agents);
	}
	return states;
}

RESULTS *initialize_results()
{
#ifdef VERBOSE
	printf("initializing result arrays...\n");
#endif
	RESULTS *r = (RESULTS *)malloc(sizeof(RESULTS));
	r->avg_fail = (float *)malloc((_p.time_steps / _p.test_interval) * sizeof(float));
	return r;
}

void free_results(RESULTS *r)
{
#ifdef VERBOSE
	printf("freeing result arrays...\n");
#endif
	if (r) {
		if (r->avg_fail) free(r->avg_fail);
		free(r);
	}
}

void display_results(const char *str, RESULTS *r)
{
	printf("%s \n", str);
	printf("    TEST  Avg Fails\n");
	for (int i = 0; i < _p.num_tests; i++) {
		printf("   [%4d]%9.4f\n", i, r->avg_fail[i]);
	}
}

unsigned *create_actions(unsigned num_agents, unsigned num_actions)
{
	unsigned *actions = (unsigned *)malloc(num_agents * num_actions * sizeof(unsigned));
	for (int i = 0; i < num_agents * num_actions; i++) {
		actions[i] = num_actions;	// not possible action
	}
	return actions;
}

// Initialize agents on the CPU.  Some values will be re-used for GPU agents
AGENT_DATA *initialize_agentsCPU()
{
#ifdef VERBOSE
	printf("initializing agents on CPU...\n");
#endif
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	ag->seeds = create_seeds(_p.agents);
	ag->theta = create_theta(_p.agents, _p.num_features, _p.num_actions);
	ag->e = create_e(_p.agents, _p.num_features, _p.num_actions);
//	unsigned rows = _p.agents * ((_p.state_size + 2) * _p.sharing_interval + _p.state_size + 1);
//	ag->ep_data = (float *)malloc(rows * sizeof(float));
	ag->s = create_states(_p.agents, ag->seeds);
	ag->Q = (float *)malloc(_p.agents * _p.num_actions * sizeof(float));
	ag->action = create_actions(_p.agents, _p.num_actions);
	return ag;
}

void dump_state(float *s, unsigned stride)
{
	printf("(%9.6f,%9.6f,%9.6f,%9.6f)[%d]\n", s[0], s[stride], s[2*stride], s[3*stride], 
															feature_for_state(s, stride));
}

// run tests for all agents and return the average failures
float run_test(AGENT_DATA *ag)
{
	unsigned num_failures = 0;
	
	// initialize all agent states
	for (int agent = 0; agent < _p.agents; agent++) {
//		printf("agent %d before testing...\n", agent);
//		dump_agent(ag, agent);
//		unsigned old_num_failures = num_failures;
		
		// save agent state prior to testing
		float s0 = ag->s[agent];
		float s1 = ag->s[agent + _p.agents];
		float s2 = ag->s[agent + 2*_p.agents];
		float s3 = ag->s[agent + 3*_p.agents];
		unsigned act = ag->action[agent];
		unsigned seed0 = ag->seeds[agent];
		unsigned seed1 = ag->seeds[agent + _p.agents];
		unsigned seed2 = ag->seeds[agent + 2 * _p.agents];
		unsigned seed3 = ag->seeds[agent + 3 * _p.agents];
		float Q0 = ag->Q[agent];
		float Q1 = ag->Q[agent + _p.agents];
		
//		randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
		ag->action[agent] = best_action(ag->s + agent, ag->theta + agent, ag->Q + agent, _p.agents, _p.num_actions);
//		choose_action(ag->s + agent, ag->theta + agent, 0.0f, _p.agents, ag->Q + agent, 
//																_p.num_actions, ag->seeds + agent);

		// run the test for specified number of reps
		for (int t = 0; t < _p.test_reps; t++) {
			take_action(ag->action[agent], ag->s+agent, ag->s+agent, _p.agents);
			if (terminal_state(ag->s + agent, _p.agents)){
				++num_failures;
				randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
			}
			// choose best action
			ag->action[agent] = best_action(ag->s + agent, ag->theta + agent, ag->Q + agent, _p.agents, _p.num_actions);
		}
		
		// restore agent state
		ag->s[agent] = s0;
		ag->s[agent + _p.agents] = s1;
		ag->s[agent + 2*_p.agents] = s2;
		ag->s[agent + 3*_p.agents] = s3;
		act = ag->action[agent] = act;
		ag->seeds[agent] = seed0;
		ag->seeds[agent + _p.agents] = seed1;
		ag->seeds[agent + 2 * _p.agents] = seed2;
		ag->seeds[agent + 3 * _p.agents] = seed3;
		ag->Q[agent] = Q0;
		ag->Q[agent + _p.agents] = Q1;
		
//		printf("after testing...\n");
//		dump_agent(ag, agent);

//		printf("agent %d failues = %d\n", agent, num_failures - old_num_failures);
	}



	return num_failures / (float)_p.agents;
}

void run_CPU_noshare(AGENT_DATA *ag, RESULTS *r)
{
	unsigned tot_fails = 0;
#ifdef DUMP_INTERMEDIATE_FAIL_COUNTS
	unsigned prev_tot_fails = 0;
#endif
#ifdef VERBOSE
	printf(" no sharing\n");
#endif

	// on entry the agent's theta, eligibility trace, and state values have been initialized
	
#ifdef DUMP_AGENT_ACTIONS
		printf("-----------------------------------------------------------\n");
		printf("---------------------- INITIAL SETUP ----------------------\n");
		printf("-----------------------------------------------------------\n");
#endif

	// set-up agents to begin the loop by choosing the first action and updating traces
	for (int agent = 0; agent < _p.agents; agent++) {
		ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, _p.epsilon, _p.agents,
										ag->Q + agent, _p.num_actions, ag->seeds + agent);

#ifdef DUMP_AGENT_ACTIONS
		printf("agent %d will choose action %d from state ", agent, ag->action[agent]);
		dump_state(ag->s + agent, _p.agents);
#endif

		update_trace(ag->action[agent], ag->s + agent, ag->e + agent, _p.num_features, 
												_p.num_actions, _p.agents, _p.gamma, _p.lambda);		
	}

#ifdef DUMP_AGENT_ACTIONS
	printf("----------------------------------------------------\n");
	printf("-------------- BEGIN MAIN LOOP ---------------------\n");
	printf("----------------------------------------------------\n");
#endif	

	int k = 1;
	if (_p.time_steps > 40) {
		k = 1 + (_p.time_steps-1)/40;
	}
	
	for (int i = 0; i < (_p.time_steps / k); i++) {
		printf("-");
	}
	printf("|\n");


	// main loop, repeat for the number of trials
	for (int t = 0; t <= _p.time_steps; t++) {
		if (0 == (t+1) % k) {
			printf(".");
			fflush(NULL);
		}

		if (0 == (t % _p.test_interval) && (t > 0)) {
			// run the test and store the result
			unsigned iTest = (t-1) / _p.test_interval;
			r->avg_fail[iTest] = run_test(ag);
//			printf("*********[%3d] test results =%7.2f\n", iTest, r->avg_fail[iTest]);
		}
		if (t == _p.time_steps) break;

#ifdef DUMP_AGENT_ACTIONS
	printf("\n------------------ TIME STEP%3d ------------------------\n", t);
#endif	

		for (int agent = 0; agent < _p.agents; agent++) {

			// stored state is s      stored Q's are Q(s)  
			
#ifdef DUMP_AGENT_ACTIONS
			printf("<<<<<<<< AGENT %d >>>>>>>>>>>>\n", agent);
			printf("time step %d, agent %d ready for next action\n", t, agent);
			dump_agent(ag, agent);
#endif
			// take the action already chosen and saved in ag->action
			unsigned prev_feature = feature_for_state(ag->s, _p.agents);
			float reward = take_action(ag->action[agent], ag->s + agent, ag->s + agent, _p.agents);

#ifdef DUMP_AGENT_BRIEF
			(agent == 0) ? printf("[step%4d]", t) : printf("          ");
			printf("[agent%3d] took action:%2d, got reward:%6.3f, new state is ", agent, ag->action[agent], reward);
			dump_state(ag->s + agent, _p.agents);
#endif
			
			// stored state is s_prime      stored Q's are Q(s)
			unsigned fail = terminal_state(ag->s + agent, _p.agents);
			if (fail){
#ifdef DUMP_FAILURE_TIMES
				printf("Agent%4d Failure at %d taking action %d from state %d (%x) resulting in %s\n", agent, t, ag->action[agent], prev_feature, divs_for_feature(prev_feature), failure_type(ag->s + agent, _p.agents));
#endif
#ifdef DUMP_AGENT_STATE_ON_FAILURE
				printf("session initial state was angle=%6.2f,  angleV=%6.2f, x=%6.2f, xV=%6.2f\n",
						orig_a, orig_aV, orig_x, orig_xV);
				dump_agent(ag, agent);
#endif
				randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
				
//				if (agent == 0){
//					orig_a = ag->s[0];
//					orig_aV = ag->s[_p.agents];
//					orig_x = ag->s[2*_p.agents];
//					orig_xV = ag->s[3*_p.agents];
//				}
				++tot_fails;
			}
						

			float Q_a = ag->Q[agent + ag->action[agent] * _p.agents];

#ifdef DUMP_AGENT_ACTIONS
			if (fail) printf("-------------------------------------------------------\n!!!! terminal state reached, next state is random\n---------------------------------------------------\n\n");
			printf("agent %d, took action %d, got reward %6.3f, now in state s_prime = " , agent,	ag->action[agent], reward);
			dump_state(ag->s + agent, _p.agents);
#endif

#ifdef DUMP_CALCULATIONS
			printf("reward is %9.6f, Q[%d] for state s is %9.6f\n", reward, ag->action[agent], Q_a);
#endif

//			ag->prev_action[agent] = ag->action[agent];
//			ag->f_prev_state[agent] = feature_for_state(ag->s + agent, _p.agents);
			ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, _p.epsilon,
								_p.agents, ag->Q + agent, _p.num_actions, ag->seeds + agent);
			
			// Stored Q values are now based on the new state, s_prime

#ifdef DUMP_AGENT_ACTIONS
			printf("agent %d's next action will be %d with Q-value %9.6f\n", agent, ag->action[agent], ag->Q[agent + ag->action[agent] * _p.agents]);
//			dump_state(ag->s + agent, _p.agents);
#endif

			float Q_a_prime = ag->Q[agent + ag->action[agent] * _p.agents];
			float delta = reward - Q_a + (fail ? 0 : _p.gamma * Q_a_prime);

#ifdef DUMP_CALCULATIONS
			printf("discount is %9.6f, newQ[%d] is %9.6f, so delta is %9.6f\n", _p.gamma, 
												ag->action[agent], (fail ? 0.0f : Q_a_prime), delta);
#endif

#ifdef DUMP_AGENT_ACTIONS
			printf("[update_theta]:\n");
#endif

			update_thetas(ag->theta + agent, ag->e + agent, _p.alpha, delta, _p.num_features,
																	 _p.agents, _p.num_actions);
			if (fail) reset_trace(ag->e + agent, _p.num_features, _p.num_actions, _p.agents);

			update_stored_Q(ag->Q + agent, ag->s + agent, ag->theta + agent, _p.agents, 
																				_p.num_actions);
			
#ifdef DUMP_AGENT_ACTIONS
			printf("[update_trace]\n");
#endif

			update_trace(ag->action[agent], ag->s + agent, ag->e + agent, _p.num_features, _p.num_actions, _p.agents, _p.gamma, _p.lambda);
			
#ifdef DUMP_AGENT_ACTIONS
//			printf("agent state after updating theta and eligibility trace:\n");
//			dump_agent(ag, agent);
#endif
		}

#ifdef DUMP_INTERMEDIATE_FAIL_COUNTS
		if (0 == (1+t) % _p.test_interval) {
			printf("intermediate fail count =%7.2f\n", (tot_fails - prev_tot_fails)/(float)_p.trials);
			prev_tot_fails = tot_fails;
		}
#endif


	}
	
//	printf("*********[%3d] test results =%7.2f\n", _p.time_steps / _p.test_interval, run_test(ag));

#ifdef DUMP_TERMINAL_AGENT_STATE
	printf("\n----------------------------------------------\n");
	dump_agents("               ENDING AGENT STATES\n", ag);
#endif		
	printf("total failures = %d\n", tot_fails);
}

void run_CPU_share(AGENT_DATA *cv, RESULTS *r)
{
#ifdef VERBOSE
	printf(" sharing in agent blocks of %d\n", _p.agent_group_size);
#endif

}

void run_CPU(AGENT_DATA *ag, RESULTS *r)
{
#ifdef VERBOSE
	printf("\n==============================================\nrunning on CPU...\n");
#endif

#ifdef DUMP_INITIAL_AGENTS
	dump_agents("Initial agents on CPU", ag);
#endif
	unsigned timer;
	CREATE_TIMER(&timer);
	START_TIMER(timer);
	_p.agent_group_size > 1 ? run_CPU_share(ag, r) : run_CPU_noshare(ag, r);	
	STOP_TIMER(timer, "run on CPU");
}

void free_agentsCPU(AGENT_DATA *ag)
{
#ifdef VERBOSE
	printf("freeing agents on CPU...\n");
#endif
	if (ag) {
		if (ag->seeds) free(ag->seeds);
		if (ag->theta) free(ag->theta);
		if (ag->e) free(ag->e);
		if (ag->s) free(ag->s);
		if (ag->Q) free(ag->Q);
		if (ag->action) free(ag->action);
		free(ag);
	}
}

#pragma mark -
#pragma mark GPU

AGENT_DATA *copy_GPU_agents()
{
	AGENT_DATA *agGPUcopy = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	agGPUcopy->seeds = host_copyui(d_seeds, _p.agents * 4);
	agGPUcopy->theta = host_copyf(d_theta, _p.agents * _p.num_features * _p.num_actions);
	agGPUcopy->e = host_copyf(d_e, _p.agents * _p.num_features * _p.num_actions);
	agGPUcopy->s = host_copyf(d_s, _p.agents * _p.state_size);
	agGPUcopy->Q = host_copyf(d_Q, _p.agents * _p.num_actions);
	agGPUcopy->action = host_copyui(d_action, _p.agents);
	return agGPUcopy;
}

// check if s1[i] and s2[i] within small value of each other
unsigned mismatch(float *s1, float *s2, unsigned i)
{
	float small = 1.0e-4;
	return s1[i] > (s2[i]+small) || s1[i] < (s2[i]-small);
}

unsigned mismatchui(unsigned *s1, unsigned *s2, unsigned i)
{
	unsigned small = 0;
	return s1[i] > (s2[i]+small) || s1[i] < (s2[i]-small);
}

// check that the GPU agent information copied from the device is the same as the
// CPU agent information pointed to by last_CPU_agent_dump
void check_agents(AGENT_DATA *agGPUcopy)
{
	for (int agent = 0; agent < _p.agents; agent++) {
		printf("[agent%4d] ", agent);
		unsigned match = 1;

		for (int s = 0; s < 4; s++) {
			if (mismatchui(agGPUcopy->seeds, last_CPU_agent_dump->seeds, agent + s*_p.agents)){
				match = 0;
				printf("seed mismatch, ");
				break;
			}
			if (mismatch(agGPUcopy->s, last_CPU_agent_dump->s, agent + s*_p.agents)){
				match = 0;
				printf("state mismatch, ");
				break;
			}
		}
		
		
		
		for (int th = 0; th < _p.num_features * _p.num_actions; th++) {
			if (mismatch(agGPUcopy->theta, last_CPU_agent_dump->theta, agent + th*_p.agents)){
				match = 0;
				printf("theta mismatch feature=%d, action=%d, %f vs %f\n", th/_p.num_actions, th % _p.num_actions, agGPUcopy->theta[agent + th * _p.agents], last_CPU_agent_dump->theta[agent + th * _p.agents]);
//				break;
			}
			if (mismatch(agGPUcopy->e, last_CPU_agent_dump->e, agent + th*_p.agents)){
				match = 0;
				printf("trace mismatch feature=%d, action=%d\n", th/_p.num_actions, th % _p.num_actions);
//				break;
			}
		}
		
		printf(match ? "match\n" : "\n");
	}
}

void dump_agents_GPU(const char *str, unsigned check)
{
	AGENT_DATA *agGPUcopy = copy_GPU_agents();
	if (check) check_agents(agGPUcopy);
	dump_agents(str, agGPUcopy);
	free_agentsCPU(agGPUcopy);
}

/*
	Initializes agent data on GPU by copying the CPU data.
	Also initialized constant memory pointers to point to the GPU data.
	Allocates device memory for:
		dc_seeds, dc_theta, dc_e, dc_s, dc_Q, and dc_action
	Device pointers also stored in host memory: d_seeds, d_theta, d_e, d_s, d_Q, and d_action,
	which are used to free the device memory.
*/
void initialize_agentsGPU(AGENT_DATA *agCPU)
{
#ifdef VERBOSE
	printf("initializing agents on GPU...\n");
#endif
	d_seeds = device_copyui(agCPU->seeds, _p.agents * 4);
	d_theta = device_copyf(agCPU->theta, _p.agents * _p.num_features * _p.num_actions);
	d_e = device_copyf(agCPU->e, _p.agents * _p.num_features * _p.num_actions);
	d_s = device_copyf(agCPU->s, _p.agents * _p.state_size);
	d_Q = device_copyf(agCPU->Q, _p.agents * _p.num_actions);
	d_action = device_copyui(agCPU->action, _p.agents);
	
	cudaMemcpyToSymbol("dc_seeds", &d_seeds, sizeof(unsigned *));
	cudaMemcpyToSymbol("dc_theta", &d_theta, sizeof(float *));
	cudaMemcpyToSymbol("dc_e", &d_e, sizeof(float *));
	cudaMemcpyToSymbol("dc_s", &d_s, sizeof(float *));
	cudaMemcpyToSymbol("dc_Q", &d_Q, sizeof(float *));
	cudaMemcpyToSymbol("dc_action", &d_action, sizeof(unsigned *));
}

// free all agent data from GPU
void free_agentsGPU()
{
#ifdef VERBOSE
	printf("freeing agents on GPU...\n");
#endif
	if (d_seeds) cudaFree(d_seeds);
	if (d_theta) cudaFree(d_theta);
	if (d_e) cudaFree(d_e);
	if (d_s) cudaFree(d_s);
	if (d_Q) cudaFree(d_Q);
	if (d_action) cudaFree(d_action);
}

/*
	copy state information from global device memory to shared memory
	assumes stride is BLOCK_SIZE for shared memory and dc_agents for global memory
*/
#define COPY_STATE_TO_SHARED(iLocal, iGlobal)															\
			s_s[iLocal] = dc_s[iGlobal];										\
			s_s[iLocal + BLOCK_SIZE] = dc_s[iGlobal + dc_agents];				\
			s_s[iLocal + 2*BLOCK_SIZE] = dc_s[iGlobal + 2*dc_agents];			\
			s_s[iLocal + 3*BLOCK_SIZE] = dc_s[iGlobal + 3*dc_agents];			\
			s_action[iLocal] = dc_action[iGlobal];									\
			s_seeds[iLocal] = dc_seeds[iGlobal];								\
			s_seeds[iLocal + BLOCK_SIZE] = dc_seeds[iGlobal + dc_agents];		\
			s_seeds[iLocal + 2*BLOCK_SIZE] = dc_seeds[iGlobal + 2*dc_agents];	\
			s_seeds[iLocal + 3*BLOCK_SIZE] = dc_seeds[iGlobal + 3*dc_agents];	\
			s_Q[iLocal] = dc_Q[iGlobal];										\
			s_Q[iLocal + BLOCK_SIZE] = dc_Q[iGlobal + dc_agents];
			
#define COPY_STATE_TO_GLOBAL(iLocal, iGlobal)									\
			dc_s[iGlobal] = s_s[iLocal];										\
			dc_s[iGlobal + dc_agents] = s_s[iLocal + BLOCK_SIZE];				\
			dc_s[iGlobal + 2*dc_agents] = s_s[iLocal + 2*BLOCK_SIZE];			\
			dc_s[iGlobal + 3*dc_agents] = s_s[iLocal + 3*BLOCK_SIZE];			\
			dc_action[iGlobal] = s_action[iLocal];									\
			dc_seeds[iGlobal] = s_seeds[iLocal];								\
			dc_seeds[iGlobal + dc_agents] = s_seeds[iLocal + BLOCK_SIZE];		\
			dc_seeds[iGlobal + 2*dc_agents] = s_seeds[iLocal + 2*BLOCK_SIZE];	\
			dc_seeds[iGlobal + 3*dc_agents] = s_seeds[iLocal + 3*BLOCK_SIZE];	\
			dc_Q[iGlobal] = s_Q[iLocal];										\
			dc_Q[iGlobal + dc_agents] = s_Q[iLocal + BLOCK_SIZE];

__global__ void pole_kernel(float *results)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;
	
	__shared__ float s_s[4 * BLOCK_SIZE];
	__shared__ unsigned s_action[BLOCK_SIZE];
	__shared__ unsigned s_seeds[4 * BLOCK_SIZE];
	__shared__ float s_Q[2*BLOCK_SIZE];
	
	COPY_STATE_TO_SHARED(idx, iGlobal);
	
	// prepare for first iteration by chosing first action and updating the trace
	if (dc_start_time == 0) {
//		dc_action[iGlobal] = choose_actionGPU(dc_s + iGlobal, dc_theta + iGlobal, dc_Q + iGlobal,
//																				 dc_seeds + iGlobal);
//		update_trace(dc_action[iGlobal], dc_s + iGlobal, dc_e + iGlobal, dc_num_features, 
//													dc_num_actions, dc_agents, dc_gamma, dc_lambda);
		s_action[idx] = choose_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx, s_seeds + idx);
		update_traceGPU(s_action[idx], s_s + idx, dc_e + iGlobal);
	}
	
	for (int t = dc_start_time; t <= dc_end_time; t++) {
		
		// run the test
		if (0 == (t % dc_test_interval) && (t > dc_start_time)){
			// run the test and record the results

			// save state to back to global memory
			COPY_STATE_TO_GLOBAL(idx, iGlobal);
	
			s_action[idx] = best_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx);

			// run the test using shared memory
			unsigned num_failures = 0;
			for (int tt = 0; tt < dc_test_reps; tt++) {
				take_action(s_action[idx], s_s + idx, s_s + idx, BLOCK_SIZE);
				if (terminal_state(s_s + idx, BLOCK_SIZE)) {
					++num_failures;
					randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);
				}
				s_action[idx] = best_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx);
			}
			unsigned iTest = (t-1) / dc_test_interval;
			results[iGlobal + iTest * dc_agents] = num_failures;
			
			// restore agent state
			COPY_STATE_TO_SHARED(idx, iGlobal);
		}
		if (t == dc_end_time) break;
		
		float reward = take_action(s_action[idx], s_s + idx, s_s + idx, BLOCK_SIZE);
		unsigned fail = terminal_state(s_s + idx, BLOCK_SIZE);
		if (fail) randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);			
		float Q_a = s_Q[idx + s_action[idx] * BLOCK_SIZE];
		s_action[idx] = choose_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx, s_seeds + idx);
		float Q_a_prime = s_Q[idx + s_action[idx] * BLOCK_SIZE];
		float delta = reward - Q_a + (fail ? 0 : dc_gamma * Q_a_prime);
		update_thetasGPU(dc_theta + iGlobal, dc_e + iGlobal, delta);
		if (fail) reset_traceGPU(dc_e + iGlobal);
		update_stored_QGPU(s_Q + idx, s_s + idx, dc_theta + iGlobal);
		update_traceGPU(s_action[idx], s_s + idx, dc_e + iGlobal);
	}
	COPY_STATE_TO_GLOBAL(idx, iGlobal);
}

/*
	set all eligibility trace values to 0.0f
*/
__global__ void pole_clear_trace_kernel()
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	if (iGlobal < dc_agents * dc_num_features * dc_num_actions) dc_e[iGlobal] = 0.0f;
}

/*
	Do a learning session for specified number of steps.
	On entry, the theta values are valid from prior learning episodes.
	e values should all be set to 0
	
		First, randomize the state,
		Then repeat the learning process for specified number of iterations
	
	Ending state is not saved.
*/
__global__ void pole_learn_kernel(unsigned steps, unsigned first_time)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;
	if (iGlobal >= dc_agents) return;
	
	__shared__ float s_s[4 * BLOCK_SIZE];
	__shared__ unsigned s_action[BLOCK_SIZE];
	__shared__ unsigned s_seeds[4 * BLOCK_SIZE];
	__shared__ float s_Q[2*BLOCK_SIZE];

	COPY_STATE_TO_SHARED(idx, iGlobal);
	
	// randomize state, determine first action and update eligibility trace
//	randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);
	s_action[idx] = choose_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx, s_seeds + idx);
	update_traceGPU(s_action[idx], s_s + idx, dc_e + iGlobal);

	// loop through specified number of time steps
	for (int t = 0; t < steps; t++) {		
		float reward = take_action(s_action[idx], s_s + idx, s_s + idx, BLOCK_SIZE);
		unsigned fail = terminal_state(s_s + idx, BLOCK_SIZE);
		if (fail) randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);			
		float Q_a = s_Q[idx + s_action[idx] * BLOCK_SIZE];
		s_action[idx] = choose_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx, s_seeds + idx);
		float Q_a_prime = s_Q[idx + s_action[idx] * BLOCK_SIZE];
		float delta = reward - Q_a + (fail ? 0 : dc_gamma * Q_a_prime);
		update_thetasGPU(dc_theta + iGlobal, dc_e + iGlobal, delta);
		if (fail) reset_traceGPU(dc_e + iGlobal);
		update_stored_QGPU(s_Q + idx, s_s + idx, dc_theta + iGlobal);
		update_traceGPU(s_action[idx], s_s + idx, dc_e + iGlobal);
	}
	COPY_STATE_TO_GLOBAL(idx, iGlobal);
}

__global__ void pole_test_kernel(float *results)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;
	if (iGlobal >= dc_agents) return;
	
	__shared__ float s_s[4 * BLOCK_SIZE];
	__shared__ unsigned s_action[BLOCK_SIZE];
	__shared__ unsigned s_seeds[4 * BLOCK_SIZE];
	__shared__ float s_Q[2*BLOCK_SIZE];
	
	COPY_STATE_TO_SHARED(idx, iGlobal);
	
//	s_action[idx] = best_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx);

	// run the test using shared memory
	unsigned num_failures = 0;
	for (int t = 0; t < dc_test_reps; t++) {
		take_action(s_action[idx], s_s + idx, s_s + idx, BLOCK_SIZE);
		if (terminal_state(s_s + idx, BLOCK_SIZE)) {
			++num_failures;
			randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);
		}
		s_action[idx] = best_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx);
	}
	results[iGlobal] = num_failures;
	
	// restore agent state
//	COPY_STATE_TO_GLOBAL(idx, iGlobal);
}

void run_GPU(RESULTS *r)
{
#ifdef VERBOSE
	printf("\n==============================================\nRunning on GPU...\n");
#endif

	// on entry the device constant pointers have been initialized to agent's theta, 
	// eligibility trace, and state values

#ifdef DUMP_INITIAL_AGENTS
//	dump_agents_GPU("initial agents on GPU", ag);
#endif
	
	// setup constant memory on device
	set_constant_params(_p);
	
	// allocate an array to hold individual thread test results
	float *d_results = device_allocf(_p.agents * _p.num_tests);
	
	// one thread for each agent in each trial
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(1 + (_p.agents - 1) / BLOCK_SIZE);
	if (gridDim.x > 65535){
		gridDim.y = 1 + (gridDim.x-1) / 65535;
		gridDim.x = 1 + (gridDim.x-1) / gridDim.y;
	}
	
	dim3 clearTraceBlockDim(512);
	dim3 clearTraceGridDim(1 + (_p.agents * _p.num_features * _p.num_actions - 1) / 512);
	if (clearTraceGridDim.x > 65535) {
		clearTraceGridDim.y = 1 + (clearTraceGridDim.x-1) / 65535;
		clearTraceGridDim.x = 1 + (clearTraceGridDim.x-1) / clearTraceGridDim.y;
	}
#ifdef VERBOSE
	printf("%d total agents\n", _p.agents);
	printf("%d threads per block, (%d x %d) grid of blocks\n", blockDim.x, gridDim.x, gridDim.y);
	printf("for clearing trace: %d threads per block, (%d x %d) grid of blocks\n", 
						clearTraceBlockDim.x, clearTraceGridDim.x, clearTraceGridDim.y);
#endif
	unsigned timer;
	CREATE_TIMER(&timer);
	START_TIMER(timer);
	
	for (int i = 0; i < max(1, _p.num_tests); i++) {
#ifdef VERBOSE
		printf("[%6d] About to run pole_clear_trace_kernel ... ", i);
#endif
		pole_clear_trace_kernel<<<clearTraceGridDim, clearTraceBlockDim>>>();
	CUT_CHECK_ERROR("pole_clear_trace_kernel execution failed");
#ifdef VERBOSE
		printf("done.\n");
		printf("         About to run pole_learn_kernel for %d steps ... ", _p.test_interval);
#endif
		pole_learn_kernel<<<gridDim, blockDim>>>(_p.test_interval, i==0);
	CUT_CHECK_ERROR("pole_learn_kernel execution failed");
#ifdef VERBOSE
		printf("done.\n");
#endif

#ifdef VERBOSE
		printf("         About to run pole_test_kernel (test #%d) ... ", i);
#endif
		pole_test_kernel<<<gridDim, blockDim>>>(d_results + i * _p.agents);
		CUT_CHECK_ERROR("pole_test_kernel execution failed");
#ifdef VERBOSE
		printf("done.\n");
#endif
	}
	cudaThreadSynchronize();
	STOP_TIMER(timer, "run pole kernel on GPU");
	
//	// Check if kernel execution generated an error
//	CUT_CHECK_ERROR("Kernel execution failed");
	
	START_TIMER(timer);
	// reduce the result array on the device and copy back to the host
	row_reduce(d_results, _p.agents, _p.num_tests);
	for (int i = 0; i < _p.num_tests; i++) {
		CUDA_SAFE_CALL(cudaMemcpy(r->avg_fail + i, d_results + i * _p.agents, sizeof(float), 
																cudaMemcpyDeviceToHost));
		r->avg_fail[i] /= _p.trials;
	}
	cudaThreadSynchronize();
	STOP_TIMER(timer, "reduce GPU results and copy data back to host");
	
#ifdef DUMP_TERMINAL_AGENT_STATE
	dump_agents_GPU("--------------------------------------\n       Ending Agent States\n", 0);
#endif

	if (d_results) cudaFree(d_results);
}

