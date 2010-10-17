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
__constant__ unsigned dc_agent_group_size;
__constant__ unsigned dc_time_steps;

__constant__ float dc_epsilon;
__constant__ float dc_gamma;
__constant__ float dc_lambda;
__constant__ float dc_gammaXlambda;
__constant__ float dc_alpha;

__constant__ unsigned dc_num_actions;
__constant__ unsigned dc_num_actionsXagents;
__constant__ unsigned dc_num_features;
__constant__ unsigned dc_num_featuresXactionsXagents;

__constant__ unsigned dc_test_interval;
__constant__ unsigned dc_test_reps;

__constant__ unsigned dc_start_time;
__constant__ unsigned dc_end_time;


// fixed pointers in constant memory on the device
__constant__ unsigned *dc_seeds;
__constant__ float *dc_theta;
__constant__ float *dc_e;
__constant__ float *dc_wgt;
__constant__ float *dc_s;
__constant__ float *dc_Q;
__constant__ unsigned *dc_action;

static AGENT_DATA *last_CPU_agent_dump;

// pointers to agent data on device
static unsigned *d_seeds;
static float *d_theta;
static float *d_e;
static float *d_wgt;
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
	cudaMemcpyToSymbol("dc_agent_group_size", &p.agent_group_size, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_time_steps", &p.time_steps, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_epsilon", &p.epsilon, sizeof(float));
	cudaMemcpyToSymbol("dc_gamma", &p.gamma, sizeof(float));
	cudaMemcpyToSymbol("dc_lambda", &p.lambda, sizeof(float));

	float gammaXlambda = p.gamma * p.lambda;
	cudaMemcpyToSymbol("dc_gammaXlambda", &gammaXlambda, sizeof(float));

	cudaMemcpyToSymbol("dc_alpha", &p.alpha, sizeof(float));
	cudaMemcpyToSymbol("dc_num_actions", &p.num_actions, sizeof(unsigned));

	unsigned num_actionsXagents = p.num_actions * p.agents;
	cudaMemcpyToSymbol("dc_num_actionsXagents", &num_actionsXagents, sizeof(unsigned));

	cudaMemcpyToSymbol("dc_num_features", &p.num_features, sizeof(unsigned));

	unsigned num_featuresXactionsXagents = p.num_features * p.num_actions * p.agents;
	cudaMemcpyToSymbol("dc_num_featuresXactionsXagents", &num_featuresXactionsXagents, sizeof(unsigned));

	cudaMemcpyToSymbol("dc_test_interval", &p.test_interval, sizeof(unsigned));
	cudaMemcpyToSymbol("dc_test_reps", &p.test_reps, sizeof(unsigned));
}

// copy agent data pointers (device pointers) to constant memory on device
//void set_constant_pointers(AGENT_DATA *ag)
//{
//	cudaMemcpyToSymbol("dc_seeds", &ag->seeds, sizeof(unsigned *));
//	cudaMemcpyToSymbol("dc_theta", &ag->theta, sizeof(float *));
//	cudaMemcpyToSymbol("dc_e", &ag->e, sizeof(float *));
//	cudaMemcpyToSymbol("dc_wgt", &ag->wgt, sizeof(float *));
//	cudaMemcpyToSymbol("dc_s", &ag->s, sizeof(float *));
//	cudaMemcpyToSymbol("dc_Q", &ag->Q, sizeof(float *));
//	cudaMemcpyToSymbol("dc_action", &ag->action, sizeof(unsigned *));
//}


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
__host__ void randomize_state(float *s, unsigned *seeds, unsigned stride)
{
//	printf("..... randomizing state .....\n");
	s[0] = random_interval(seeds, stride, ANGLE_MAX);
	s[stride] = random_interval(seeds, stride, ANGLE_VEL_MAX/4.0f);
	s[2*stride] = random_interval(seeds, stride, X_MAX);
	s[3*stride] = random_interval(seeds, stride, X_VEL_MAX/4.0f);
}

// assumed stride for state is BLOCK_SIZE and for seeds is dc_agents
__device__ void randomize_stateGPU(float *s, unsigned *seeds)
{
	s[0] = random_interval(seeds, dc_agents, ANGLE_MAX);
	s[BLOCK_SIZE] = random_interval(seeds, dc_agents, ANGLE_VEL_MAX/4.0f);
	s[2*BLOCK_SIZE] = random_interval(seeds, dc_agents, X_MAX);
	s[3*BLOCK_SIZE] = random_interval(seeds, dc_agents, X_VEL_MAX/4.0f);
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
	for (int f = 0; f < dc_num_featuresXactionsXagents; f += dc_num_actionsXagents) {
		for (int a = 0; a < dc_num_actionsXagents; a += dc_agents) {
			e[a + f] = 0.0f;
		}
	}
}

__device__ __host__ unsigned terminal_state(float *s, unsigned stride)
{
	float s2 = s[2*stride];
	return (s2 < X_MIN) || (s2 > X_MAX) || (s[0] < ANGLE_MIN) || (s[0] > ANGLE_MAX);
}

__host__ unsigned terminal_stateLog(float *s, unsigned stride)
{
	float s2 = s[2*stride];
	printf("X = %7.4f, Angle = %7.4f\n", s2, s[0]);
	printf("s2 < X_MIN?");
	printf((s2 < X_MIN) ? "YES" : "NO");
	printf("s2 > X_MAX?");
	printf((s2 > X_MAX) ? "YES" : "NO");
	unsigned result = (s2 < X_MIN) || (s2 > X_MAX) || (s[0] < ANGLE_MIN) || (s[0] > ANGLE_MAX);
	printf("result is %d\n", result);
	return (s2 < X_MIN) || (s2 > X_MAX) || (s[0] < ANGLE_MIN) || (s[0] > ANGLE_MAX);
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

///*
//	stride of s and s_prime is assumed to be BLOCK_SIZE
//*/
//__device__ float take_actionGPU(unsigned a, float *s, float *s_prime)
//{
//	// formulas are from: Brownlee. The pole balancing problem: a benchmark control theory problem. hdl.handle.net (2005)
//	
//	// determine force from the action
//	float F = a ? FORCE : -FORCE;
//
//	float ang = s[0];
//	float ang_vel = s[BLOCK_SIZE];
//	float cos_a = cos(ang);
//	float sin_a = sin(ang);
//	
//	// calculate angular acceleration
//	float ang_accel = GRAV * sin_a;
//	ang_accel += cos_a * (-F - POLE_MASS * POLE_LENGTH * ang_vel * ang_vel * sin_a) / 
//							(CART_MASS + POLE_MASS);
//	ang_accel /= POLE_LENGTH * (4.0f/3.0f - POLE_MASS * cos_a * cos_a / (CART_MASS + POLE_MASS));
//	
//	float x = s[2*BLOCK_SIZE];
//	float x_vel = s[3*BLOCK_SIZE];
//
//	// calculate x acceleration
//	float x_accel = F + POLE_MASS * POLE_LENGTH * (ang_vel * ang_vel * sin_a - ang_accel * cos_a);
//	x_accel /= (CART_MASS + POLE_MASS);
//	
//	// update ang, ang_vel and x, x_vel
//	s_prime[0] = ang + TAU * ang_vel;
//	s_prime[BLOCK_SIZE] = ang_vel + TAU * ang_accel;
//	s_prime[2*BLOCK_SIZE] = x + TAU * x_vel;
//	s_prime[3*BLOCK_SIZE] = x_vel + TAU * x_accel;
//	
//	// determine the reward
//	float reward = terminal_state(s_prime, BLOCK_SIZE) ? REWARD_FAIL : REWARD_NON_FAIL;
//	
//	return reward;
//}

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

//__device__ unsigned feature_for_stateGPU(float *s)
//{
//	unsigned feature = feature_val_for_state_val(s[0], ANGLE_MIN, ANGLE_MAX, ANGLE_DIV);
//	feature += (ANGLE_DIV) * 
//				feature_val_for_state_val(s[BLOCK_SIZE], ANGLE_VEL_MIN, ANGLE_VEL_MAX, ANGLE_VEL_DIV);
//	feature += (ANGLE_DIV * ANGLE_VEL_DIV) * 
//				feature_val_for_state_val(s[2 * BLOCK_SIZE], X_MIN, X_MAX, X_DIV);
//	feature += (ANGLE_DIV * ANGLE_VEL_DIV * X_DIV) * 
//				feature_val_for_state_val(s[3 * BLOCK_SIZE], X_VEL_MIN, X_VEL_MAX, X_VEL_DIV);
//	return feature;
//}

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
__host__ float calc_Q(float *s, unsigned a, float *theta, unsigned stride, 
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
__device__ float calc_QGPU(float *s, unsigned a, float *theta, unsigned feature)
{
	// only one feature corresponds with any given state
//	unsigned feature = feature_for_state(s, BLOCK_SIZE);
	float Q = theta[(a + feature * NUM_ACTIONS) * dc_agents];
	return Q;
}

__host__ void update_stored_Q(float *Q, float *s, float *theta, unsigned stride, 
																			unsigned num_actions)
{
	for (int a = 0; a < num_actions; a++) {
		Q[a * stride] = calc_Q(s, a, theta, stride, num_actions);
	}
}

__device__ void update_stored_QGPU(float *Q, float *s, float *theta, unsigned feature)
{
	for (int a = 0; a < NUM_ACTIONS; a++) {
		Q[a * BLOCK_SIZE] = calc_QGPU(s, a, theta, feature);
	}
}

// Calculate the Q value for each action from the given state, storing the values in Q
// Return the action with the highest Q value
__host__ unsigned best_action(float *s, float *theta, float *Q, unsigned stride,
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
__device__ unsigned best_actionGPU(float *s, float *theta, float *Q, unsigned feature)
{
	// calculate the Q value for each action
	Q[0] = calc_QGPU(s, 0, theta, feature);
	unsigned best_action = 0;
	float bestQ = Q[0];

	unsigned index = BLOCK_SIZE;

	for (int a = 1; a < NUM_ACTIONS; a++, index += BLOCK_SIZE) {
		Q[index] = calc_QGPU(s, a, theta, feature);
		if (Q[index] > bestQ) {
			bestQ = Q[index];
			best_action = a;
		}
	}
	return best_action;
}

// choose action from current state, storing Q values for each possible action in Q
__host__ unsigned choose_action(float *s, float *theta, float epsilon, unsigned stride, 
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
__device__ unsigned choose_actionGPU(float *s, float *theta, float *Q, unsigned *seeds, unsigned feature)
{
	// always calcualte the best action and store all the Q values for each action
	unsigned a = best_actionGPU(s, theta, Q, feature);
	if (dc_epsilon > 0.0f && RandUniform(seeds, dc_agents) < dc_epsilon){
		// choose random action
		float r = RandUniform(seeds, dc_agents);
		a = r * NUM_ACTIONS;
	}
	return a;
}


// Update eligibility traces based on action and state
__host__ void update_trace(unsigned action, float *s, float *e, unsigned num_features,
										unsigned num_actions, unsigned stride)
{
	unsigned feature = feature_for_state(s, stride);
	float gl = _p.gamma * _p.lambda;
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
				e[index] *= gl;
			}
		}
	}
}

/*
	Update the eligibility trace values on the GPU
	Stride is assumed to be BLOCK_SIZE for s and dc_agents for e
*/
__device__ void update_traceGPU(unsigned action, float *s, float *e, unsigned feature)
{
//	unsigned feature = feature_for_state(s, BLOCK_SIZE);
//	for (int f = 0; f < dc_num_features; f++) {
//		for (int a = 0; a < dc_num_actions; a++) {
//			unsigned index = (a + f * dc_num_actions) * dc_agents;
//			// Replacing trace with optional block
//			if (f == feature) {
//				// set to 1.0 for action selected from current state,
//				// set to 0.0 for actions not taken from current state
//				e[index] = (a == action) ? 1.0f : 0.0f;
//			}else{
//				// decay all other values
//				e[index] *= gl;
//			}
//		}
//	}
	unsigned ff = feature * dc_num_actionsXagents;
	unsigned aa = action * dc_agents;
	for (unsigned f = 0; f < dc_num_featuresXactionsXagents; f += dc_num_actionsXagents) {
		for (unsigned a = 0; a < dc_num_actionsXagents; a += dc_agents) {
			unsigned index = a + f;
			// Replacing trace with optional block
			if (f == ff) {
				// set to 1.0 for action selected from current state,
				// set to 0.0 for actions not taken from current state
				e[index] = (a == aa) ? 1.0f : 0.0f;
			}else{
				// decay all other values
				e[index] *= dc_gammaXlambda;
			}
		}
	}
}



// Update theta values for one agent
// theta = theta + alpha * delta * eligibility trace
__host__ void update_thetas(float *theta, float *e, float *wgt, float alpha, float delta, unsigned num_features, unsigned stride, unsigned num_actions)
{
//#ifdef DUMP_THETA_UPDATE_CALCULATIONS
//	printf("updating thetas for alpha = %9.6f, delta = %9.6f\n", alpha, delta);
//#endif
	for (int fa = 0; fa < num_features * num_actions * stride; fa += stride) {
//#ifdef DUMP_THETA_UPDATE_CALCULATIONS
//			printf("   feature-action %5d(%4x) %3d with trace %9.6f changed from %9.6f", (fa/num_actions), divs_for_feature(fa/num_actions), (fa%num_actions), e[fa*stride], theta[fa*stride]);
//#endif
			theta[fa] += alpha * delta * e[fa];
			wgt[fa] += alpha * e[fa];
//#ifdef DUMP_THETA_UPDATE_CALCULATIONS
//			printf(" to %9.6f\n", theta[fa*stride]);
//#endif
	}
}

/*
	Update thetas on GPU
	Stride is assumed to be dc_agents for theta and e
*/
__device__ void update_thetasGPU(float *theta, float *e, float *wgt, float delta)
{
	float ad = dc_alpha * delta;
	for (int fa = 0; fa < dc_num_featuresXactionsXagents; fa += dc_agents) {
			theta[fa] += ad * e[fa];
			wgt[fa] += dc_alpha * e[fa];
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
	printf("FEATURE       ACTION    THETA       E        WGT\n");
	for (int f = 0; f < _p.num_features; f++) {
		for (int action = 0; action < _p.num_actions; action++) {
			printf("%7d %4x %7d %9.6f %9.6f %9.6f\n", f, divs_for_feature(f), action, 
				   ag->theta[agent + (action + f * _p.num_actions) * _p.agents], 
				   ag->e[agent + (action + f * _p.num_actions) * _p.agents],
				   ag->wgt[agent + (action + f * _p.num_actions) * _p.agents]);
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
	if (terminal_stateLog(ag->s + agent, _p.agents)) {
		printf("******* In terminal state! *********\n");
	}else {
		printf("****** not in terminal state ******\n");
		printf("        MIN   Actual  Max\n");
		printf("    X %7.4f %7.4f %7.4f\n", X_MIN, ag->s[agent + 2 *_p.agents], X_MAX);
		printf("Angle %7.4f %7.4f %7.4f\n", ANGLE_MIN, ag->s[agent], ANGLE_MAX);
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
		
//		theta[i] = 0.0f;
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

// initial wgt's set to 0.0f
float *create_wgt(unsigned num_agents, unsigned num_features, unsigned num_actions)
{
#ifdef VERBOSE
	printf("create_wgt for %d agents and %d features and %d actions\n", num_agents, num_features, num_actions);
#endif
	float *wgt = (float *)malloc(num_agents * num_features * num_actions * sizeof(float));
	for (int i = 0; i < num_agents * num_features * num_actions; i++) {
		wgt[i] = INITIAL_WGT_FOR_SHARING;
	}
	return wgt;
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
	ag->wgt = create_wgt(_p.agents, _p.num_features, _p.num_actions);
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
//		unsigned seed0 = ag->seeds[agent];
//		unsigned seed1 = ag->seeds[agent + _p.agents];
//		unsigned seed2 = ag->seeds[agent + 2 * _p.agents];
//		unsigned seed3 = ag->seeds[agent + 3 * _p.agents];
		float Q0 = ag->Q[agent];
		float Q1 = ag->Q[agent + _p.agents];
		
		randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
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
//		ag->seeds[agent] = seed0;
//		ag->seeds[agent + _p.agents] = seed1;
//		ag->seeds[agent + 2 * _p.agents] = seed2;
//		ag->seeds[agent + 3 * _p.agents] = seed3;
		ag->Q[agent] = Q0;
		ag->Q[agent + _p.agents] = Q1;
		
//		printf("after testing...\n");
//		dump_agent(ag, agent);

//		printf("agent %d failues = %d\n", agent, num_failures - old_num_failures);
	}



	return num_failures / (float)_p.agents;
}

void clear_traces(AGENT_DATA *ag)
{
	for (int i = 0; i < _p.agents * _p.num_features * _p.num_actions; i++) {
		ag->e[i] = 0.0f;
	}
}

void randomize_all_states(AGENT_DATA *ag)
{
	// randomize the state for all agents, preparing for a new test session
	for (int agent = 0; agent < _p.agents; agent++) {
		randomize_state(ag->s + agent,  ag->seeds + agent, _p.agents);
		ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, _p.epsilon, _p.agents,
										ag->Q + agent, _p.num_actions, ag->seeds + agent);
		update_trace(ag->action[agent], ag->s + agent, ag->e + agent, _p.num_features, 
												_p.num_actions, _p.agents);
	}
}

void learning_session(AGENT_DATA *ag)
{
	// run learning session for all agents for one chunk of time
	for (int agent = 0; agent < _p.agents; agent++) {
		// loop over the time steps in the chunk
		for (int t = 0; t < _p.chunk_interval; t++) {
			float reward = take_action(ag->action[agent], ag->s + agent, ag->s + agent, _p.agents);
			unsigned fail = terminal_state(ag->s + agent, _p.agents);
			if (fail) randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
			float Q_a = ag->Q[agent + ag->action[agent] * _p.agents];
			ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, _p.epsilon,
								_p.agents, ag->Q + agent, _p.num_actions, ag->seeds + agent);
			float Q_a_prime = ag->Q[agent + ag->action[agent] * _p.agents];
			float delta = reward - Q_a + (fail ? 0 : _p.gamma * Q_a_prime);
			update_thetas(ag->theta + agent, ag->e + agent, ag->wgt + agent, _p.alpha, delta, _p.num_features, _p.agents, _p.num_actions);
			if (fail) reset_trace(ag->e + agent, _p.num_features, _p.num_actions, _p.agents);
			update_stored_Q(ag->Q + agent, ag->s + agent, ag->theta + agent, _p.agents, 
																				_p.num_actions);
			update_trace(ag->action[agent], ag->s + agent, ag->e + agent, _p.num_features, 
												_p.num_actions, _p.agents);
		}
	}
}

// calculate average theta values within each agent group and duplicate
// for all agents in the group
void share_theta(AGENT_DATA *ag)
{
	// loop over every agent group and accumulate the theta values and wgt's
	// in agent 0 in that group, then duplicate for all agents in group
	for (int i = 0; i < _p.trials; i++) {
		for (int fa = 0; fa < _p.num_features * _p.num_actions; fa++) {
			unsigned agent0 = i * _p.agent_group_size + fa * _p.agents;
			float block_theta = 0.0f;
			float block_wgt = 0.0f;
			// accumulate wgtd theta and total wgt
			for (int a = agent0; a < agent0 + _p.agent_group_size; a++) {
				block_theta += ag->theta[a] * ag->wgt[a];
				block_wgt += ag->wgt[a];
//				printf("fa[%d] agent[%d] theta=%7.4f wgt=%7.4f   block_thetat=%8.4f block_wgt=%8.4f\n",
//						fa, a-agent0, ag->theta[a], ag->wgt[a], block_theta, block_wgt);
			}
			if (block_wgt > 0.0f){
				block_theta /= block_wgt;	// convert to the average theta
				block_wgt /= _p.agent_group_size;		// evenly divide total wgt over all agents
			}
//			printf("final block_theta is %7.4f and block_wgt is %7.4f\n", block_theta, block_wgt);
			if (block_wgt > 0.0f) {
				for (int a = agent0; a < agent0 + _p.agent_group_size; a++) {
					ag->theta[a] = block_theta;
					ag->wgt[a] = block_wgt;
				}
			}
		}
	}
}

void run_CPU_noshare(AGENT_DATA *ag, RESULTS *r)
{
	// on entry the agent's theta, eligibility trace, and state values have been initialized
	
	int k = 1;
	if (_p.num_chunks > 40) {
		k = 1 + (_p.num_chunks-1)/40;
	}
	
	for (int i = 0; i < (_p.num_chunks / k); i++) {
		printf("-");
	}
	printf("|\n");

#ifdef VERBOSE
			printf("%d chunks per share\n", _p.chunks_per_share);
#endif

	for (int i = 0; i < _p.num_chunks; i++) {
#ifdef VERBOSE
			printf("--------------- new chunk [%d]------------------\n", i);
#endif
		// print progress indicator dots
		if (0 == (i+1) % k) { printf("."); fflush(NULL); }

		if(0 == (i % _p.chunks_per_restart)){

#ifdef VERBOSE
			printf("clearing traces ...\n");
#endif
			clear_traces(ag);
#ifdef VERBOSE
//			dump_agents("\n agent state after clearing, before randomizing", ag);
			printf("randomizing state ...\n");
#endif
			randomize_all_states(ag);
		}
#ifdef VERBOSE
		printf("learning session ...\n");
#endif
		learning_session(ag);
//		dump_agents("\n agent state after learning", ag);
		
		if ((_p.agent_group_size > 1) && 0 == ((i+1)%_p.chunks_per_share)) {
#ifdef VERBOSE
			printf("sharing ...\n");
#endif
#ifdef DUMP_AGENT_BEFORE_SHARING
			dump_agents("\n agent state before sharing", ag);
#endif
			share_theta(ag);
//			dump_agents("\n agent state after sharing", ag);
		}
		
		if (0 == ((i+1)%_p.chunks_per_test)) {
#ifdef VERBOSE
			printf("testing...\n");
#endif
			r->avg_fail[i/_p.chunks_per_test] = run_test(ag);
//			dump_agents("\n agent state after test", ag);
		}
#ifdef DUMP_AGENT_END_OF_CHUNK
			dump_agents("\n agent state at end of chunk", ag);
#endif
	}

#ifdef DUMP_TERMINAL_AGENT_STATE
	printf("\n----------------------------------------------\n");
	dump_agents("               ENDING AGENT STATES\n", ag);
#endif		
	printf("\n");
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
	_p.agent_group_size > 1 ? run_CPU_noshare(ag, r) : run_CPU_noshare(ag, r);
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
		if (ag->wgt) free(ag->wgt);
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
	agGPUcopy->wgt = host_copyf(d_wgt, _p.agents * _p.num_features * _p.num_actions);
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
	d_wgt = device_copyf(agCPU->wgt, _p.agents * _p.num_features * _p.num_actions);
	d_s = device_copyf(agCPU->s, _p.agents * _p.state_size);
	d_Q = device_copyf(agCPU->Q, _p.agents * _p.num_actions);
	d_action = device_copyui(agCPU->action, _p.agents);
	
	cudaMemcpyToSymbol("dc_seeds", &d_seeds, sizeof(unsigned *));
	cudaMemcpyToSymbol("dc_theta", &d_theta, sizeof(float *));
	cudaMemcpyToSymbol("dc_e", &d_e, sizeof(float *));
	cudaMemcpyToSymbol("dc_wgt", &d_wgt, sizeof(float *));
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
	if (d_wgt) cudaFree(d_wgt);
	if (d_s) cudaFree(d_s);
	if (d_Q) cudaFree(d_Q);
	if (d_action) cudaFree(d_action);
}

/*
	copy state information from global device memory to shared memory
	assumes stride is BLOCK_SIZE for shared memory and dc_agents for global memory
*/
#define COPY_STATE_TO_SHARED(iLocal, iGlobal)	{						\
			s_s[iLocal] = dc_s[iGlobal];								\
			s_s[iLocal + BLOCK_SIZE] = dc_s[iGlobal + dc_agents];		\
			s_s[iLocal + 2*BLOCK_SIZE] = dc_s[iGlobal + 2*dc_agents];	\
			s_s[iLocal + 3*BLOCK_SIZE] = dc_s[iGlobal + 3*dc_agents];	\
			s_action[iLocal] = dc_action[iGlobal];						\
			s_Q[iLocal] = dc_Q[iGlobal];								\
			s_Q[iLocal + BLOCK_SIZE] = dc_Q[iGlobal + dc_agents];		\
		}
			
#define COPY_STATE_TO_GLOBAL(iLocal, iGlobal)	{						\
			dc_s[iGlobal] = s_s[iLocal];								\
			dc_s[iGlobal + dc_agents] = s_s[iLocal + BLOCK_SIZE];		\
			dc_s[iGlobal + 2*dc_agents] = s_s[iLocal + 2*BLOCK_SIZE];	\
			dc_s[iGlobal + 3*dc_agents] = s_s[iLocal + 3*BLOCK_SIZE];	\
			dc_action[iGlobal] = s_action[iLocal];						\
			dc_Q[iGlobal] = s_Q[iLocal];								\
			dc_Q[iGlobal + dc_agents] = s_Q[iLocal + BLOCK_SIZE];		\
		}
/*
*	Calculate average thetas for each feature/action value for the entire group and share with 
*	all threads in the group
*	The group's y dimension is the feature/action index.
*	Shared memory is used to do the reduction to get total values for the group.
*/
__global__ void pole_share_kernel()
{
	unsigned idx = threadIdx.x;
	unsigned fa = blockIdx.y;
	unsigned iGlobal = idx + blockIdx.x * blockDim.x + fa * dc_agents;
	
	// copy thetas and wgts to shared memory
	extern __shared__ float s_theta[];
	float *s_wgt = s_theta + dc_agent_group_size;
	
	s_theta[idx] = dc_theta[iGlobal];
	s_wgt[idx] = dc_wgt[iGlobal];
	
	float old_theta = s_theta[idx];
	s_theta[idx] *= s_wgt[idx];			// convert to wgtd theta

	__syncthreads();
	
	// do a reduction on theta for this group
	for (unsigned half = dc_agent_group_size >> 1; half > 0; half >>= 1) {
		if (idx < half) {
			s_theta[idx] += s_theta[idx + half];
			s_wgt[idx] += s_wgt[idx + half];
		}
		__syncthreads();
	}
	
	// copy the values at index 0 to all threads
	s_theta[idx] = s_theta[0];
	s_wgt[idx] = s_wgt[0];
	__syncthreads();
	
	if (s_wgt[idx] > 0.0f) {
		s_theta[idx] /= s_wgt[idx];
		s_wgt[idx] /= dc_agent_group_size;
	}else {
		s_theta[idx] = old_theta;
	}
	
	__syncthreads();
	// copy s_theta back to global memory, converting it to the average value
	dc_theta[iGlobal] = s_theta[idx];
	dc_wgt[iGlobal] = s_wgt[idx];
}


/*
	set all eligibility trace values to 0.0f
*/
__global__ void pole_clear_trace_kernel()
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	if (iGlobal < dc_num_featuresXactionsXagents) dc_e[iGlobal] = 0.0f;
}

/*
	Do a learning session for specified number of steps.
	On entry, the theta values are valid from prior learning episodes.
	e values should all be set to 0
	
		First, randomize the state,
		Then repeat the learning process for specified number of iterations
	
	Ending state is not saved.
*/
__global__ void pole_learn_kernel(unsigned steps, unsigned isRestart)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;
	if (iGlobal >= dc_agents) return;
	
	__shared__ float s_s[4 * BLOCK_SIZE];
	__shared__ unsigned s_action[BLOCK_SIZE];
	__shared__ float s_Q[2*BLOCK_SIZE];

	if (isRestart) {
		// randomize state, determine first action and update eligibility trace
		randomize_stateGPU(s_s + idx, dc_seeds + iGlobal);
		unsigned feature = feature_for_state(s_s + idx, BLOCK_SIZE);
		s_action[idx] = choose_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx, dc_seeds + iGlobal, feature);
		update_traceGPU(s_action[idx], s_s + idx, dc_e + iGlobal, feature);
	} else COPY_STATE_TO_SHARED(idx, iGlobal);

	// loop through specified number of time steps
	float *s_sidx = s_s + idx;
	float *s_Qidx = s_Q + idx;
	for (int t = 0; t < steps; t++) {		
		float reward = take_action(s_action[idx], s_sidx, s_sidx, BLOCK_SIZE);
		unsigned fail = (reward == REWARD_FAIL);
		if (fail) randomize_stateGPU(s_sidx, dc_seeds + iGlobal);
		unsigned feature = feature_for_state(s_sidx, BLOCK_SIZE);
		float Q_a = s_Q[idx + s_action[idx] * BLOCK_SIZE];
		s_action[idx] = choose_actionGPU(s_sidx, dc_theta + iGlobal, s_Qidx, dc_seeds + iGlobal, feature);
		float Q_a_prime = s_Q[idx + s_action[idx] * BLOCK_SIZE];
		float delta = reward - Q_a + (fail ? 0 : dc_gamma * Q_a_prime);
		update_thetasGPU(dc_theta + iGlobal, dc_e + iGlobal, dc_wgt + iGlobal, delta);
		if (fail) reset_traceGPU(dc_e + iGlobal);
		update_stored_QGPU(s_Qidx, s_sidx, dc_theta + iGlobal, feature);
		update_traceGPU(s_action[idx], s_sidx, dc_e + iGlobal, feature);
	}

//	if (!nextIsRestart) 
	COPY_STATE_TO_GLOBAL(idx, iGlobal);
}

__global__ void pole_test_kernel(float *results)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;
	if (iGlobal >= dc_agents) return;
	
	__shared__ float s_s[4 * BLOCK_SIZE];
	__shared__ unsigned s_action[BLOCK_SIZE];
//	__shared__ unsigned s_seeds[4 * BLOCK_SIZE];
	__shared__ float s_Q[2*BLOCK_SIZE];
	
//	COPY_STATE_TO_SHARED(idx, iGlobal);
	
	randomize_stateGPU(s_s + idx, dc_seeds + iGlobal);
	unsigned feature = feature_for_state(s_s + idx, BLOCK_SIZE);
	s_action[idx] = best_actionGPU(s_s + idx, dc_theta + iGlobal, s_Q + idx, feature);

	// run the test using shared memory
	unsigned num_failures = 0;
	float *s_sidx = s_s + idx;
	float *s_Qidx = s_Q + idx;
	for (int t = 0; t < dc_test_reps; t++) {
		take_action(s_action[idx], s_sidx, s_sidx, BLOCK_SIZE);
		if (terminal_state(s_sidx, BLOCK_SIZE)) {
			++num_failures;
			randomize_stateGPU(s_sidx, dc_seeds + iGlobal);
		}
		unsigned feature = feature_for_state(s_s + idx, BLOCK_SIZE);
		s_action[idx] = best_actionGPU(s_sidx, dc_theta + iGlobal, s_Qidx, feature);
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

//	printf("X_MIN = %7.4f X_MAX = %7.4f, ANGLE_MIN=%7.4f ANGLE_MAX=%7.4f\n",
//		X_MIN, X_MAX, ANGLE_MIN, ANGLE_MAX);

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
	
	dim3 shareBlockDim(_p.agent_group_size);
	dim3 shareGridDim(_p.trials, _p.num_features * _p.num_actions);
	
#ifdef VERBOSE
	printf("%d total agents\n", _p.agents);
	printf("%d threads per block, (%d x %d) grid of blocks\n", blockDim.x, gridDim.x, gridDim.y);
	printf("for clearing trace: %d threads per block, (%d x %d) grid of blocks\n", 
						clearTraceBlockDim.x, clearTraceGridDim.x, clearTraceGridDim.y);
#endif

	float timeClear = 0.0f;
	float timeLearn = 0.0f;
	float timeShare = 0.0f;
	float timeTest = 0.0f;
	float timeReduce = 0.0f;
	unsigned timerCPU;
	CREATE_TIMER(&timerCPU);
	START_TIMER(timerCPU);
	
	CUDA_EVENT_PREPARE;
	
#ifdef VERBOSE
	printf("chunk interval is %d and there are %d chunks in the total time steps of %d\n", 
			_p.chunk_interval, _p.num_chunks, _p.time_steps);
	printf("  restart interval is %d which is %d chunks\n", _p.restart_interval, _p.chunks_per_restart);
	printf("  sharing interval is %d which is %d chunks\n", _p.sharing_interval, _p.chunks_per_share);
	printf("  testing interval is %d which is %d chunks\n", _p.test_interval, _p.chunks_per_test);
#endif
	
	for (int i = 0; i < _p.num_chunks; i++) {
#ifdef VERBOSE
			printf("--------------- new chunk [%d]------------------\n", i);
#endif

		unsigned isRestart = (0 == (i % _p.chunks_per_restart));
		if(isRestart){
//			printf("clearing traces ...\n");
			// reset traces
			CUDA_EVENT_START
			pole_clear_trace_kernel<<<clearTraceGridDim, clearTraceBlockDim>>>();
			CUDA_EVENT_STOP(timeClear);
			CUT_CHECK_ERROR("pole_clear_trace_kernel execution failed");
//			dump_agents_GPU("\n agent state after clearing, before randomizing\n", 0);
		}


		// always do learning for this chunk of time
//		if (isRestart) printf("randomizing state ...\n");
//		printf("learning session ...\n");
		CUDA_EVENT_START
		pole_learn_kernel<<<gridDim, blockDim>>>(_p.chunk_interval, isRestart);
		CUDA_EVENT_STOP(timeLearn);
		CUT_CHECK_ERROR("pole_learn_kernel execution failed");
//		dump_agents_GPU("\n agent state after learning\n", 0);

		if ((_p.agent_group_size > 1) && (0 == ((i+1) % _p.chunks_per_share))) {
//			printf("sharing ...\n");

#ifdef DUMP_AGENT_BEFORE_SHARING
			dump_agents_GPU("\n agent state prior to sharing", 0);
#endif
			CUDA_EVENT_START;
			pole_share_kernel<<<shareGridDim, shareBlockDim, _p.agent_group_size * sizeof(float)>>>();
			CUDA_EVENT_STOP(timeShare);
			CUT_CHECK_ERROR("pole_share_kernel execution failed");
//			dump_agents_GPU("\n agent state after sharing\n", 0);
		}
		
		if (0 == ((i+1) % _p.chunks_per_test)) {
//			printf("testing ...\n");
			CUDA_EVENT_START;
			pole_test_kernel<<<gridDim, blockDim>>>(d_results + (i / _p.chunks_per_test) * _p.agents);
			CUDA_EVENT_STOP(timeTest);
			CUT_CHECK_ERROR("pole_test_kernel execution failed");
//			dump_agents_GPU("\n agent state after test\n", 0);
		}
#ifdef DUMP_AGENT_END_OF_CHUNK
		dump_agents_GPU("\nagent state at end of chunk", 0);
#endif
	}
	
	// reduce the result array on the device and copy back to the host
	CUDA_EVENT_START;
	row_reduce(d_results, _p.agents, _p.num_tests);
	for (int i = 0; i < _p.num_tests; i++) {
		CUDA_SAFE_CALL(cudaMemcpy(r->avg_fail + i, d_results + i * _p.agents, sizeof(float), 
																cudaMemcpyDeviceToHost));
		r->avg_fail[i] /= _p.agents;
	}
	CUDA_EVENT_STOP(timeReduce);
	CUDA_EVENT_CLEANUP;
	STOP_TIMER(timerCPU, "total GPU time");	
	PRINT_TIME(timeClear, "pole_clear_trace_kernel");
	PRINT_TIME(timeLearn, "pole_learn_kernel");
	PRINT_TIME(timeShare, "pole_share_kernel");
	PRINT_TIME(timeTest, "pole_test_kernel");
	PRINT_TIME(timeReduce, "pole_reduce_kernel");
	
#ifdef DUMP_TERMINAL_AGENT_STATE
	dump_agents_GPU("--------------------------------------\n       Ending Agent States\n", 0);
#endif

	if (d_results) cudaFree(d_results);
}

