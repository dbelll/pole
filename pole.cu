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

/*
	Procedures for setting up and running the pole balancing experiements on CPU and GPU
 */

static PARAMS _p;
static unsigned g_seeds[4] = {2784565659u, 1491908209u, 3415062841u, 3293636241u};

#pragma mark CPU & GPU

// random number in an interval from -max to +max using random normal with standard deviation = sd
__host__ __device__ float random_interval(unsigned *seeds, unsigned stride, float max, float sd)
{
	float r;
	// keep generating values until one is within -max to +max
	do {
		r = RandNorm(seeds, stride) / (sd * max);
	} while (r < -max || r > max);
	return r;
}

// randomize the state
__host__ __device__ void randomize_state(float *s, unsigned *seeds, unsigned stride)
{
	s[0] = random_interval(seeds, stride, ANGLE_MAX, ANGLE_MAX / SD_FOR_MAX);
	s[stride] = random_interval(seeds, stride, ANGLE_VEL_MAX, ANGLE_VEL_MAX / SD_FOR_MAX);
	s[2*stride] = random_interval(seeds, stride, X_MAX, X_MAX / SD_FOR_MAX);
	s[3*stride] = random_interval(seeds, stride, X_VEL_MAX, X_VEL_MAX / SD_FOR_MAX);
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

__device__ __host__ void update_stored_Q(float *Q, float *s, float *theta, unsigned stride, 
																			unsigned num_actions)
{
	for (int a = 0; a < num_actions; a++) {
		Q[a * stride] = calc_Q(s, a, theta, stride, num_actions);
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

// choose action from current state, storing Q values for each possible action in Q
__device__ __host__ unsigned choose_action(float *s, float *theta, float epsilon, unsigned stride, 
											float *Q, unsigned num_actions, unsigned *seeds)
{
	// always calcualte the best action and store all the Q values for each action
	unsigned a = best_action(s, theta, Q, stride, num_actions);
	if (RandUniform(seeds, stride) < epsilon){
		// choose random action
		float r = RandUniform(seeds, stride);
		a = r * num_actions;
	}
	return a;
}

// Update eligibility traces based on action and state
__device__ __host__ void update_trace(unsigned action, float *s, float *e, unsigned num_features,
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

// Update theta values for one agent
// theta = theta + alpha * delta * eligibility trace
//__device__
 __host__ void update_thetas(float *theta, float *e, float alpha, float delta, unsigned num_features, unsigned stride, unsigned num_actions)
{
	if (alpha == 0.0f || delta == 0.0f) return;
#ifdef DUMP_THETA_UPDATE_CALCULATIONS
	printf("updating thetas for alpha = %9.6f, delta = %9.6f\n", alpha, delta);
#endif
	for (int fa = 0; fa < num_features * num_actions; fa++) {
		if (e[fa*stride] > 0.001f) {
#ifdef DUMP_THETA_UPDATE_CALCULATIONS
			printf("   feature-action %5d(%4x) %3d with trace %9.6f changed from %9.6f", (fa/num_actions), divs_for_feature(fa/num_actions), (fa%num_actions), e[fa*stride], theta[fa*stride]);
#endif
			theta[fa * stride] += alpha * delta * e[fa * stride];
#ifdef DUMP_THETA_UPDATE_CALCULATIONS
			printf(" to %9.6f\n", theta[fa*stride]);
#endif
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
	printf("%s, agents for %s\n", str, ag->device_flag ? "device" : "host");
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
	float *states = (float *)malloc(num_agents * NUM_STATE_VALUES * sizeof(float));
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
	r->begun = (float *)malloc(_p.data_lines * sizeof(float));
	r->ended = (float *)malloc(_p.data_lines * sizeof(float));
	r->total_length = (float *)malloc(_p.data_lines * sizeof(float));
	return r;
}

void free_results(RESULTS *r)
{
#ifdef VERBOSE
	printf("freeing result arrays...\n");
#endif
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
	printf("    TIME    BEGUN   ENDED   TOT_LENGTH\n");
	for (int i = 0; i < _p.data_lines; i++) {
		printf("   [%4d] %7.0f %7.0f %12.0f\n", i, r->begun[i], r->ended[i], r->total_length[i]);
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
	unsigned rows = _p.agents * ((_p.state_size + 2) * _p.sharing_interval + _p.state_size + 1);
	ag->ep_data = (float *)malloc(rows * sizeof(float));
	ag->s = create_states(_p.agents, ag->seeds);
	ag->Q = (float *)malloc(_p.agents * _p.num_actions * sizeof(float));
	ag->action = create_actions(_p.agents, _p.num_actions);
//	ag->prev_action = (unsigned *)malloc(_p.agents * sizeof(unsigned));
//	ag->f_prev_state = (unsigned *)malloc(_p.agents * sizeof(unsigned));
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
		
		// save agent state
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
		
		randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);

		// run the test for specified number of reps
		for (int t = 0; t < _p.test_reps; t++) {
			take_action(ag->action[agent], ag->s+agent, ag->s+agent, _p.agents);
			if (terminal_state(ag->s + agent, _p.agents)){
				++num_failures;
				randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
			}
			// choose action with epsilon = 0.0
			ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, 0.0f, 
									_p.agents, ag->Q + agent, _p.num_actions, ag->seeds + agent);
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
//	float orig_a = ag->s[0];
//	float orig_aV = ag->s[_p.agents];
//	float orig_x = ag->s[2*_p.agents];
//	float orig_xV = ag->s[3*_p.agents];
//	printf("orig state: %6.2f %6.2f %6.2f %6.2f\n", orig_a, orig_aV, orig_x, orig_xV);

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

	// main loop, repeat for the number of trials
	for (int t = 0; t < _p.time_steps; t++) {
	
		if (0 == (t % _p.test_interval) && (t > 0)) {
			printf("*********[%3d] test results =%7.2f\n", t / _p.test_interval, run_test(ag));
		}

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
		if (0 == (1+t) % DUMP_INTERMEDIATE_FAIL_COUNTS) {
			printf("intermediate fail count =%7.2f\n", (tot_fails - prev_tot_fails)/(float)_p.trials);
			prev_tot_fails = tot_fails;
		}
#endif


	}
	
	printf("*********[%3d] test results =%7.2f\n", _p.time_steps / _p.test_interval, run_test(ag));

#ifdef DUMP_TERMINAL_AGENT_STATE
	printf("\n==================================================\n");
	printf(  "               ENDING AGENT STATES");
	for (int agent = 0; agent < _p.agents; agent++) {
		dump_agent(ag, agent);

#ifdef AGENT_TERMINAL_DUMP_INCLUDE_THETA_E
		printf("FEATURE       ACTION    THETA       E  \n");
		for (int f = 0; f < _p.num_features; f++) {
			for (int action = 0; action < _p.num_actions; action++) {
				printf("%7d %4x %7d %9.6f %9.6f\n", f, divs_for_feature(f), action, 
					   ag->theta[agent + (action + f * _p.num_actions) * _p.agents], 
					   ag->e[agent + (action + f * _p.num_actions) * _p.agents]);
			}
		}
#endif

	}
#endif		
	printf("total failures = %d\n", tot_fails);
}

void run_CPU_share(AGENT_DATA *cv, RESULTS *r)
{
#ifdef VERBOSE
	printf(" sharing in agent blocks of %d\n", _p.agent_group_size);
#endif

}

void run_CPU(AGENT_DATA *cv, RESULTS *r)
{
#ifdef VERBOSE
	printf("running on CPU...");
#endif
	
	_p.block_sharing ? run_CPU_share(cv, r) : run_CPU_noshare(cv, r);	
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
		if (ag->ep_data) free(ag->ep_data);
		if (ag->s) free(ag->s);
		if (ag->Q) free(ag->Q);
		free(ag);
	}
}

#pragma mark -
#pragma mark GPU

AGENT_DATA *initialize_agentsGPU(AGENT_DATA *agCPU)
{
#ifdef VERBOSE
	printf("initializing agents on GPU...\n");
#endif
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	
	return ag;	
}

void free_agentsGPU(AGENT_DATA *ag)
{
#ifdef VERBOSE
	printf("freeing agents on GPU...\n");
#endif
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

void run_GPU(AGENT_DATA *cv, RESULTS *r)
{
#ifdef VERBOSE
	printf("running on CPU...\n");
#endif
}

__global__ void kernel_operation(int n, float *x)
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

