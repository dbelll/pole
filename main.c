//
//  main.c
//  pole
//
//  Created by Dwight Bell on 8/18/10.
//  Copyright dbelll 2010. All rights reserved.
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "pole.h"

int main(int argc, const char **argv)
{
	PARAMS p = read_params(argc, argv);
	AGENT_DATA *agCPU = initialize_agentsCPU();
	AGENT_DATA *agGPU = (p.run_on_GPU) ? initialize_agentsGPU(agCPU) : NULL;
	RESULTS *rCPU = NULL;
	RESULTS *rGPU = NULL;
	
	if (p.run_on_CPU){
		rCPU = initialize_results();
		run_CPU(agCPU, rCPU);
		display_results("CPU:", rCPU);
	}
	
	if (p.run_on_GPU) {
		rGPU = initialize_results();
		run_GPU(agGPU, rGPU);
		display_results("GPU:", rGPU);
	}
	
	free_agentsCPU(agCPU);
	free_agentsGPU(agGPU);
	
	free_results(rGPU);
	free_results(rCPU);

	return 0;
}

