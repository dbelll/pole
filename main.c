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
	read_params(argc, argv);
	COMMON_VALS *cv = initialize_common_values();
	transfer_to_device(cv);
	RESULTS *r = allocate_result_arrays();
	run_GPU(cv, r);
	run_CPU(cv, r);
	free_common_values(cv);
	free_result_arrays(r);
	return 0;
}

