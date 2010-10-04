#!/bin/bash
# Testing script for home computer
_trials="--TRIALS=100"
_runCPU="--RUN_ON_CPU=1 --RUN_ON_GPU=0"
_print="--NO_PRINT"
_location="./bin/darwin/release"

_time="--TIME_STEPS=100000"
_groups="--AGENT_GROUP_SIZE=1"
_interval="--SHARING_INTERVAL=4"
_test="--TEST_INTERVAL=10000 --TEST_REPS=10000"
#_test=""

_a="--ALPHA"
#_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

for _epsilon in "--EPSILON=.02" #"--EPSILON=.04" "--EPSILON=.08"
do


_common="$_trials $_runCPU $_print $_time $_groups $_interval $_test"

#$_location/pole $_common $_a=.10 $_e=.00 $_g=.95 $_lambda
#$_location/pole $_common $_a=.30 $_e=.00 $_g=.95 $_lambda
$_location/pole $_common $_a=.50 $_epsilon $_g=.95 $_l=.70
#$_location/pole $_common $_a=.70 $_e=.00 $_g=.95 $_lambda

done
