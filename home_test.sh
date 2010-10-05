#!/bin/bash
# Testing script for home computer
_trials="--TRIALS=1024"
_time="--TIME_STEPS=262144"
_groups="--AGENT_GROUP_SIZE=1"
_interval="--SHARING_INTERVAL=4"
_ss="--STATE_SIZE=4"
_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/darwin/release"

_test="--TEST_INTERVAL=16384 --TEST_REPS=16384"


_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

#for _epsilon in "--EPSILON=.00" #"--EPSILON=.02" "--EPSILON=.04" "--EPSILON=.08"
#do


_common="$_trials $_run $_print $_time $_groups $_interval $_test $_ss"

#$_location/pole $_common $_a=.10 $_e=.00 $_g=.95 $_lambda
#$_location/pole $_common $_a=.30 $_e=.00 $_g=.95 $_lambda
$_location/pole $_common $_a=.50 $_e=.00 $_g=.95 $_l=.70
#$_location/pole $_common $_a=.70 $_e=.00 $_g=.95 $_lambda

#done
