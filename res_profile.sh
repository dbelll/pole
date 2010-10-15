#!/bin/bash
# Testing script for res computer
#
#  Test restart interavl
#

# turn profiling on
    export CUDA_PROFILE=1
    export CUDA_PROFILE_CONFIG=profile_config
    export CUDA_PROFILE_LOG=pole_profile_log.csv
    export CUDA_PROFILE_CSV=1
    echo "Profiling is on!!!"


_trials="--TRIALS=2048"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"
_restart="--RESTART_INTERVAL=1024"
_test="--TEST_REPS=16384"


_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_parms="$_a=.50 $_e=.00 $_l=.70"


_sharing="--SHARING_INTERVAL=1024"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"


$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=1024


