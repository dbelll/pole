#!/bin/bash
# Baseline script for res computer
#
_trials="--TRIALS=4096"
_time="--TIME_STEPS=131072"
_groups="--AGENT_GROUP_SIZE=1"
_interval="--SHARING_INTERVAL=4"
_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_INTERVAL=16384 --TEST_REPS=16384"


_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_common="$_trials $_time $_groups $_interval $_run $_test"

$_location/pole $_common $_a=.50 $_e=.00 $_g=.95 $_l=.70

