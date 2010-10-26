#!/bin/bash
# Testing script for res computer
#
#  Test restart interavl
#

_trials="--TRIALS=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_restart="--RESTART_INTERVAL=2048"
_test="--TEST_REPS=8192 --TEST_INTERVAL=16384"
_sharing="--SHARING_INTERVAL=16384"  # use large value to maximize chunk size
_wgtinfo="--INIT_SHARING_WGT=0.50 --INIT_THETA_MIN=0.0 --INIT_THETA_MAX=1.0"

_time="--TIME_STEPS=524288"

_grpsize="--AGENT_GROUP_SIZE=1"

_a=.02
_e=.00
_l=.70



_parms="--EPSILON=$_e --LAMBDA=$_l --ALPHA=$_a"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common $_grpsize $_time $_test $_restart


