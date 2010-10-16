#!/bin/bash
# Testing script for res computer
#
#  Test restart interavl
#

_trials="--TRIALS=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"
_test="--TEST_REPS=16384"

_restart="--RESTART_INTERVAL=1024"
_test="--TEST_INTERVAL=8192"
_time="--TIME_STEPS=524288"
_grpsize="--AGENT_GROUP_SIZE=1"

_e="--EPSILON"
_l="--LAMBDA"

_parms="$_e=.00 $_l=.70"

_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"


for _a in .01 .02
do

$_location/pole $_common $_grpsize $_time $_test $_restart --ALPHA=$_a
 
done