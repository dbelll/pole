#!/bin/bash
# Testing script for home computer
_trials="--TRIALS=16384"
_time="--TIME_STEPS=131072"
_groups="--AGENT_GROUP_SIZE=1"
_interval="--SHARING_INTERVAL=4"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/darwin/release"

_test="--TEST_INTERVAL=4096 --TEST_REPS=16384"

_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_parms="$_a=.50 $_e=.00 $_l=.70"
_common="$_trials $_run $_print $_time $_groups $_interval $_test $_ss $_parms"

#$_location/pole $_common "--RESTART_INTERVAL=32768"
#$_location/pole $_common "--RESTART_INTERVAL=16384"
#$_location/pole $_common "--RESTART_INTERVAL=8192"
$_location/pole $_common "--RESTART_INTERVAL=4096"
$_location/pole $_common "--RESTART_INTERVAL=2048"
$_location/pole $_common "--RESTART_INTERVAL=1024"
$_location/pole $_common "--RESTART_INTERVAL=512"
$_location/pole $_common "--RESTART_INTERVAL=256"
$_location/pole $_common "--RESTART_INTERVAL=128"






