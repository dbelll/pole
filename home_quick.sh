#!/bin/bash
# Testing script for home computer
_trials="--TRIALS=2048"
_time="--TIME_STEPS=65536"
_groups="--AGENT_GROUP_SIZE=8"
_interval="--SHARING_INTERVAL=512"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/darwin/release"

_test="--TEST_INTERVAL=1024 --TEST_REPS=16384"
_restart="--RESTART_INTERVAL=1024"

_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_common="$_trials $_run $_print $_time $_groups $_interval $_test $_ss"

$_location/pole $_common $_a=.10 $_e=.00 $_g=.95 $_l=.70





