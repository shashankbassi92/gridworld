#!/bin/bash
set -e

IMAGE="grid:latest"
SERVICE="gridworld"



function start() {
    docker-compose up -d --force-recreate
    sleep 3
    docker-compose ps
}

function stop() {
    docker-compose down
}

function build_image() {
    local current_uid=$1
    echo "Building Gridworld image..."
    docker build -t ${IMAGE} .
}

function problem_one() {
    docker-compose exec ${SERVICE} sh -c "python /hw_gridworld/src/gridworld_value_iteration.py"
}

function problem_two() {
    docker-compose exec ${SERVICE} sh -c "python /hw_gridworld/src/gridworld_monte_carlo.py"
}

function problem_three() {
    docker-compose exec ${SERVICE} sh -c "python /hw_gridworld/src/gridworld_dqn.py"
}

function print_usage() {
    cat <<EOF
Usage:

-- Options:

    `basename $0` build-image
        builds docker image with environment for running grid world
    `basename $0` start
        starts container for grid world image
    `basename $0` restart
        restarts container
    `basename $0` stop
        stops service
    `basename $0` problem_one
        runs script for first RL problem on value iteration
    `basename $0` problem_two
        runs script for second RL problem on monte carlo
    `basename $0` problem_three
        runs script for third RL problem on double DQN

EOF
}

#------------------------------------------------------------

case "$1" in
    "start")
        start
        ;;
    "stop")
        stop
        ;;
    "build-image")
        build_image
        ;;
    "restart")
        stop
        start
        ;;
    "problem_one")
        problem_one
        ;;
    "problem_two")
        problem_two
        ;;
    "problem_three")
        problem_three
        ;;
    *)
        echo "Unknown option <$1>. Please tell me what to do :/"
        print_usage
        exit 1
        ;;
esac