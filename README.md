# Gridworld Algorithms

This repository contains reusable modules for solving 2-D toy gridworld
problem. Also, with the modules you get standardized environment packaged
with docker and executed using docker-compose. 

## Requirements:
`docker`

## Instructions for usage

1. Build docker image
   
   `./run.sh build-image`

2. Start the container

   `./run.sh start`

3. To get output for problems:

    `./run.sh problem_one`

    `./run.sh problem_two`

    `./run.sh problem_three`
    
    After you run problem three you can connect to http://localhost:6006 to view Tensorboard results