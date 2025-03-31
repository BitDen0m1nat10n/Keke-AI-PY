#!/bin/bash


# This file is used by my bash-scripts in this repo to get the correct environment
# To create the correct environment on your machine, follow the steps below.
# If you have to change this file in your setup, please don't push the changes to my repo.


## Step1: Install Conda
# Either:
#   Install conda into '~/nobackup/anaconda_stuff/tmp/':
#       > wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
#       > chmod u+x Anaconda3-2020.02-Linux-x86_64.sh
#       > ./Anaconda3-2020.02-Linux-x86_64.sh
#             Agree To License Agreement?
#                 => yes
#             Path:
#                 => ~/nobackup/anaconda_stuff/tmp/anaconda3
#                 OR choose another path and correct this file as described in the 'ALREADY INSTALLED'-block below
#             Do you wish the installer to initialize Anaconda3 by running conda init?
#                 => no (!THE CLUSTER DOCUMENTATION STATES, THAT YOU MIGHT LOCK YOURSELF OUT OF THE CLUSTER IF YOU AGREE TO THIS!)
# OR: if conda is ALREADY INSTALLED:
#     In the line below: replace '~/nobackup/anaconda_stuff/tmp/anaconda3' by the path to your conda installation.
source ~/nobackup/anaconda_stuff/tmp/anaconda3/etc/profile.d/conda.sh


# Step2: Create the Environment:
# With conda activated (source conda.sh as above):
# > conda create --name KekeAgentOptimizationEnvironment --file PATH/TO/REPO/environment.txt
# You can also give the environment another name, but then you have to change the line below respectively.
conda activate KekeAgentOptimizationEnvironment