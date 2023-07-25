#!/bin/bash
xhost +
docker run --gpus all -it --name lpr -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --ipc=host -v /home/erongeray/lpr:/workspace lpr
