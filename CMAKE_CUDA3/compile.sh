#!/bin/bash
# nvcc hello.cu -o HELLO -arch=sm_61 -lcudadevrt -rdc=true
nvcc hello.cu -o HELLO -arch=sm_61 -lcudadevrt --relocatable-device-code=true
