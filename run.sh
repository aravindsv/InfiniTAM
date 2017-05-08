#!/usr/bin/env bash
# Helper script for development. You should probably not run InfiniTAM using
# this in a serious setting.

dir=$1
echo "Using dataset directory: ${dir}"

cd build
make -j6 && \
./InfiniTAM "${dir}/calib.txt" "${dir}/Frames/%04i.ppm" "${dir}/Frames/%04i.pgm"

