#!/bin/bash

set -e
set -x -v

rm -rf test_output/
mkdir test_output

[ "$ELV_MODEL_TEST_GPU_TO_USE" != "" ] || ELV_MODEL_TEST_GPU_TO_USE=0

mkdir -p .cache
mkdir -p .cache-ro

##--volume=$(pwd)/models:/elv/models:ro podman run --rm
#podman run nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE player test/*.jpg
#--volume=$(pwd)/test:/elv/test:ro
#--volume=$(pwd)/test_output:/elv/tags
#--volume=$(pwd)/.cache:/root/.cache --network host --device

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache-ro:/root/.cache-ro:ro --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE player test/*.JPG

ex=$?

cd test_output
find

exit $ex
