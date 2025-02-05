#!/bin/bash

if ! ssh-add -l ; then
    echo ssh agent does not have any identities loaded, will not be able to build
    echo add them by running ssh-add on your local machine, or on the remote if you have keys there
    echo you may also need to restart vs code and the remote server for this to work
    exit 1
fi

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
MODEL_PATH=$(yq -r .storage.weights $SCRIPT_PATH/config.yml)
rm -rf $SCRIPT_PATH/weights
cp -r $MODEL_PATH $SCRIPT_PATH/weights

PLAYER_LIST=$(yq -r .storage.player_info $SCRIPT_PATH/config.yml)
rm -rf $SCRIPT_PATH/player_info.json
cp $PLAYER_LIST $SCRIPT_PATH/player_info.json

podman build --format docker -t player . --network host --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK --volume "${SSH_AUTH_SOCK}:${SSH_AUTH_SOCK}"