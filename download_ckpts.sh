#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

# Download, Unzip, and Remove zip
if [ "$1" = "armpointnav-disturb-free-2021" ]
then
    echo "Downloading pretrained ArmPointNav disturb-free models..."
    wget https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/armpointnav/armpointnav-2021.tar.gz
    tar -xf armpointnav-2021.tar.gz && rm armpointnav-2021.tar.gz
    mv armpointnav armpointnav-disturb-free-2021 && mv armpointnav-disturb-free-2021 pretrained_model_ckpts/
    echo "saved folder: pretrained_model_ckpts/armpointnav-disturb-free-2021"
else
    echo "Failed: Usage download_navigation_model_ckpts.sh armpointnav-disturb-free-2021"
    exit 1
fi
