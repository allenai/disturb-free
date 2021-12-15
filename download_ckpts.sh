#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

# Download, Unzip, and Remove zip
if [ "$1" = "armpointnav-disturb-free-2022" ]
then
    echo "Downloading pretrained ArmPointNav disturb-free models..."
    wget <>
    tar -xf <> && rm <>
    echo "saved folder: armpointnav-disturb-free-2022"
else
    echo "Failed: Usage download_navigation_model_ckpts.sh armpointnav-disturb-free-2022"
    exit 1
fi
