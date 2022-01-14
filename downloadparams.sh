#!/bin/bash

CreateDIR=./pretrained
if [ ! -d $CreateDIR ]; then
  mkdir $CreateDIR
fi

file_id="1bnD4cZC6oKLRx7WG5bbyWmHDhsVRBVzz"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o cifar10.zip
unzip cifar10.zip -d ./pretrained/cifar10
