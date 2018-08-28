#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -w -c tgs-salt-identification-challenge

unzip test.zip -d ./test
rm test.zip
unzip train.zip -d ./train
rm train.zip
