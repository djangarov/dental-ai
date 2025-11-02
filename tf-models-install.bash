#!/bin/bash

rm -rf ./storage/tf-models
git clone https://github.com/tensorflow/models.git storage/tf-models
cd storage/tf-models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
uv pip install .