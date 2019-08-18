# tensorflow_lite_converter
- This reposigory has toot sets to generate tensorflow_lite file from ONNX file.
- ONNX file for test execution can be downloaded from [here](https://drive.google.com/open?id=1QzBz4ZO6laC8pZi_-BJpzseX3KiD_e_Q)

# How to use
1. run `convert_onnx_to_pb.py` with ONNX file path. <br> 
With this script, tensorflow graph model is generated. 
1. run `convert_pb_to_tf_lite.py` with pb file path. <br>
With this script tensorflow lite model is generated.

# Reference
- [Converting a Simple Deep Learning Model from PyTorch to TensorFlow
](https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d)