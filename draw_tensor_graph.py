import onnx_tf.backend
import onnx
import tensorflow as tf

import numpy as np
from PIL import Image
from onnx_tf.backend import prepare



def main():
    model_path = "models/yolo_v3.onnx"
    arr = np.random.rand(1, 3, 320, 320)

    onnx_model = onnx.load(model_path)
    tf_model = onnx_tf.backend.prepare(onnx_model, device='CPU')
    summary_writer = tf.summary.FileWriter("log", tf_model.graph)
    tf_model.run(arr)
    summary_writer.close()

if __name__ == "__main__":
    main()