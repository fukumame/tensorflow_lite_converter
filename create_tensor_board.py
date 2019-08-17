import onnx_tf.backend
import onnx
import tensorflow as tf

import numpy as np
from PIL import Image
from onnx_tf.backend import prepare

img_path = "data/IMG_20190812_132137.jpg"
model_path = "yolo_v3.onnx"


def main():
    # 画像の読み込みと加工
    img = Image.open(img_path)
    img = img.resize((320, 320))
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
    arr = arr.transpose(0, 3, 1, 2)

    onnx_model = onnx.load(model_path)
    tf_model = onnx_tf.backend.prepare(onnx_model, device='CPU')
    summary_writer = tf.summary.FileWriter("log", tf_model.graph)
    tf_model.run(arr)
    summary_writer.close()

if __name__ == "__main__":
    main()