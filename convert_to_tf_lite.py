import tensorflow as tf


def main():
    converter = tf.lite.TFLiteConverter.from_frozen_graph("models/yolo_v3.pb", input_arrays=["main_input"], output_arrays=['concat_98'])
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open("models/yolo_v3.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    main()