import tensorflow as tf


def main():
    input_path = "models/yolo_v3.pb"
    output_path = "models/yolo_v3.tflite"

    input_name = "main_input"
    output_name = 'concat_98'

    converter = tf.lite.TFLiteConverter.from_frozen_graph(input_path, input_arrays=[input_name], output_arrays=[output_name])
    tflite_model = converter.convert()
    open(output_path, "wb").write(tflite_model)


if __name__ == "__main__":
    main()