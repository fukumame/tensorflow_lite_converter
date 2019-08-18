import onnx
from onnx_tf.backend import prepare


def main():
    onnx_path = "models/yolo_v3.onnx"
    pb_path = "models/yolo_v3.pb"

    onnx_model = onnx.load(onnx_path)
    tf_exp = prepare(onnx_model, device='CPU')
    print(tf_exp.tensor_dict)
    tf_exp.export_graph(pb_path)

if __name__ == "__main__":
    main()