import onnx
from onnx_tf.backend import prepare


def main():
    model_path = "models/yolo_v3.onnx"
    onnx_model = onnx.load(model_path)
    tf_exp = prepare(onnx_model)
    print(tf_exp.tensor_dict)
    tf_exp.export_graph("models/yolo_v3.pb")

if __name__ == "__main__":
    main()