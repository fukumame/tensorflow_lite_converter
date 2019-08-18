import tensorflow as tf
from PIL import Image
import numpy as np

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def main():
    input_file = 'models/yolo_v3.pb'
    tf_graph = load_pb(input_file)
    sess = tf.Session(graph=tf_graph)

    output_tensor = tf_graph.get_tensor_by_name('concat_98:0')
    input_tensor = tf_graph.get_tensor_by_name('main_input:0')

    arr = np.random.rand(1, 3, 320, 320)
    output = sess.run(output_tensor, feed_dict={input_tensor: arr})
    print(output.shape)


if __name__ == "__main__":
    main()