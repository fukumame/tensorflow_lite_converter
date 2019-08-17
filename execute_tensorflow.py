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

tf_graph = load_pb('models/yolo_v3.pb')
sess = tf.Session(graph=tf_graph)

output_tensor = tf_graph.get_tensor_by_name('concat_98:0')
input_tensor = tf_graph.get_tensor_by_name('main_input:0')

img_path = "data/IMG_20190812_132137.jpg"
img = Image.open(img_path)
img = img.resize((320, 320))
arr = np.asarray(img, dtype=np.float32)[np.newaxis, :, :, :]
arr = arr.transpose(0, 3, 1, 2)

output = sess.run(output_tensor, feed_dict={input_tensor: arr})
print(output.shape)