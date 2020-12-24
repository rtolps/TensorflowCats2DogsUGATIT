import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants

def export_from_frozen_graph(frozen_graph_filename,
                             input_node_names=['OneShotIterator'],
                             output_node_names=['generator_B/Tanh'], 
                             output_filename='output.pb',
                             optimize=True):
    tf.reset_default_graph()
    graph_def = tf.GraphDef()

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def.ParseFromString(f.read())
        print("%d ops in original graph." % len(graph_def.node))

        if optimize:
            graph_def = optimize_graph(graph_def,
                                       input_node_names,
                                       output_node_names)
            print("%d ops in optimized graph." % len(graph_def.node))


        # Serialize and write to file
        if output_filename:
            with tf.gfile.GFile(output_filename, "wb") as f:
                f.write(graph_def.SerializeToString())

    return graph_def