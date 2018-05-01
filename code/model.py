import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time


def dynamic_raw_rnn(cell, input_, batch_size, seq_length, horizon, output_dim):
    # raw_rnn expects time major inputs as TensorArrays
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_length)
    inputs_ta = inputs_ta.unstack(_transpose_batch_time(input_))  # model_input is the input placeholder

    input_dim = input_.get_shape()[-1].value  # the dimensionality of the input to each time step
    output_dim = output_dim  # the dimensionality of the model's output at each time step

    def loop_fn(time, cell_output, cell_state, loop_state):
        # check if finished 
        elements_finished = (time >= seq_length)
        finished = tf.reduce_all(elements_finished)
        if cell_output is None:
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            emit_output = tf.zeros([output_dim])
            # create input
            next_input = inputs_ta.read(time)    
        else:
            next_cell_state = cell_state
            # emit_output = cell_output
            # since we want the 2d x, y position output
            emit_output = tf.contrib.layers.fully_connected(inputs=cell_output, num_outputs=output_dim)
            # create input
            next_input = tf.cond(finished, 
                                 lambda: tf.zeros([batch_size, input_dim], dtype=tf.float32), 
                                 lambda: tf.cond(tf.equal(tf.mod(time, horizon+1), tf.constant(0)),
                                                 lambda: inputs_ta.read(time),
                                                 lambda: tf.concat((emit_output, inputs_ta.read(time)[:, 2:]), axis=1)))
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
    
    outputs_ta, last_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = _transpose_batch_time(outputs_ta.stack())
    final_state = last_state

    return outputs, final_state


# https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
class ImportGraph:
    def __init__(self, policy_path, model_name='model', model_path='./models/'):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # reload the network
            saver = tf.train.import_meta_graph(model_path + policy_path + model_name + '.meta')
            # load the parameters
            print(model_path+model_name)
            saver.restore(self.sess, tf.train.latest_checkpoint(model_path + policy_path))

            # now acess things that you want to run
            self.X = self.graph.get_tensor_by_name('train_input:0')
            self.Y = self.graph.get_tensor_by_name('train_label:0')
            self.loss = self.graph.get_tensor_by_name('MSEloss/loss:0')
            self.pred = self.graph.get_tensor_by_name('prediction:0')
            self.seq_len = self.graph.get_tensor_by_name('sequence_length:0')
            self.h = self.graph.get_tensor_by_name('horizon:0')


    def run(self, train_x, train_y, seq_len, h):
        return self.sess.run([self.pred, self.loss], 
                              feed_dict={self.X: train_x, self.Y: train_y, 
                                         self.seq_len: seq_len, self.h: h})
