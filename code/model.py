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
