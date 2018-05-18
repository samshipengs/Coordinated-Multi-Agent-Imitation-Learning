import tensorflow as tf
from tensorflow.python.ops.rnn import _transpose_batch_time
from datetime import datetime
import os


def dynamic_raw_rnn(cell, input_, batch_size, seq_length, horizon, output_dim, policy_number):
    # raw_rnn expects time major inputs as TensorArrays
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_length, clear_after_read=False)
    inputs_ta = inputs_ta.unstack(_transpose_batch_time(input_))  # model_input is the input placeholder

    input_dim = input_.get_shape()[-1].value  # the dimensionality of the input to each time step
    output_dim = output_dim  # the dimensionality of the model's output at each time step

    player_fts = 4
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
            dense = tf.contrib.layers.fully_connected(inputs=cell_output, num_outputs=output_dim)
            emit_output = tf.contrib.layers.dropout(inputs=dense, keep_prob=0.6)
            # create input
            next_input = tf.cond(finished, 
                                 lambda: tf.zeros([batch_size, input_dim], dtype=tf.float32), 
                                 lambda: tf.cond(tf.equal(tf.mod(time, horizon+1), tf.constant(0)),
                                                 lambda: inputs_ta.read(time),
                                                 lambda: tf.concat((inputs_ta.read(time)[:, :policy_number*player_fts],
                                                                    emit_output, 
                                                                    inputs_ta.read(time)[:, policy_number*player_fts+2:]), axis=1)))
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
    
    outputs_ta, last_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = _transpose_batch_time(outputs_ta.stack())
    final_state = last_state

    return outputs, final_state


class SinglePolicy:
    def __init__(self, policy_number, state_size, batch_size, input_dim, output_dim,
                 learning_rate, seq_len, l1_weight_reg = False, logs_path = './train_logs/'):
        tf.reset_default_graph()
        # use training start time as the unique naming
        train_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # hyper-parameters
        self.state_size = state_size
        self.batch_size = batch_size
        self.dimx = input_dim # 4(x,y,vx,vy)*14 + 6 = 62
        self.dimy = output_dim 
        self.learning_rate = learning_rate
        self.seq_len = seq_len 
        self.policy_number = policy_number

        # lstm cells
        lstm1 = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.)
        lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=0.8)

        lstm2 = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.)
        lstm2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, output_keep_prob=0.8)

        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2])

        # input placeholders
        self.h = tf.placeholder(tf.int32, name='horizon')
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.dimx], name = 'train_input')
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.dimy], name = 'train_label')

        # rnn structure
        output, last_states = dynamic_raw_rnn(cell = lstm_cell, 
                                              input_ = self.X,
                                              batch_size = self.batch_size,
                                              seq_length = self.seq_len,
                                              horizon = self.h,
                                              output_dim = self.dimy, 
                                              policy_number=self.policy_number)

        # output as the prediction
        self.pred = tf.identity(output, name='prediction')

        # tensorboard's graph visualization more convenient
        with tf.name_scope('MSEloss'):
            if l1_weight_reg:
                # loss (also add regularization on params)
                tv = tf.trainable_variables()
                # l1 loss
                l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
                regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, tv)
                self.loss = tf.identity(tf.losses.mean_squared_error(self.Y, self.pred) + regularization_cost, name='loss')
            else:
                # no weight loss
                self.loss = tf.identity(tf.losses.mean_squared_error(self.Y, self.pred), name='loss')

        with tf.name_scope('Optimizer'):
            # optimzier
            self.opt = tf.train.AdamOptimizer(self.learning_rate, name='Adam').minimize(self.loss)
    
        # initialize variables
        init = tf.global_variables_initializer()
        # create a summary to monitor cost tensor
        self.train_summary = tf.summary.scalar("TrainMSEloss", self.loss)
        self.valid_summary = tf.summary.scalar("ValidMSEloss", self.loss)
        # # Merge all summaries into a single op
        # merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        # session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # initializing the variables
        self.sess.run(init)
        # op to write logs to Tensorboard
        self.train_writer = tf.summary.FileWriter(logs_path+'/train'+train_time, graph=tf.get_default_graph())
        self.valid_writer = tf.summary.FileWriter(logs_path+'/valid'+train_time, graph=tf.get_default_graph())

    def train(self, train_xi, train_yi, k):
        return self.sess.run([self.pred, self.loss, self.opt, self.train_summary], 
                             feed_dict={self.X: train_xi, self.Y: train_yi, self.h: k})
    
    def validate(self, val_xi, val_yi, k):
        return self.sess.run([self.loss, self.valid_summary], 
                             feed_dict={self.X: val_xi, self.Y: val_yi, self.h: k})

    def save_model(self, models_path):
        # save model
        save_path = models_path + 'policy{0:}'.format(self.policy_number) + '/model'
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        self.saver.save(self.sess, save_path)


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
            self.opt = self.graph.get_operation_by_name('Optimizer/Adam')
            self.pred = self.graph.get_tensor_by_name('prediction:0')
            # self.seq_len = self.graph.get_tensor_by_name('sequence_length:0')
            self.h = self.graph.get_tensor_by_name('horizon:0')

    def forward_pass(self, input_x, h):
        return self.sess.run([self.pred], 
                             feed_dict={self.X: input_x, self.h: h})

    def backward_pass(self, input_x, input_y, h):
        return self.sess.run([self.pred, self.loss, self.opt], 
                            feed_dict={self.X: input_x, self.Y: input_y, self.h: h})