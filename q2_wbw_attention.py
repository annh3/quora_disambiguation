import os
import time
import tensorflow as tf
import cPickle
import numpy as np
import parser_orig

from model import Model
#from q2_initialization import xavier_weight_init
from general_utils import Progbar, get_minibatches

#from parser import load_and_preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    #n_features = 36
    max_len = 50
    n_classes = 2
    dropout = .3 # was .5
    embed_size = 50
    hidden_size = 200
    batch_size = 45
    n_epochs = 10
    lr = 0.0005 # was .001
    beta = 0.0001
    max_grad_norm = 5
    cell = "lstm"


class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE
        self.input_placeholder1 = tf.placeholder(tf.int32, shape=(None, self.config.max_len))
        self.input_placeholder2 = tf.placeholder(tf.int32, shape=(None, self.config.max_len))
        self.padding_placeholder1 = tf.placeholder(tf.int32, shape = (None,))
        self.padding_placeholder2 = tf.placeholder(tf.int32, shape = (None,))

        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,self.config.n_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch1, inputs_batch2, inputs_size1,
                         inputs_size2, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        feed_dict = {self.input_placeholder1: inputs_batch1, self.input_placeholder2: inputs_batch2,
                     self.padding_placeholder1: inputs_size1, self.padding_placeholder2: inputs_size2}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        embeddings_tbl = tf.Variable(self.pretrained_embeddings,dtype=np.float32)
        embeddings1 = tf.nn.embedding_lookup(embeddings_tbl, self.input_placeholder1)
        embeddings2 = tf.nn.embedding_lookup(embeddings_tbl, self.input_placeholder2)
        embeddings1 = tf.reshape(embeddings1, shape = [-1, self.config.max_len, self.config.embed_size])
        embeddings2 = tf.reshape(embeddings2, shape = [-1, self.config.max_len, self.config.embed_size])
        ### END YOUR CODE
        return embeddings1, embeddings2

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features*embed_size, hidden_size)
                    b1: (hidden_size,)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)
        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x1, x2 = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        # if self.config.cell == "rnn":
        #     cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        # elif self.config.cell == "gru":
        #     cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        # elif self.config.cell == "lstm":
        #     cell = tf.nn.rnn_cell.LSTMCell(Config.hidden_size)
        # else:
        #     raise ValueError("Unsuppported cell type: " + self.config.cell)


        with tf.variable_scope("RNN"):
            cell = tf.nn.rnn_cell.LSTMCell(Config.hidden_size)
            output1, state1 = tf.nn.dynamic_rnn(cell, x1, self.padding_placeholder1, dtype = tf.float32,time_major=False)
            tf.get_variable_scope().reuse_variables()
            output2, state2 = tf.nn.dynamic_rnn(cell, x2, self.padding_placeholder2, initial_state = state1, dtype = tf.float32,time_major=False)
            tf.get_variable_scope().reuse_variables()
        ### YOUR CODE HERE
        #state1 = tf.sigmoid(state1)
        #state2 = tf.sigmoid(state2)
        #output has dimension max_len,bs, hidden_size
        xavier = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        inf = 1e9
        #Y = tf.transpose(output1, [1,0,2])
        Y = tf.reshape(output1, [-1, self.config.hidden_size]) #concatenate batches

        W_y = tf.get_variable("W_y", [self.config.hidden_size, self.config.hidden_size], initializer=xavier,dtype=np.float32)
        W_h = tf.get_variable("W_h", [self.config.hidden_size, self.config.hidden_size], initializer=xavier,dtype=np.float32)
        W_r = tf.get_variable("W_r", [self.config.hidden_size, self.config.hidden_size], initializer=xavier,dtype=np.float32)
        W_p = tf.get_variable("W_p", [self.config.hidden_size, self.config.hidden_size], initializer=xavier,dtype=np.float32)
        W_x = tf.get_variable("W_x", [self.config.hidden_size, self.config.hidden_size], initializer=xavier,dtype=np.float32)

        w = tf.get_variable("w", [self.config.hidden_size,1], initializer=xavier,dtype=np.float32)

        Wts = []
        for time in range(self.config.max_len):
            W_t = tf.get_variable("W_" + str(time), [self.config.hidden_size, self.config.hidden_size], initializer=xavier,dtype=np.float32)
            Wts.append(W_t)
        mask = tf.sequence_mask(self.padding_placeholder1, self.config.max_len)
        r = tf.zeros([1, self.config.hidden_size], dtype = tf.float32)
        for t in range(self.config.max_len):
            term1 = tf.matmul(Y, W_y)
            term1 = tf.reshape(term1, [-1, self.config.max_len, self.config.hidden_size])
            # at this point term1 is of shape (batch_size, max_len, hidden_size)
            #term1 = tf.transpose(term1, [0,2,1])
            #print("wow1")

            #print(output1.get_shape())
            term2 = tf.matmul(tf.transpose(output1, [1,0,2])[t], W_h) + tf.matmul(r, W_r)# batch size, hidden size

            #print ("wow2")
            expanded = tf.expand_dims(term2, 1) #batch size,1, hidden size
            term2 = tf.tile(expanded, tf.pack([1, self.config.max_len, 1]))  #batch size, max_len, hidden_size
            M = tf.nn.tanh(term1 + term2)

            #print("wow3")
            # MASK THIS
            presoft = tf.reshape(tf.matmul(tf.reshape(M, [-1, self.config.hidden_size]), w), [-1, self.config.max_len])
            masked = tf.select(mask, presoft, -inf * tf.ones_like(presoft))
            alpha = tf.nn.softmax(masked)
            # batch size, max len
            #print("wow4")

            last = tf.nn.tanh(tf.matmul(r, Wts[t]))
            r = tf.squeeze(tf.batch_matmul(tf.expand_dims(alpha, 1), output1)) + last
        
            
        h_star = tf.nn.tanh(tf.matmul(r, W_p) + tf.matmul(state2[1], W_x)) 
        #print("wow6")
            
        #print("wow7")

        #state = tf.transpose(state, perm=[1, 0, 2])
        #state = tf.slice(state, [1, 0, 0], 2)
        # state = tf.gather_nd(state, [[1]])
        # state = tf.reshape(state, [-1, 50])
        #h = tf.nn.tanh(tf.matmul(state, U) + b1)
        h_drop = tf.nn.dropout(h_star, dropout_rate)
        W_f = tf.get_variable("W_f", [self.config.hidden_size, self.config.n_classes], initializer=xavier,dtype=np.float32)
        self.regularizer = tf.nn.l2_loss(W_y) + tf.nn.l2_loss(W_h) + tf.nn.l2_loss(W_r) + tf.nn.l2_loss(W_p) + tf.nn.l2_loss(W_x) + tf.nn.l2_loss(W_f)
        for idx in range(self.config.max_len):
            self.regularizer += tf.nn.l2_loss(Wts[idx])
        self.regularizer += tf.nn.l2_loss(w)
        ### END YOUR CODE
        pred = tf.matmul(h_drop, W_f)
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        regularization = self.config.beta * self.regularizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)) 
        return tf.reduce_mean(loss + regularization)

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        ### YOUR CODE HERE (~6-10 lines)
        gradients = optimizer.compute_gradients(loss)
        grads = zip(*gradients)
        #if self.config.clip_gradients:
        #grads[0], _ = tf.clip_by_global_norm(grads[0], self.config.max_grad_norm)
        self.grad_norm = tf.global_norm(grads[0])
        #train_op = optimizer.apply_gradients(zip(*grads))
        # - Remember to clip gradients only if self.config.clip_gradients
        # is True.
        # - Remember to set self.grad_norm

        ### END YOUR CODE

        assert self.grad_norm is not None, "grad_norm was not set properly!"
        #return train_op
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def train_on_batch(self, sess, inputs_batch1, inputs_batch2, inputs_size1, inputs_size2, labels_batch):
        feed = self.create_feed_dict(inputs_batch1=inputs_batch1, inputs_batch2=inputs_batch2, 
                                    inputs_size1=inputs_size1, inputs_size2=inputs_size2, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss, pred, norm, reg = sess.run([self.train_op, self.loss, self.pred, self.grad_norm, self.regularizer], feed_dict=feed)
        return loss, pred, norm, reg

    def predict_on_batch(self, sess, inputs_batch1, inputs_batch2, inputs_size1, inputs_size2):
        feed = self.create_feed_dict(inputs_batch1=inputs_batch1, inputs_batch2=inputs_batch2, 
                                    inputs_size1=inputs_size1, inputs_size2=inputs_size2)
        print tf.shape(self.pred)
        #predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        predictions = sess.run(self.pred, feed_dict=feed)

        return predictions

    def run_epoch(self, sess, train_examples1, dev_set1, train_examples2, dev_set2, train_sizes, dev_sizes):
        #prog = Progbar(target=1 + (int)(len(train_examples1) / self.config.batch_size))
        print("NEW EPOCH\n")
        ##############################
        # train_x1 = []
        # train_x2 = []
        # pad_size1 = []
        # pad_size2 = []
        # train_y = []
        # for i in range(45):
        #     train_x1.append(train_examples1[i][0])
        #     train_x2.append(train_examples2[i][0])
        #     pad_size1.append(train_sizes[0][i])
        #     pad_size2.append(train_sizes[1][i])
        #     train_y.append(train_examples1[i][1])
        # print "finished loading train data"
        # for _ in range(100):
        #     loss, pred, norm, reg = self.train_on_batch(sess, train_x1, train_x2, pad_size1, pad_size2, train_y)
        #     print("TRAINING PREDICTIONS: ")
        #     print(pred)
        #     # print("labels: ")
        #     # print(train_y)
        #     tcorrect = 0
        #     for idx in range(len(pred)):
        #         # print "trainy"
        #         # print train_y[idx]
        #         if train_y[idx] == [0,1]:
        #             lbl = 1
        #         else:
        #             lbl = 0
        #         # print "label"
        #         # print lbl
        #         #print pred[idx][0]
        #         #print pred[idx][1]

        #         if pred[idx][0] > pred[idx][1]:
        #             p = 0
        #         else:
        #             p = 1
        #         # print "p"
        #         # print p
        #         # print "-----------\n"
        #         if lbl == p:
        #             tcorrect += 1
            
        #     # print(train_x)
        #     # print("labels: ")
        #     # print(train_y)
        #     #prog.update(i + 1, [("train loss", loss)])
        #     if _ % 2 == 0:
        #         print("train loss: ")
        #         print loss
        #         print("grad norm: ")
        #         print norm
        #         print("train accuracy: ")
        #         print(float(tcorrect)/float(len(pred)))
                
        ################################
        
        for i, (obj, train_y) in enumerate(parser_orig.minibatches(train_examples1, train_examples2, train_sizes, self.config.batch_size)):
            #print(len(train_x[1005]))
            
            train_x1 = [obj[j][0] for j in range(len(obj))]
            train_x2 = [obj[j][1] for j in range(len(obj))]
            pad_size1 = [obj[j][2] for j in range(len(obj))]
            pad_size2 = [obj[j][3] for j in range(len(obj))]
            # print "random example: "
            # print train_x1[0]
            # print train_x2[0]
            # print pad_size1[0]
            # print pad_size2[0]
            loss, pred, norm, reg = self.train_on_batch(sess, train_x1, train_x2, pad_size1, pad_size2, train_y)
            
            tcorrect = 0
            for idx in range(len(pred)):
                # print "trainy"
                # print train_y[idx]
                if train_y[idx] == [0,1]:
                    lbl = 1
                else:
                    lbl = 0
                # print "label"
                # print lbl
                #print pred[idx][0]
                #print pred[idx][1]

                if pred[idx][0] > pred[idx][1]:
                    p = 0
                else:
                    p = 1
                # print "p"
                # print p
                # print "-----------\n"
                if lbl == p:
                    tcorrect += 1
            # print(train_x)
            # print(train_y)
            #prog.update(i + 1, [("train loss", loss)])
            if i % 10 == 0:
                print("TRAINING PREDICTIONS: ")
                print(pred)

                print("train loss: ")
                print loss

                print("train acc: ")
                print float(tcorrect)/float(len(pred))

                print("grad norm: ")
                print norm

                print("regularization: ")
                print reg
                
        

        print "Evaluating on dev set"
        first_sentences = []
        second_sentences = []
        first_sizes = []
        second_sizes = []
        labels_of_dev = []
        for i in range(len(dev_set1)):
            first_sentences.append(dev_set1[i][0])
            second_sentences.append(dev_set2[i][0])
            first_sizes.append(dev_sizes[0][i])
            second_sizes.append(dev_sizes[1][i])
            labels_of_dev.append(dev_set1[i][1])

        predictions = self.predict_on_batch(sess, first_sentences, second_sentences, first_sizes, second_sizes)
        correct = 0
        print predictions
        for i in range(len(predictions)):

            if labels_of_dev[i] == [0,1]:
                lbl = 1
            else:
                lbl = 0
            if (predictions[i][0] > predictions[i][1]):
                predictd_lbl = 0
            else:
                predictd_lbl = 1
            # print("actual: ")
            # print lbl
            # print("predicted: ")
            # print predictd_lbl
            if predictd_lbl == lbl:
                correct = correct + 1
        accuracy = float(correct) / float(len(predictions))
        print "dev accuracy: ", accuracy
        return accuracy

    def fit(self, sess, saver, train_examples1, dev_set1, train_examples2, dev_set2, train_sizes, dev_sizes):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_UAS = self.run_epoch(sess, train_examples1, dev_set1, train_examples2, dev_set2, train_sizes, dev_sizes)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print "New best dev UAS! Saving model in ./data/parser_attention.weights"
                    saver.save(sess, './data/parser_attention.weights')
            print

    # def build(self):
    #     self.add_placeholders()
    #     self.pred = self.add_prediction_op()
    #     self.loss = self.add_loss_op(self.pred)
    #     self.train_op = self.add_training_op(self.loss)

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    print "hello 1"
    embeddings, train_examples1, dev_set1, test_set1, train_examples2, dev_set2, test_set2, train_sizes, dev_sizes, test_sizes = parser_orig.load_and_preprocess_data(True, config.max_len)
    # print "hello 2"
    # print "embeddings"
    # print np.array(embeddings)
    # print "train_examples"
    # print np.array(train_examples)
    # print "dev_set"
    # print np.array(dev_set)
    # print "test_set"
    # print np.array(test_set)

    print(embeddings)
    print("-------------\n")
    #for x in train_examples:
     #   print(x)
    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        model = ParserModel(config, embeddings)
        #parser.model = model
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            #parser.session = session
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="

            model.fit(session, saver, train_examples1, dev_set1, train_examples2, dev_set2, train_sizes, dev_sizes)

            # if not debug:
            #     print 80 * "="
            #     print "TESTING"
            #     print 80 * "="
            #     print "Restoring the best model weights found on the dev set"
            #     saver.restore(session, './data/weights/parser.weights')
            #     print "Final evaluation on test set",
            #     UAS, dependencies = parser.parse(test_set)
            #     print "- test UAS: {:.2f}".format(UAS * 100.0)
            #     print "Writing predictions"
            #     with open('q2_test.predicted.pkl', 'w') as f:
            #         cPickle.dump(dependencies, f, -1)
            #     print "Done!"

if __name__ == '__main__':
    main()


