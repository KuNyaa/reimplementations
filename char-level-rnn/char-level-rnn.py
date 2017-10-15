import nltk
import datetime
import numpy as np
import tensorflow as tf
from nltk.corpus import gutenberg
from string import ascii_uppercase as uppercase

class RNN(object):

    def __init__(self, vocab_size, hidden_size, learning_rate):

        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        batch_size = tf.shape(self.input_x)[0]
        
        cell = tf.contrib.rnn.GRUCell(hidden_size)
        init_state = cell.zero_state(batch_size, tf.float32)
        
        input = tf.one_hot(self.input_x, vocab_size)
        seq_lengths = tf.reduce_sum(tf.reduce_max(tf.sign(input), 2), 1)
        in_state = tf.placeholder_with_default(init_state, shape=[None, hidden_size],
                                               name='in_state')
        output, out_state = tf.nn.dynamic_rnn(cell, input, seq_lengths, in_state)
        self.in_state, self.out_state = in_state, out_state

        output_dropouted = tf.nn.dropout(output, self.dropout_keep_prob)
        scores = tf.contrib.layers.fully_connected(output_dropouted, vocab_size, None)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores[:, :-1],
                                                            labels=input[:, 1:]))
        self.temprature = tf.placeholder(tf.float32, [])
        self.sample = tf.multinomial(tf.exp(scores[:, -1] / self.temprature), 1)[:, 0]
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.global_setp = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_setp)


    def train(self, sess, input_x, dropout_keep_prob):
        feed_dict = {self.input_x:input_x, self.dropout_keep_prob:dropout_keep_prob}
        sess.run(self.train_op, feed_dict=feed_dict)
        
    def step(self, sess):
        return sess.run(self.global_setp)

    def eval_loss(self, sess, input_x):
        feed_dict = {self.input_x:input_x, self.dropout_keep_prob:1.0}
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def generate(self, sess, start, genr_length, temprature):
        sentence = start
        state = None
        for _ in range(genr_length):
            batch = [[sentence[-1]]] 
            feed_dict = {self.input_x:batch, self.temprature:temprature, self.dropout_keep_prob:1.0}
            if state is not None:
                feed_dict.update({self.in_state:state})
            id, state = sess.run([self.sample, self.out_state], feed_dict=feed_dict)
            sentence.append(id[0])
        return sentence


def encode(text, vocab):
    return [vocab.index(char) + 1 for char in text]


def decode(text, vocab):
    return ''.join([vocab[id - 1] for id in text])

def get_batches(text, seq_length, batch_size):

    step_size = 4
    full_batch = [text[start: start + seq_length] for start in range(0, len(text) - seq_length + 1, step_size)]
    size = len(full_batch)
    num_blocks = size // batch_size

    mini_batches = []
    for i in range(num_blocks):
        start, end = i * batch_size, (i + 1) * batch_size
        mini_batches.append(full_batch[start:end])

    if size % batch_size != 0:
        mini_batches.append(full_batch[num_blocks * batch_size:size])

    return full_batch, mini_batches

def generation(sess, rnn, vocab, start, genr_length, temprature):

    start_encoded = encode(start, vocab)
    sentence = rnn.generate(sess, start_encoded, genr_length, temprature)
    sentence_decoded = decode(sentence, vocab)

    return sentence_decoded
        

def training(sess, rnn, text, vocab, seq_length, epochs, batch_size , genr_length, temprature, dropout_keep_prob):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())

    full_batch, mini_batches = get_batches(text, seq_length, batch_size)
    print(len(full_batch), len(mini_batches))
    for _ in range(epochs):
        for batch in mini_batches:
            rnn.train(sess, batch, dropout_keep_prob)
            global_step = rnn.step(sess)
            if global_step % 2000 == 0:
                saver.save(sess, './checkpoints/checkpoint', global_step) 
                time_str = datetime.datetime.now().isoformat()
                loss = rnn.eval_loss(sess, full_batch)
                start_char = uppercase[np.random.randint(0, 26)]
                genr_sentence = generation(sess, rnn, vocab, start_char, genr_length, temprature)
                print("{}: step:{}  loss: {}".format(time_str, global_step, loss))
                print("Generate sample: {}".format(genr_sentence))


text = gutenberg.raw('blake-poems.txt')


vocab = ''.join(sorted(set(list(text))))

epochs = 200
hidden_size = 256
learning_rate = 0.001
dropout_keep_prob = 0.5
batch_size = 128
temprature = 0.6
seq_length = 64
genr_length = 300
vocab_size = len(vocab)

rnn = RNN(vocab_size=vocab_size, hidden_size=hidden_size, learning_rate=learning_rate)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

text = encode(text, vocab)
training(sess, rnn, text, vocab, seq_length, epochs, batch_size, 100, temprature, dropout_keep_prob)
'''
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./checkpoints')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
'''
file = open("generate.txt", "w")  
for start in uppercase:
    file.write("\n\nCase #" + start + ":\n")
    file.write(generation(sess, rnn, vocab, start, genr_length, temprature))

sess.close()
