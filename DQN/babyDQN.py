import gym
import random
import datetime
import numpy as np
import tensorflow as tf
from collections import deque

#open a tensorflow session
sess = tf.Session()

class DNN(object):
    '''a simple feedforward neural network for function approximation.'''
    def __init__(self, num_classes, layer_sizes, learning_rate):

        layers = len(layer_sizes)
        self.input_x = tf.placeholder(tf.float32, shape=[None, layer_sizes[0]], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.filters = tf.placeholder(tf.int32, shape=[None], name='filters')

        self.cache = {}
        self.cache['A0'] = self.input_x
        
        #full connect layers
        for i in range(layers - 1):
            cur_size, nxt_size = layer_sizes[i], layer_sizes[i + 1]
            with tf.name_scope('FC-' + str(nxt_size)):
                W = tf.Variable(tf.truncated_normal([cur_size, nxt_size], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.0, shape=[nxt_size]), name='b')
                Z = tf.nn.xw_plus_b(self.cache['A' + str(i)], W, b, name='Z')
                self.cache['A' + str(i + 1)] = tf.nn.relu(Z, name='A')

        #regression layer
        with tf.name_scope('regression'):
            W = tf.Variable(tf.truncated_normal([layer_sizes[-1], num_classes], stddev=0.1),name='W')
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.cache['A' + str(layers - 1)], W, b, name='scores')

        #loss
        with tf.name_scope('loss'):
            filters = tf.one_hot(self.filters, num_classes, axis=-1, name='filters')
            Q_values = tf.reduce_sum(tf.multiply(self.scores, filters), axis=1)
            self.tmp = filters
            self.loss = tf.reduce_mean(tf.square(self.input_y - Q_values))

        #optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        sess.run(tf.global_variables_initializer())
        
    def train(self, input_x, input_y, filters):
        feed_dict = {self.input_x:input_x, self.input_y:input_y, self.filters:filters}
        sess.run(self.train_op, feed_dict=feed_dict)
        
    def eval(self, input_x):
        feed_dict = {self.input_x:input_x}
        scores = sess.run(self.scores, feed_dict=feed_dict)
        return scores

        
class DQN(object):
    '''a simple discrete action spaces Deep Q-learning model for several OpenAI Gym`s baby tasks.'''
    def __init__(self, state_dim, num_actions, discount_factor, replay_size, \
                 Qnet_layer_sizes, Qnet_learning_rate, batch_size):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.replay_buffer = deque()
        
        self.Qnet = DNN(num_classes=num_actions, layer_sizes=[state_dim] + Qnet_layer_sizes, \
                        learning_rate=Qnet_learning_rate)
        
    def preceive(self, state, action, reward, next_state, done):
        '''recive the info from the environment'''
        self.replay_buffer.append((state, action, reward, next_state, done))
        #popout earlier replay
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()
        #training the Qnet
        if len(self.replay_buffer) > self.batch_size:
            mini_batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            Q_value = np.max(self.Qnet.eval(next_states), axis=1)

            targets = []
            for i in range(self.batch_size):
                if dones[i]:
                    target = rewards[i] 
                else:
                    target = rewards[i] + self.discount_factor * Q_value[i]
                targets.append(target)

            self.Qnet.train(states, targets, actions)
        
    def best_action(self, state):
        action = np.argmax(self.Qnet.eval([state])[0])
        return action

    def egreedy_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return self.best_action(state)
        
def train(env, agent, episodes, steps, test_episodes, init_epsilon):

    for episode in range(episodes):
        state = env.reset()
        epsilon = init_epsilon * (episodes - episode) / episodes
        for _ in range(steps):
            action = agent.egreedy_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            agent.preceive(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        if episode % 10 == 0:
            finished = test(env, agent, test_episodes, steps, episode)
        if finished:
            print("Mission Complete.")
            break

def test(env, agent, episodes, steps, train_episode):

    total_reward = 0
    for episode in range(episodes):
        state = env.reset()
        for _ in range(steps):
            env.render()
            action = agent.best_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
            
    avg_reward = total_reward / episodes
    time_str = datetime.datetime.now().isoformat()
    print("{} episode: {}  avg_reward: {}".format(time_str, train_episode, avg_reward))
    return avg_reward >= 200

#initialize hyperparameters
FLAG = {}
FLAG['env_name'] = 'CartPole-v0'
FLAG['episodes'] = 1000
FLAG['test_episodes'] = 10
FLAG['steps'] = 300
FLAG['discount_factor'] = 0.9
FLAG['init_epsilon'] = 0.5
FLAG['replay_size'] = 4096
FLAG['layer_sizes'] = [32, 16]
FLAG['learning_rate'] = 0.002
FLAG['batch_size'] = 256

#initialize environment
env = gym.make(FLAG['env_name'])
agent = DQN(state_dim=env.observation_space.shape[0], num_actions=env.action_space.n,\
            discount_factor=FLAG['discount_factor'], replay_size=FLAG['replay_size'],\
            Qnet_layer_sizes=FLAG['layer_sizes'], Qnet_learning_rate=FLAG['learning_rate'],\
            batch_size=FLAG['batch_size'])
#train the agent
train(env, agent, episodes=FLAG['episodes'], steps=FLAG['steps'], \
      test_episodes=FLAG['test_episodes'], init_epsilon=FLAG['init_epsilon'])

sess.close()
