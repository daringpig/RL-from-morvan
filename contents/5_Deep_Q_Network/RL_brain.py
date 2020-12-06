"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        # 采样的训练样本的个数
        self.memory_size = memory_size
        # 训练时，每一批的样本数
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        # 获取target_net网络的所有参数
        t_params = tf.get_collection('target_net_params')
        # 获取eval_net网络的所有参数
        e_params = tf.get_collection('eval_net_params')
        # 将eval_net网络参数赋值给target_net网络参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 第一处run：初始化网络参数
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # 学习Q-value，通过状态s学习得到Q(s,a)的值
        # evaluate_net类似于Q-learning中的Q-table，用于计算q_predict对应的值
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        # 问题：输入一个batch的数据并进行训练，什么时候训练中止呢？
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # target_net不具体对应Q-table，可以认为是旧的Q-table，可用于计算q_target对应的值
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 更新采样的训练数据集
    def store_transition(self, s, a, r, s_):
        # 上一个episode运行完成后，此memory并没有并清空，off-policy可见一斑
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # s,s_ => (x,y), [a,r] => (a,r), 拼接起来的shape正好为（6,)
        transition = np.hstack((s, [a, r], s_))
        #print("Shape of state: ", s.shape)
        #print("Shape of transition: ", transition.shape)

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 选择action，实现映射s->a
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        # 维度由 (2,) -> (1,2)
        observation = observation[np.newaxis, :]
        # exploitation
        if np.random.uniform() < self.epsilon:
            # 第二处run：查表Q-table，实现映射s->a
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        # exploration
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # 学习Q-table
    def learn(self):
        # 每学习一点的次数，便更新target_net网络的参数
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 第三处run：当需要更新target网路参数时，就更新网络参数
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从与环境交互的训练数据中，采样出batch_size大小的数据
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 第四处run：推理得到q_eval，顺带得到q_next
        # q_next是根据target_net推理得出的，q_eval是根据eval_net推理得出的
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        print("Shape of q_target: ", q_target.shape)
        print("Size of batch: ", self.batch_size)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 从该batch训练数据中获取action的列表
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 从该batch训练数据中获取reward的列表
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """
        # 第五处run：进行训练，更新网络参数，相当于对Q-table进行更新
        # train eval network，输入为批量的状态s和q_target
        # 训练的目的就是最小化q_eval和q_target的差距，在此过程中更新参数
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # 把cost的曲线画出来，以方便后面可视化训练曲线
        self.cost_his.append(self.cost)

        # 随着训练的进行，exploitation的比例越来越小，exploration的比例越来越大，用epsilon_max来进行限制
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        #print("Epsilon in RL_brain:", self.epsilon)
        #print("Epsilon_increment in RL_brain:", self.epsilon_increment)

        # 每学一步，计数器加一
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()