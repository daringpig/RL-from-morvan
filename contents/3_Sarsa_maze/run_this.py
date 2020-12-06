"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable

'''
1. 如果你用的是Q-learning, 你会观看一下在s2上选取哪一个动作会带来最大的奖励, 但是在真正要做决定时,却不一定会选取到那个带来最大奖励的动作, Q-learning 在这一步只是估计了一下接下来的动作值;
2. Sarsa是实践派, 他说到做到, 在s2这一步估算的动作也是接下来要做的动作.
'''
def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # 先走一步
        # RL choose action based on observation
        action = RL.choose_action(str(observation))


        # 在learn的时候，在下一步确实采用了推测出来的action
        # 在迭代的时候，本步的action确实是从上一步得来的
        while True:
            # fresh env
            env.render()

            # 与环境交互，得到下一步的state和reward
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # 查Q-table表，获取下一步预采取的action
            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # 根据当前的s和a得到q_predict，根据下一步的s_和a（以及reward）得到q_target,并进行学习
            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()