"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    # 运行100局游戏
    for episode in range(100):
        # 每一局游戏开始前要重置一下环境
        # initial observation
        observation = env.reset()

        while True:
            # 更新环境
            # fresh env
            env.render()

            # 查Q-table表，完成预测s->a
            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # 与环境交互，得到下一个state和reward，即s_和reward
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # 更新Q-table
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # 向下走一步，即将下一步的state变成当前步的state
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    # 创建env环境对象
    env = Maze()
    # 创建agent智能体对象，agent主要用于执行动作
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()