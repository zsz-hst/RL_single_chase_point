import filter_env
from ddpg import *
import gc
import matplotlib.pyplot as plt
gc.enable()

ENV_NAME = 'InvertedPendulum-v1'
EPISODES = 100000
TEST = 3

from blenv import *

def main():
    env = BlueSkyEnv('TEST')
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in range(EPISODES):

        state = env.reset()
        #print "episode:",episode
        # Train
        rew = 0
        for step in range(env.max_step):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            rew += reward
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        print(rew)
        # Testing:
        if episode % 20 == 0 and episode > 20:
            total_reward = 0
            for i in range(TEST):
                x = []
                y = []
                state = env.reset()

                for j in range(env.max_step):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    x.append(state[0])
                    y.append(state[1])
                    total_reward += reward
                    if done:
                        break
                if x != []:

                    plt.plot(x, y, 'r')
                    plt.plot(x[0], y[0], 'or')
                    plt.plot(env.x, env.y, 'ob')
                    plt.axis('equal')
                    plt.show()

            ave_reward = total_reward/TEST
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)

    #env.monitor.close()

if __name__ == '__main__':
    main()
