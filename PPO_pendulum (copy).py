import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from blenv_2 import BlueSkyEnv
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True,help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'a2', 'a_log_p2', 'r', 's_'])

writer = SummaryWriter('./PPO_tensorboard/logs')

class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(6, 128)
        self.b = nn.Linear(128,256)
        # self.c = nn.Linear(256,512)
        # self.d = nn.Linear(512,256)
        self.e = nn.Linear(256,128)
        self.mu_head = nn.Linear(128, 1)
        self.sigma_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.b(x))
        # x = F.relu(self.c(x))
        # x = F.relu(self.d(x))
        x = F.relu(self.e(x))
        mu = F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(6, 128)
        self.b = nn.Linear(128, 256)
        # self.c = nn.Linear(256, 512)
        # self.d = nn.Linear(512, 256)
        self.e = nn.Linear(256, 128)
        self.v_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.b(x))
        # x = F.relu(self.c(x))
        # x = F.relu(self.d(x))
        x = F.relu(self.e(x))
        state_value = self.v_head(x)
        return state_value


class Agent():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 10000, 320

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []
        self.counter = 0

        self.anet2 = ActorNet().float()
        self.cnet2 = CriticNet().float()

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

        self.optimizer_a2 = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c2 = optim.Adam(self.cnet.parameters(), lr=3e-4)

        self.update_num = 0

    def select_action(self, state):
        state1 = torch.from_numpy(state[0]).float().unsqueeze(0)
        state2 = torch.from_numpy(state[1]).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state1)
            (mu2, sigma2) = self.anet2(state2)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-1.0, 1.0)

        dist2 = Normal(mu2, sigma2)
        action2 = dist2.sample()
        action_log_prob2 = dist.log_prob(action2)
        action2 = action2.clamp(-1.0, 1.0)

        return action.item(), action_log_prob.item(),action2.item(), action_log_prob2.item()

    # def get_value(self, state):
    #
    #     state = torch.from_numpy(state).float().unsqueeze(0)
    #     with torch.no_grad():
    #         state_value = self.cnet(state)
    #         state_value2 = self.cnet2(state)
    #     return state_value.item(),state_value2.item()

    def save_param(self):
        torch.save(self.anet.state_dict(), 'param/ppo_anet_params.pkl')
        torch.save(self.cnet.state_dict(), 'param/ppo_cnet_params.pkl')
        torch.save(self.anet2.state_dict(), 'param/ppo_anet2_params.pkl')
        torch.save(self.cnet2.state_dict(), 'param/ppo_cnet2_params.pkl')


    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s[0] for t in self.buffer], dtype=torch.float)
        s2 = torch.tensor([t.s[1] for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        a2 = torch.tensor([t.a2 for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r[0] for t in self.buffer], dtype=torch.float).view(-1, 1)
        r2 = torch.tensor([t.r[1] for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_[0] for t in self.buffer], dtype=torch.float)
        s_2 = torch.tensor([t.s_[1] for t in self.buffer], dtype=torch.float)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        old_action_log_probs2 = torch.tensor(
            [t.a_log_p2 for t in self.buffer], dtype=torch.float).view(-1, 1)

        r = (r - r.mean()) / (r.std() + 1e-5)
        r2 = (r2 - r2.mean()) / (r2.std() + 1e-5)
        with torch.no_grad():
            target_v = r + args.gamma * self.cnet(s_)
            target_v2 = r2 + args.gamma * self.cnet2(s_2)

        adv = (target_v - self.cnet(s)).detach()
        adv2 = (target_v2 - self.cnet2(s2)).detach()


        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.anet(s[index])
                (mu2, sigma2) = self.anet2(s2[index])
                dist = Normal(mu, sigma)
                dist2 = Normal(mu2,sigma2)
                action_log_probs = dist.log_prob(a[index])
                action_log_probs2 = dist2.log_prob(a2[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])
                ratio2 = torch.exp(action_log_probs2 - old_action_log_probs2[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]

                surr3 = ratio2 * adv2[index]
                surr4 = torch.clamp(ratio2, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv2[index]

                action_loss = -torch.min(surr1, surr2).mean()
                action_loss2 = -torch.min(surr3, surr4).mean()

                writer.add_scalar('loss/action_loss', action_loss, self.update_num)
                writer.add_scalar('loss/action_loss2', action_loss2, self.update_num)



                self.optimizer_a2.zero_grad()
                action_loss2.backward()
                nn.utils.clip_grad_norm_(self.anet2.parameters(), self.max_grad_norm)
                self.optimizer_a2.step()

                value_loss2 = F.smooth_l1_loss(self.cnet2(s2[index]), target_v2[index])
                self.optimizer_c2.zero_grad()
                value_loss2.backward()
                nn.utils.clip_grad_norm_(self.cnet2.parameters(), self.max_grad_norm)
                self.optimizer_c2.step()


                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

                writer.add_scalar('loss/value_loss', value_loss, self.update_num)
                writer.add_scalar('loss/value_loss2', value_loss2, self.update_num)
                self.update_num += 1

        del self.buffer[:]


def main():
    env = BlueSkyEnv('TEST')

    agent = Agent()

    training_records = []
    running_reward = -400
    all_rew = []
    for i_ep in range(1000000):
        x = []
        y = []
        x1 = []
        y1 = []
        score = 0
        state = env.reset()

        for t in range(3000):
            action, action_log_prob, action2, action_log_prob2 = agent.select_action(state)

            # action2 = -1 if state[1][5]>0 else 1


            state_, reward, done, _ = env.step([action,action2])
            x.append(state_[0][0])
            y.append(state_[0][1])
            x1.append(state_[1][0])
            y1.append(state_[1][1])

            rew = [(reward[0] + 8) / 8, (reward[1] + 8) / 8]
            if agent.store(Transition(state, action, action_log_prob, action2, action_log_prob2, rew, state_)):
                agent.update()
            score += reward[0]
            state = state_

        all_rew.append(score)
        writer.add_scalar('loss/action_loss2', score, i_ep)

        plt.cla()
        plt.subplot(211)
        plt.cla()
        plt.plot(x, y, 'r')
        plt.plot(x1,y1,'b')
        if x != []:
            plt.plot(x[0], y[0], 'or')
            plt.plot(x1[0], y1[0], 'ob')
        plt.axis('scaled')
        # plt.pause(0.000000000001)
        plt.subplot(212)
        plt.plot(all_rew)
        plt.pause(0.00000001)
        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % 10 == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))


        # if i_ep % args.log_interval == 0:
        #     plt.plot(x, y, 'r')
        #     plt.plot(x[0], y[0], 'or')
        #     plt.plot(env.x, env.y, 'ob')
        #     plt.axis('equal')
        #     plt.show()

        # if running_reward > -200:
        #     print("Solved! Moving average score is now {}!".format(running_reward))
        #     env.close()
        #     #agent.save_param()
        #     with open('log/ppo_training_records.pkl', 'wb') as f:
        #         pickle.dump(training_records, f)
        #     break

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("img/ppo.png")
    plt.show()


if __name__ == '__main__':
    main()
