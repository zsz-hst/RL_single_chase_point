import bluesky as bs
import bluesky.core as core
import gym
from gym import spaces
import re
import sys
from bluesky.core.simtime import timed_function, preupdate,step,update
import numpy as np
import math


from opendis.DataOutputStream import DataOutputStream
from opendis.dis7 import EntityStatePdu
from opendis.RangeCoordinates import GPS
from opendis.HeadingPitchRollToEuler import HeadingPitchRollToEuler

from opendis.dis_sender import env_send_dis
from math import radians

import time

import random
import math

kts = 0.514444  # m/s  of 1 knot
# 强化学习环境常量
ac_number = 2
MAX_STEP = 3000

# 用于坐标转化
pi = 3.14159265358979323846
R = 6371000

# 攻击范围
attack_radius = 3  # 公里
attack_angle_p = 40  # 度
attack_angle_v = 30  # 度


class BlueSkyEnv(gym.Env):
    def __init__(self, filename):
        bs.init()
        self.max_step = MAX_STEP  # 设置每回合的最大步数
        self.reward = 0  # 初始化奖励
        self.filename = filename

        self._set_action_space()  # 初始化动作空间
        self._set_observation_space()  # 初始化状态空间

        self.step_number = 0  # 初始化回合数

        self.n = ac_number

        self.num_dis = 0

    def reset(self):  # resetbl并根据文件创建飞机

        # self.x = random.uniform(38.33, 38.66)
        # self.y = random.uniform(105.60, 105.95)
        self.x = 38.49
        self.y = 105.3

        xy_tar = MillierToXY(self.x, self.y)
        self.x = (xy_tar[0] - self.observation_space[0]) / self.xlen
        self.y = (xy_tar[1] - self.observation_space[1]) / self.xlen


        bs.traf.reset()  # 初始化bluesky

        actype = 'f16'

        number, lat, lon, hdg, alt, tas = self._get_initstate()

        length = len(lat)

        for i in range(length):  # 创建飞机
            bs.traf.cre(actype=actype, acalt=alt[i], aclat=lat[i], aclon=lon[i],
                        achdg=hdg[i], acid='000%s' % number[i], acspd=tas[i] * kts)

        obs = self.get_observations()

        return obs

    def _set_action_space(self):  # 六个动作，速度，高度，航向加减
        self.action_space = []
        for i in range(ac_number):
            self.action_space.append(spaces.Discrete(1))

    def _set_observation_space(self):  # 纬度，经度，高度，航向和速度
        lim_low = MillierToXY(38.49, 105.74)
        lim_high = MillierToXY(38.50, 105.9)
        self.xlen = lim_high[0] - lim_low[0]
        self.ylen = lim_high[1] - lim_low[1]
        self.observation_space = [lim_low[0], lim_low[1], lim_high[0], lim_high[1]]

        # self.observation_space = spaces.Box(
        #     low=np.array([38.3,105.5,0,5000,500]),
        #     high=np.array([38.6,106.0,360,7000,700]),
        #     dtype=np.float32
        #     )

    def step(self, actions):  # 传入长度为8的列表,

        for i in range(ac_number):
            self.__apply_action(i, actions)



        self.step_number += 1

        done = self.get_done()
        obs = self.get_observations()
        reward = self.get_reward()
        info = self.get_info()

        return obs, reward, done, info

    def _get_initstate(self):  # 读取txt文件获得8架飞机初始状态信息
        number = []
        lat = []
        lon = []
        alt = []
        tas = []
        hdg = []
        data = []
        read = []
        self.step_number = 0
        a = sys.path[0] + '/%s.txt' % self.filename
        for line in open(sys.path[0] + '/%s.txt' % self.filename, "r"):
            data.append(line.replace('\n', ''))
            data = list(map(str, data))
        length = len(data)
        for i in range(length):
            data[i] = re.findall(r"\d+\.?\d*", data[i])
        for i in range(len(data)):
            read = read + data[i]
            read = list(map(float, read))

        for i in range(len(read) // 6):
            # number.append(read[6*i])
            # lat.append(random.uniform(38.48, 38.51))
            # lon.append(random.uniform(105.65, 105.90))
            # hdg.append(random.randint(0,355))
            # alt.append(read[6*i+4])
            # tas.append(read[6*i+5])


            number.append(read[6 * i])
            lat.append(read[6 * i + 1])
            lon.append(read[6 * i + 2])
            hdg.append(read[6 * i + 3])
            alt.append(read[6 * i + 4])
            tas.append(read[6 * i + 5])

        return number, lat, lon, hdg, alt, tas

    def get_done(self):
        # reward = self.get_reward()
        if self.step_number > self.max_step:
            done = [True]
        else:
            done = [False]
        return done

    def get_observations(self):
        obs = np.array([])

        cur_lats = bs.traf.lat
        cur_lons = bs.traf.lon
        cur_alts = bs.traf.alt
        cur_hdgs = bs.traf.hdg
        cur_cass = bs.traf.cas
        cur_banks = bs.traf.bank
        self.num_dis = self.num_dis+1
        if self.num_dis%1000 == 0:
            dis_send([cur_lats,cur_lons,cur_alts,cur_hdgs,cur_cass,cur_banks])



        # obs:self.action 1.lat 1.lon 1.hdg 2.lat ........ 4.hdg  4.action
        obs1 = np.array([])

        xy1 = MillierToXY(cur_lats[0], cur_lons[0])

        xy1[0] = (xy1[0] - self.observation_space[0]) / self.xlen
        xy1[1] = (xy1[1] - self.observation_space[1]) / self.xlen

        obs1 = np.append(obs1, xy1[0])
        obs1 = np.append(obs1, xy1[1])
        obs1 = np.append(obs1, cur_hdgs[0] / 360)

        xy2 = MillierToXY(cur_lats[1], cur_lons[1])
        xy2[0] = (xy2[0] - self.observation_space[0]) / self.xlen
        xy2[1] = (xy2[1] - self.observation_space[1]) / self.xlen

        obs1 = np.append(obs1, xy2[0] - xy1[0])
        obs1 = np.append(obs1, xy2[1] - xy1[1])
        target_hdg_arg = math.atan2(obs1[4],obs1[3])+pi/2 if math.atan2(obs1[4],obs1[3])+pi/2>0 else math.atan2(obs1[4],obs1[3])+2.5*pi
        target_hdg_degree = target_hdg_arg*180/pi  #degree

        obs1 = np.append(obs1, differ_angle(target_hdg_degree,obs1[2]*360)/360)


        #obs1 = np.array([obs1[5]])
        obs1.reshape((6,))



        obs2 = np.array([])





        obs2 = np.append(obs2, xy2[0])
        obs2 = np.append(obs2, xy2[1])
        obs2 = np.append(obs2, cur_hdgs[1] / 360)

        obs2 = np.append(obs2, xy1[0] - xy2[0])
        obs2 = np.append(obs2, xy1[1] - xy2[1])
        target_hdg_arg2 = math.atan2(obs2[4], obs2[3]) + pi / 2 if math.atan2(obs2[4],obs2[3]) + pi / 2 > 0 else math.atan2( obs2[4], obs2[3]) + 2.5 * pi
        target_hdg_degree2 = target_hdg_arg2 * 180 / pi  # degree

        obs2 = np.append(obs2, differ_angle(target_hdg_degree2, obs2[2] * 360) / 360)

        # obs1 = np.array([obs1[5]])
        obs2.reshape((6,))



        return [obs1,obs2]

    def get_reward(self):

        obs = self.get_observations()
        #rew1 = -(obs[3] ** 2 + obs[4] ** 2) ** 0.5 - abs(obs[5])
        # rew1 = -(obs[3] ** 2 + obs[4] ** 2) ** 0.5
        rew1 = -(((obs[0][3])**2 + obs[0][4]**2)**0.5)
            # if ((obs[0][3])**2 + obs[0][4]**2)**0.5 < 0.5:
            #     rew1 = 1000
            # else:
            #     rew1 = 0

        self.reward = [rew1,-rew1]        # self.reward = [rew1, rew2, rew3, good_rew]

        # for i in range(2):   #目前只判断了后四架飞机是否前四架的攻击范围内
        #     for j in range(2,4):
        #         a = self.isattack2ac(lat[i],lon[i],alt[i],hdg[i],lat[j],lon[j],alt[j])
        #         if a == 1:
        #             reward = 1
        #         elif a == 2:
        #             reward = 10
        #         elif a == 3:
        #             reward = 100
        #         else:
        #             reward = 0
        #         self.reward = self.reward + reward
        return self.reward

    def get_info(self):
        return {}

    def __apply_action(self, idx, actionarray):  # idx是第几个飞机注意与飞机呼号acid区分,
        cur_hdg = bs.traf.hdg[idx]
        cur_alt = bs.traf.alt[idx]
        cur_cas = bs.traf.cas[idx]

        # action有6个，0：航向增加1度，1：航向减少1度，2：速度增加10，
        # 3：速度减少10，4：高度增加10，5：高度减少10
        action = actionarray[idx]
        # action = action[idx]
        self.hdg = bs.traf.hdg[idx]
        # try:
        #     print('pro_hdg:   ', self.prohdg, '   hdg:  ', self.hdg, '  action:  ', action, ' d_hdg: ', self.hdg - self.prohdg)
        # except:
        #     pass
        bs.traf.ap.selhdgcmd(idx, cur_hdg + action)
        self.prohdg = cur_hdg
        preupdate()
        step()
        bs.traf.update()
        update()


        # if action <= 0:
        #     bs.traf.ap.selhdgcmd(idx,cur_hdg+60)
        #     #stack.stack('alt 000%s 5500'%idx)
        # else :
        #     bs.traf.ap.selhdgcmd(idx,cur_hdg-60)
        # if action == 0:
        #     bs.traf.ap.selhdgcmd(idx,cur_hdg+60)
        #     #stack.stack('alt 000%s 5500'%idx)
        # elif action == 1:
        #     bs.traf.ap.selhdgcmd(idx,cur_hdg-60)
        #     #stack.stack('alt 000%s 5500'%idx)
        # elif action == 2:
        #     bs.traf.ap.selspdcmd(idx,cur_cas+200)
        #     #stack.stack('alt 000%s 5500'%idx)
        # elif action == 3:
        #     bs.traf.ap.selspdcmd(idx,cur_cas-200)
        #     #stack.stack('alt 000%s 5500'%idx)
        # elif action == 4:
        #     bs.traf.ap.selaltcmd(idx,cur_alt+300)
        #     #stack.stack('alt 000%s 5500'%idx)
        # elif action == 5:
        #     bs.traf.ap.selaltcmd(idx,cur_alt-300)
        # stack.stack('alt 000%s 5500'%idx)
        # bs.traf.simt+=bs.traf.simdt
        # print(bs.traf.simt)

    def isattack2ac(self, lat1, lon1, alt1, hdg1, lat2, lon2, alt2):
        disb2ac = haversine(lat1, lon1, lat2, lon2)  # 计算两个飞机之间距离
        if disb2ac > 3:
            return 0  # 暂时用返回值为0表示互不在自己的攻击范围内
        else:
            ac1poz = MillierToXY(lat1, lon1)
            ac2poz = MillierToXY(lat2, lon2)

            vector1_1 = [ac2poz[0] - ac1poz[0], ac2poz[1] - ac1poz[1]]
            if hdg1 <= 90:
                vector1_2 = [math.sin(du2hudu(hdg1)), math.cos(du2hudu(hdg1))]
            elif 90 < hdg1 <= 180:
                vector1_2 = [math.cos(du2hudu(hdg1 - 90)), -math.sin(du2hudu(hdg1 - 90))]
            elif 180 < hdg1 <= 270:
                vector1_2 = [-math.sin(du2hudu(hdg1 - 180)), -math.cos(du2hudu(hdg1 - 180))]
            elif 270 < hdg1 <= 360:
                vector1_2 = [-math.cos(du2hudu(hdg1 - 270)), math.sin(du2hudu(hdg1 - 270))]

            if angle(vector1_1, vector1_2) > attack_angle_p:
                return 1  # 暂时用1表示ac2在ac1的半径内，但是水平角度不对
            else:
                v_vector1 = [1, 0]
                v_vector2 = [ac2poz[0] - ac1poz[0], alt2 - alt1]
                if angle(vector1_1, vector1_2) > attack_angle_v:
                    return 2  # 暂时用2表示ac2在ac1的半径内，水平方向角度符合，垂直方向角度不符合
                else:
                    return 3  # 暂时用3表示ac2完全在ac1的攻击范围内


def differ_angle(angle_from, angle_to):
    d_angle = angle_to - angle_from
    if d_angle > 180:
        d_angle -= 360
    elif d_angle < -180:
        d_angle += 360
    return d_angle


def MillierToXY(lat, lon):  # 在米勒坐标系下将经纬度转换为平面坐标

    L = 6381372 * pi * 2  # 地球周长
    W = L  # 平面展开后，x轴等于周长
    H = L / 2  # y轴约等于周长的一半
    mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间
    x = lon * pi / 180  # 将经度从度数转换为弧度
    y = lat * pi / 180  # 将纬度从度数转换为弧度
    y = 1.25 * math.log(math.tan(0.25 * pi + 0.4 * y))
    x = (W / 2) + (W / (2 * pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    xy_coordinate = [x, y]
    return xy_coordinate


def MillerToLonLat(x, y):  # 米勒坐标系下的平面坐标转换为经纬度坐标
    lonlat_coordinate = []  # 转换后的经纬度坐标集
    L = 6381372 * pi * 2  # 地球周长
    W = L  # 平面展开后，x轴等于周长
    H = L / 2  # y轴约等于周长的一半
    mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间
    lat = ((H / 2 - y) * 2 * mill) / (1.25 * H)
    lat = ((math.atan(math.exp(lat)) - 0.25 * pi) * 180) / (0.4 * pi)
    lon = (x - W / 2) * 360 / W
    lonlat_coordinate = [lat, lon]
    return lonlat_coordinate


def haversine(lat1, lon1, lat2, lon2):  # 根据经纬度计算距离，单位为公里

    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


def du2hudu(a):
    return (a / 360) * 2 * pi


def angle(l1, l2):  # 返回的是度数
    dx1 = l1[0]
    dy1 = l1[1]
    dx2 = l2[0]
    dy2 = l2[1]

    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
        included_angle = 360 - included_angle

    return included_angle


def dis_send(obs):
    lvpos = []
    cur_lats = obs[0]
    cur_lons = obs[1]
    cur_alts = obs[2]
    cur_hdgs = obs[3]
    cur_banks = obs[5]

    length = len(cur_lats)

    for i in range(length):
        Rad_hdg, Rad_pitch, Rad_roll = radians(cur_hdgs[i]), radians(0), radians(cur_banks[i])
        Rad_lat, Rad_lon = radians(cur_lats[i]), radians(cur_lons[i])
        Psi, Theta, Phi = HeadingPitchRollToEuler(Rad_hdg, Rad_pitch, Rad_roll, Rad_lat, Rad_lon)

        env_send_dis(i, [cur_lats[i], cur_lons[i], cur_alts[i]], Psi, Theta, Phi)



if __name__ == "__main__":
    env = BlueSkyEnv('TEST')
    obs = env.reset()
    while True:
        action = np.random.randint(0, 2, size=(1, 4))
        print(action)
        obs, reward, done, _ = env.step(action)
        print(obs)
        print('reward=%s' % reward)
        time.sleep(0.2)
        dis_send(obs)
