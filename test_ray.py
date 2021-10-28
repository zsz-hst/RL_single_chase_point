import gym, ray
from ray.rllib.agents import ppo

from blenv import BlueSkyEnv
ray.init()
trainer = ppo.PPOTrainer(env=BlueSkyEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(trainer.train())