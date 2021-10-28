import gym, ray
from ray.rllib.agents import ppo
import blenv



ray.init()
trainer = ppo.PPOTrainer(env=blenv.BlueSkyEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(trainer.train())