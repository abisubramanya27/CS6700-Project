
"""
DO NOT EDIT THIS FILE. ANY CHANGES WILL BE OVERWRITTEN
THIS FILE USES THE FUNCTIONS IMPLEMENTED IN agent.py TO EVALUATE YOUR AGENTS
"""

import os
import aicrowd_gym
import gym_bellman
import numpy as np
from tqdm import tqdm

from agent import Agent


def train(agent, env):
    obs = env.reset()
    action = agent.register_reset_train(obs)
    done = False
    while not done:
        obs, reward, done, info = env.step(action)
        action = agent.compute_action_train(obs, reward, done, info)

def evaluate(agent, env, ok=False):
    rewards = 0
    obs = env.reset()
    action = agent.register_reset_test(obs)
    policy = [[obs, action]]
    done = False
    while not done:
        obs, reward, done, info = env.step(action)
        policy[-1].append(reward)
        action = agent.compute_action_test(obs, reward, done, info)
        rewards += reward
        policy += [[obs, action]]

    if ok:
        for pair in policy:
            print(pair)
    return rewards


if __name__ == "__main__":
    ENV_NAME = os.getenv("ENV_NAME", "acrobot")

    N_TRAIN_EPISODES = {"acrobot": 2000, "taxi": 1500, "kbca": 2000, "kbcb": 2000, "kbcc": 2000}

    N_EVAL_EPISODES = 100

    if ENV_NAME == "acrobot":
        env = aicrowd_gym.make("Acrobot-v1")
    elif ENV_NAME == "taxi":
        env = aicrowd_gym.make("Taxi-v3")
    elif ENV_NAME == "kbca":
        env = aicrowd_gym.make("gym_bellman:kbc-a-v0")
    elif ENV_NAME == "kbcb":
        env = aicrowd_gym.make("gym_bellman:kbc-b-v0")
    elif ENV_NAME == "kbcc":
        env = aicrowd_gym.make("gym_bellman:kbc-c-v0")


    agent = Agent(ENV_NAME)

    # print(env.action_space)
    # print(env.observation_space)

    for i in tqdm(range(N_TRAIN_EPISODES[ENV_NAME])):
        train(agent, env)

    rewards = []
    for i in tqdm(range(N_EVAL_EPISODES)):
        rewards.append(evaluate(agent, env, bool(i == N_EVAL_EPISODES-1)))

    print(f"Mean reward on your agent for {ENV_NAME} is {np.mean(rewards)}")
