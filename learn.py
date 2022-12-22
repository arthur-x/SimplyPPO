from agent import PPOAgent, Normalization
import gym
import pybullet_envs
import argparse
import csv


def learn(device=0, environment=0, log=1):
    env = gym.make(env_list[environment])
    log_dir = 'saves/' + str(environment+1) + '/log' + str(log) + '.csv'
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    state_norm = Normalization(state_dim)
    truncate_length = env._max_episode_steps
    rollout_length = 2048
    evaluate_length = 2e3
    agent = PPOAgent(state_dim, action_dim, rollout_length, device=device)
    total_frames = 0
    with open(log_dir, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frames', 'return'])
        writer.writerow([total_frames, evaluate(environment, agent, state_norm)])
    while total_frames < 1e6:
        state = env.reset()
        state = state_norm(state)
        frame = 0
        while 1:
            action, log_p = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            frame += 1
            total_frames += 1
            next_state = state_norm(next_state)
            agent.remember(state, next_state, action, log_p, reward, done, frame == truncate_length)
            state = next_state
            if total_frames % rollout_length == 0:
                agent.train()
            if total_frames % evaluate_length == 0:
                with open(log_dir, "a+", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([total_frames, evaluate(environment, agent, state_norm)])
            if done:
                break


def evaluate(environment, agent, state_norm):
    env = gym.make(env_list[environment])
    state = env.reset()
    state = state_norm(state, update=False)
    total_reward = 0
    while 1:
        action = agent.act(state, mean=True)
        next_state, reward, done, _ = env.step(action)
        next_state = state_norm(next_state, update=False)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env_list = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-e', '--env', type=int, default=0)
    parser.add_argument('-l', '--log', type=int, default=1)
    args = parser.parse_args()
    learn(device=args.gpu, environment=args.env, log=args.log)
