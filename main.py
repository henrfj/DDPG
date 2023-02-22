"""
main training loop
"""
import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve


if __name__ =="__main__":
    env = gym.make("Pendulum-v1")
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

    n_games = 250

    figure_file = "plots/pendulum.ong"

    best_score = env.reward_range[0] # Bound of performance
    score_history = []
    # Train vs Test
    """ Model loading:
        - Needs to call learning function before you can load it.
        - Just how TF works. Fill it with dummy, then call learn once to actually load it.    
    """
    load_checkpoint = False 
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    
    for i in range(n_games): # number of full games/episodes
        observation = env.reset()
        done=False
        score=0
        while not done: # one episode
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint: # To avoid updating if just loading to test
                agent.learn()
            observation = observation_ # update state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print("episode ", i, "score %.1" % score, "avg score %.1" %avg_score)

    if not load_checkpoint:
        # plot learning curves
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)