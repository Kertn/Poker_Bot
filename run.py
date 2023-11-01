import time

import rlcard as rlcard
from rlcard.agents import NFSPAgent
import os
import argparse
import torch
import multiprocessing
#from evaluate_model import eval
from tqdm import tqdm

from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

print(torch.cuda.device_count())

def train(args):
    # Check whether gpu is available
    device = get_device()

    print(device)
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': 42,
            'game_num_players': 6,
            'num_actions': 14
        }
    )

    # Initialize the agent and use random agents as opponents
    agent = torch.load(r'C:\Program\Neural_Network\Poker_Stars\prev_models\model_10.726.793.pth', map_location='cuda:0')

    # agent = NFSPAgent(
    #         num_actions=env.num_actions,
    #         state_shape=env.state_shape[0],
    #         hidden_layers_sizes=[520, 500, 460, 440],
    #         q_mlp_layers=[520, 500, 460, 440],
    #         device=device,
    #     )
    env.set_agents([agent for _ in range(env.num_players)])
    agents = [agent]

    start = time.time()
    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if episode % 500 == 0:
                print(' Episode - ', episode)

            agent.sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            #Evaluate the performance. Play with random agents.
            # if episode % args.evaluate_every == 0:
            #     logger.log_performance(
            #         episode,
            #         tournament(
            #             env,
            #             args.num_eval_games,
            #         )[0]
            #     )
            # if episode % args.evaluate_every == 0:
            #     logger.log_performance(
            #         episode,
            #         eval(agent, args.num_eval_games)
            #     )
            if episode % 100000 == 0:
                print('sleep')
                time.sleep(1)
            if episode % 50000 == 0:
                save_path = os.path.join(args.log_dir, 'model.pth')
                torch.save(agent, save_path)
                print('Model saved in', save_path)
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
    end = time.time()
    print('Total episodes: ', args.num_episodes, ' Total time: ', end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("NFSP Poker Try_1")
    parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='nfsp',
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=20000000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        #default=r'C:\Program\Neural_Network\Poker_Stars\Cuda_log'
        default=r'C:\Program\Neural_Network\Poker_Stars\log_dir_nfsp',
    )

    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)