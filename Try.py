import rlcard as rlcard
from rlcard.agents import nfsp_agent
from rlcard.agents import random_agent
import torch
from rlcard.utils import get_device
from tqdm import tqdm
import matplotlib.pyplot as plt

def run(env_name):
    # Make environment
    env = rlcard.make(
        env_name,
        config={
            'seed': 42,
            'game_num_players':6,
            'num_actions' : 14
        }
    )
    # device = get_device()
    agent_bot = torch.load(r'C:\Program\Neural_Network\Poker_Stars\prev_models\model_10.726.793.pth', map_location='cuda:0')
    # agent_super = torch.load(r'C:\Program\Neural_Network\Poker_Stars\log_dir_nfsp\model.pth', map_location=device)
    agent_super = torch.load(r'C:\Program\Neural_Network\Poker_Stars\prev_models\model_10.726.793.pth', map_location='cuda:0')

    bots = [agent_bot for _ in range(env.num_players-1)]
    bots.append(agent_super)
    env.set_agents(bots)

    schedule = []
    total_bank = 0
    for i in tqdm(range(20000)):
        trajectories, player_wins = env.run(is_training=False)
        total_bank += (player_wins[-1])
        schedule.append(total_bank)

    plt.plot(schedule)
    plt.show()

run('no-limit-holdem')