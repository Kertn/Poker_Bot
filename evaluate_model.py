import rlcard as rlcard
from rlcard.agents import nfsp_agent
from rlcard.agents import random_agent
import torch
from rlcard.utils import get_device
from tqdm import tqdm
import matplotlib.pyplot as plt

print('DEVICE_COUNT', torch.cuda.device_count())
device = torch.cuda.device(0)
print('GET_DEVICE', device)
def eval(agent, num_eval):
    # Make environment
    env = rlcard.make(
        'no-limit-holdem',
        config={
            'seed': 42,
            'game_num_players':6,
            'num_actions' : 14
        }
    )
    agent_bot = torch.load(r'C:\Program\Neural_Network\Poker_Stars\Step 3358299\model.pth', map_location='cuda:0')
    agent_super = agent

    bots = [agent_bot for _ in range(env.num_players-1)]
    bots.append(agent_super)
    env.set_agents(bots)

    total_bank = 0
    for i in tqdm(range(num_eval)):
        trajectories, player_wins = env.run(is_training=False)
        total_bank += player_wins[-1]
    return total_bank
