import rlcard as rlcard
from rlcard.agents import RandomAgent

def run(env_name):
    # Make environment
    env = rlcard.make(
        env_name,
        config={
            'seed': 42,
            'game_num_players':5,
        }
    )

    print(env.reset())
    print(env.step(4))





run('no-limit-holdem')