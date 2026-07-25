import numpy as np
from env_simplified import MahjongGameEnv
from pygame_visualizer import array_to_tile_string, melds_to_string, tiles_unicode

def random_agent(env, agent):
    """Select a random legal action for the given agent."""
    obs = env.observe(agent)
    action_mask = obs['action_mask']
    legal_actions = np.where(action_mask == 1)[0]
    if len(legal_actions) == 0:
        # Fallback: pass (action 74) should always be legal
        return 74
    return np.random.choice(legal_actions)

def main():
    env = MahjongGameEnv(render_mode=None)
    env.reset()

    for episode in range(100000):
        env.reset()
        print(f"\n--- Episode {episode + 1} ---")

        for agent in env.agent_iter():
            # Get observation and action mask for the current agent
            obs = env.observe(agent)
            action = random_agent(env, agent)

            # Step the environment
            env.step(action)

            # If the round ended, break out of the loop
            if env.terminations[agent]:

                if any(r > 0 for r in env.rewards.values()):
                    winner = [a for a in env.agents if env.rewards[a] > 0][0]
                    player_idx = env.agent_name_mapping[winner]
                    win_type = env.infos[winner]['win_type']
                    fan = env.infos[winner]['fan']
                    print(f"Winner: {winner} ({win_type}) fan={fan}")

                    # Hand, melds, flowers
                    hand = env.gamestate.hands[player_idx]
                    hand_str = array_to_tile_string(hand)
                    melds = env.gamestate.melds[player_idx]
                    melds_str = melds_to_string(melds) if melds else ""
                    flowers = env.gamestate.flowers[player_idx]
                    flowers_str = array_to_tile_string(flowers) if flowers.sum() > 0 else ""

                    # Winning tile
                    if win_type == 'ron':
                        win_tile = env.gamestate.last_discard
                    else:
                        win_tile = env.gamestate.last_drawn
                    win_tile_str = tiles_unicode[win_tile]

                    print(f"Winning tile: {win_tile_str}")
                    print(f"Hand: {hand_str}")
                    if melds_str:
                        print(f"Melds: {melds_str}")
                    if flowers_str:
                        print(f"Flowers: {flowers_str}")
                    print(f"Rewards: {env.rewards}")
                else:
                    pass
                break

    env.close()

if __name__ == "__main__":
    main()
