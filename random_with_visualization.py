import numpy as np
from env_simplified import MahjongGameEnv
import time
import pygame
from pygame_visualizer import array_to_tile_string, melds_to_string, tiles_unicode, render_game_state

pygame.init()
screen = pygame.display.set_mode(size=(800, 800))
font = pygame.font.Font("C:/Windows/Fonts/seguisym.ttf", 48)

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
    # Create environment (set render_mode="human" to see the GUI)
    env = MahjongGameEnv(render_mode=None)  # or None for headless
    env.reset()

    # Loop over episodes
    for episode in range(100):  # play 5 rounds
        env.reset()
        print(f"\n--- Episode {episode+1} ---")
        for agent in env.agent_iter():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            screen.fill('white')
            render_game_state(env.gamestate, screen, font)
            pygame.display.update()

            # Get observation and action mask for the current agent
            obs = env.observe(agent)
            action_mask = obs['action_mask']

            # Let the agent decide (here: random)
            action = random_agent(env, agent)

            env.step(action)
            # Step the environment
            screen.fill('white')
            render_game_state(env.gamestate, screen, font)
            print(env.rewards)
            pygame.display.update()
            time.sleep(1)

            # If the round ended, break out of the loop (the next reset will start a new one)
            if env.terminations[agent]:
                # Check if it's a win (any positive reward)
                if any(r > 0 for r in env.rewards.values()):
                    winner = [a for a in env.agents if env.rewards[a] > 0][0]
                    player_idx = env.agent_name_mapping[winner]
                    win_type = env.infos[winner]['win_type']
                    fan = env.infos[winner]['fan']
                    print(f"Winner: {winner} ({win_type}) fan={fan}")

                    # Get hand, melds, flowers
                    hand = env.gamestate.hands[player_idx]
                    hand_str = array_to_tile_string(hand)
                    melds = env.gamestate.melds[player_idx]
                    melds_str = melds_to_string(melds) if melds else ""
                    flowers = env.gamestate.flowers[player_idx]
                    flowers_str = array_to_tile_string(flowers) if flowers.sum() > 0 else ""

                    # Winning tile
                    if win_type == 'ron':
                        win_tile = env.gamestate.last_discard
                    else:  # tsumo
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
                    print("Round ended in draw.")
                break


    env.close()

if __name__ == "__main__":
    main()