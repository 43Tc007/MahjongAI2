import torch
from torchrl.envs import PettingZooWrapper
from env_v0 import MahjongGameEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import MarlGroupMapType
# from pygame_visualizer import render_game_state
# import pygame
# import time
# pygame.init()
# screen = pygame.display.set_mode(size=(800, 800))
# font = pygame.font.Font("C:/Windows/Fonts/seguisym.ttf", 48)

env = PettingZooWrapper(env=MahjongGameEnv(), use_mask=True, return_state=True, categorical_actions=True, group_map=MarlGroupMapType.ALL_IN_ONE_GROUP)

# screen.fill('white')
# render_game_state(env._env.gamestate, screen, font)
# pygame.display.update()
# time.sleep(100)

print(env.reset())