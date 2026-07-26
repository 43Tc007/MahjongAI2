import gymnasium
import numpy as np
import numpy.typing as npt

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from mahjong_helper import GameState, EAST, SOUTH, WEST, NORTH, game_state_mask, game_state_array

from typing import List
from mahjong_helper import *

from fan_calculator import calculate_fan
from copy import deepcopy

class MahjongGameEnv(AECEnv):
    
    metadata = {"render_modes": [], "name": "MahjongV0"}

    def __init__(self, render_mode=None):
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.render_mode = render_mode

    def observation_space(self, agent) -> gymnasium.Space:
        return spaces.Dict({'observation': spaces.Box(low=0, high=255, shape=(34, 29), dtype=np.uint8), 'action_mask': spaces.Box(low=0, high=1, shape=(75,), dtype=np.uint8)})
    
    def action_space(self, agent) -> gymnasium.Space:
        return Discrete(75)
    
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return None
        elif self.render_mode == "human":
            pass

    def observe(self, agent):
        return {
            'observation': game_state_mask_simplified(self.gamestate, self.agent_name_mapping[agent]).T,
            'action_mask': self.mask
        }
    
    def state(self):
        return np.vstack([np.hstack([game_state_array_simplified(self.gamestate), np.zeros(shape=(32, 8))]), wall_sequence_array_simplified(self.gamestate.wall)]).T
    
    def close(self):
        pass

    def rotate_selector_by_index(self, selector: AgentSelector, idx: int) -> str:
        """
        Rotates the agent order using an integer index to place that agent at the front.
        
        Args:
            selector: The PettingZoo AgentSelector instance.
            target_idx: The index of the agent within the CURRENT cycle to bring to front.
            
        Returns:
            The new active agent selection string.
        """
        permuted_agents = self.agents[idx:] + self.agents[:idx]
        selector.reinit(permuted_agents)
        
        # Reset internal tracking back to index 0
        return selector.reset()

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = AgentSelector(self.agents)
        self.gamestate = GameState(
            round_wind=np.random.randint(0, 4),
            game_wind=np.random.randint(0, 4),
            current_player=EAST,
            wall_remaining=144,
        )
        self.gamestate.current_player = self.gamestate.game_wind
        self.agent_selection = self.rotate_selector_by_index(self._agent_selector, idx=self.gamestate.game_wind)
        self.deal_hands()
        self.mask = self.generate_action_mask(self.gamestate.current_player, self.gamestate.last_drawn)
        
    def deal_hands(self):
        for player_idx in range(4):
            tile_count = 13
            if seat(player_idx, self.gamestate.game_wind) == EAST:
                tile_count = 14
            for _ in range(tile_count):
                while True:
                    tile = draw_tile(self.gamestate, player_idx, self.gamestate.wall)
                    if not is_flower(tile):
                        break
                    execute_flower(self.gamestate, player_idx, tile)
        flower_counts = [int(self.gamestate.flowers[player_idx].sum()) for player_idx in range(4)]
        if any(count >= 7 for count in flower_counts):
            for agent in self.agents:
                self.terminations[agent] = True
                self.rewards[agent] = 0
                self._cumulative_rewards[agent] = 0
                self.infos[agent]["reason"] = "initial_flower"
                self.gamestate.wall_remaining = len(self.gamestate.wall)

        

    def generate_action_mask(self, player_idx: int, target_tile: int | None = None) -> np.ndarray:
        action_mask = np.zeros(75, dtype=np.uint8)
        action_mask[74] = 1  # pass always available (except DISCARD phase, we'll override)
        if self.gamestate.phase == WAIT_TSUMO_ADD_KAN_AN_KAN:
            for tile in range(34):
                if (self.gamestate.hands[player_idx][tile] == 1 and tile in self.gamestate.addkanable_tiles[player_idx]) or self.gamestate.hands[player_idx][tile] == 4: 
                    action_mask[tile + 34] = 1
            assert target_tile is not None
            fan = calculate_fan(self.gamestate, player_idx, target_tile)
            if fan >= 3:
                action_mask[73] = 1  # tsumo

        elif self.gamestate.phase == DISCARD:
            action_mask[74] = 0  # no pass in discard phase
            action_mask[:42] = (self.gamestate.hands[player_idx] != 0).astype(int)

        elif self.gamestate.phase == WAIT_RESPONSE:
            assert target_tile is not None
            # chow
            if ((self.gamestate.current_player - player_idx) % 4 == 3 and target_tile <= 26):
                if (self.gamestate.hands[player_idx][target_tile + 2] >= 1 and 
                    self.gamestate.hands[player_idx][target_tile + 1] >= 1 and 
                    target_tile // 9 == (target_tile + 2) // 9 and
                    target_tile // 9 == (target_tile + 1) // 9
                ):
                    action_mask[68] = 1
                if (self.gamestate.hands[player_idx][target_tile - 1] >= 1 and 
                    self.gamestate.hands[player_idx][target_tile + 1] >= 1 and 
                    target_tile // 9 == (target_tile - 1) // 9 and
                    target_tile // 9 == (target_tile + 1) // 9
                ):
                    action_mask[69] = 1
                if (self.gamestate.hands[player_idx][target_tile - 1] >= 1 and 
                    self.gamestate.hands[player_idx][target_tile - 2] >= 1 and 
                    target_tile // 9 == (target_tile - 1) // 9 and
                    target_tile // 9 == (target_tile - 2) // 9
                ):
                    action_mask[70] = 1
            # pung
            if self.gamestate.hands[player_idx][target_tile] >= 2:
                action_mask[71] = 1
            # ming kan
            if self.gamestate.hands[player_idx][target_tile] >= 3:
                action_mask[72] = 1
            # ron
            copy_gamestate = deepcopy(self.gamestate)
            copy_gamestate.hands[player_idx][target_tile] += 1
            if calculate_fan(copy_gamestate, player_idx, target_tile) >= 3:
                action_mask[73] = 1
        else:
            # other phases (e.g., WAIT_HUA_HU)
            assert target_tile is not None
            if calculate_fan(self.gamestate, player_idx, target_tile) >= 3:
                action_mask[73] = 1

        # Store for later use
        self.mask = action_mask
        return action_mask
    
    def terminate_game(self, target_tile: int = -1, winnning_player_idx: int = -1, losing_player_idx: int = -1, terminate_type="exhausted"):
        payout = {
            3 : 1 / 16,
            4 : 2 / 16,
            5 : 3 / 16,
            6 : 4 / 16,
            7 : 6 / 16,
            8 : 8 / 16,
            9 : 12 / 16,
            10 : 16 / 16,
            11 : 24 / 16,
            12 : 32 / 16,
            13 : 48 / 16
        }

        if terminate_type == "tsumo":
            fan = calculate_fan(self.gamestate, winnning_player_idx, target_tile)
            winner = self.agents[winnning_player_idx]
            for agent in self.agents:
                self.terminations[agent] = True
                if agent == winner:
                    self.rewards[agent] = payout[fan] * 1.5
                else:
                    self.rewards[agent] = - payout[fan] * 0.5
            self.infos[winner] = {'win_type': 'tsumo', 'fan': fan}
        elif terminate_type == "ron":
            self.gamestate.hands[winnning_player_idx][target_tile] += 1
            fan = calculate_fan(self.gamestate, winnning_player_idx, target_tile)
            winner = self.agents[winnning_player_idx]
            loser = self.agents[losing_player_idx]
            for agent in self.agents:
                self.terminations[agent] = True
                if agent == winner:
                    self.rewards[agent] = payout[fan] 
                elif agent == loser:
                    self.rewards[agent] = - payout[fan] 
                else:
                    self.rewards[agent] = 0
            self.infos[winner] = {'win_type': 'ron', 'fan': fan}
        elif terminate_type == "exhausted":
            for agent in self.agents:
                self.terminations[agent] = True
                self.rewards = {agent: 0 for agent in self.agents}
                self.infos[agent] = {'win_type': 'draw', 'fan': 0}   
        return
    
    def step(self, action) -> None:
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]

        # ---- Execute the chosen action ----
        if self.gamestate.phase == WAIT_TSUMO_ADD_KAN_AN_KAN:
            if 34 <= int(action) <= 67:
                tile = int(action) - 34
                if tile in self.gamestate.addkanable_tiles[self.gamestate.current_player]:
                    execute_add_kan(self.gamestate, player_idx, tile)
                    self.gamestate.phase = WAIT_HUA_HU
                else:
                    execute_an_kan(self.gamestate, player_idx, tile)
                    self.gamestate.phase = WAIT_HUA_HU
            elif int(action) == 73:
                self.terminate_game(target_tile=self.gamestate.last_drawn, winnning_player_idx=player_idx, terminate_type="tsumo")
            else:
                assert int(action) == 74
        elif self.gamestate.phase == DISCARD:
            execute_discard(self.gamestate, player_idx, int(action))
            assert self.gamestate.hands[self.gamestate.current_player].sum() % 3 == 1
        elif self.gamestate.phase == WAIT_RESPONSE:
            self.gamestate.action_array[int(action)] = player_idx
        else:
            # WAIT_HUA_HU or other
            if int(action) == 73:
                self.terminate_game(target_tile=self.gamestate.last_drawn, winnning_player_idx=player_idx, terminate_type="tsumo")
            else:
                idxs = np.nonzero(self.gamestate.hands[self.gamestate.current_player][34:])[0]
                tile = int(34 + idxs[0])
                execute_flower(self.gamestate, self.gamestate.current_player, tile=tile)

        # ---- Post-action processing loop ----
        max_iterations = 100
        iterations = 0
        while not self.terminations[self.agent_selection]:
            iterations += 1
            if max_iterations == iterations:
                raise RecursionError

            # Phase: WAIT_TSUMO_ADD_KAN_AN_KAN -> DISCARD
            if self.gamestate.phase == WAIT_TSUMO_ADD_KAN_AN_KAN:
                self.gamestate.phase = DISCARD
                self.mask = self.generate_action_mask(self.agent_name_mapping[self.agent_selection])
                break

            # Phase: DISCARD -> WAIT_RESPONSE
            if self.gamestate.phase == DISCARD:
                self.gamestate.phase = WAIT_RESPONSE
                self.gamestate.action_array = np.zeros(shape=(75,), dtype=np.uint8)
                self.gamestate.action_array.fill(4)  # default pass (4 means no action)

            # Phase: WAIT_RESPONSE - cycle through agents
            if self.gamestate.phase == WAIT_RESPONSE:
                # Move to next agent
                self.agent_selection = self._agent_selector.next()

                # Check if we've come back to the current player (who discarded)
                if self.agent_name_mapping[self.agent_selection] == self.gamestate.current_player:
                    # Evaluate collected responses
                    if self.gamestate.action_array[73] < 4:  # ron
                        self.terminate_game(self.gamestate.last_discard, self.gamestate.action_array[73], self.gamestate.current_player, terminate_type="ron")
                        break
                    elif self.gamestate.wall_remaining == 0:
                        self.terminate_game(terminate_type="exhausted")
                        break
                    elif self.gamestate.action_array[72] < 4:  # ming kan
                        execute_ming_kan(self.gamestate, self.gamestate.action_array[72], self.gamestate.last_discard)
                        self.agent_selection = self.rotate_selector_by_index(self._agent_selector, idx=self.gamestate.action_array[72])
                        self.gamestate.current_player = self.agent_name_mapping[self.agent_selection]
                        self.gamestate.phase = WAIT_HUA_HU
                    elif self.gamestate.action_array[71] < 4:  # pon
                        execute_pon(self.gamestate, self.gamestate.action_array[71], self.gamestate.last_discard)
                        self.agent_selection = self.rotate_selector_by_index(self._agent_selector, idx=self.gamestate.action_array[71])
                        self.gamestate.current_player = self.agent_name_mapping[self.agent_selection]
                        self.gamestate.phase = WAIT_TSUMO_ADD_KAN_AN_KAN
                    elif self.gamestate.action_array[70] < 4:  # chow (low)
                        execute_chow(self.gamestate, self.gamestate.action_array[70], self.gamestate.last_discard, [self.gamestate.last_discard-2, self.gamestate.last_discard-1])
                        self.agent_selection = self.rotate_selector_by_index(self._agent_selector, idx=self.gamestate.action_array[70])
                        self.gamestate.current_player = self.agent_name_mapping[self.agent_selection]
                        self.gamestate.phase = WAIT_TSUMO_ADD_KAN_AN_KAN
                    elif self.gamestate.action_array[69] < 4:  # chow (mid)
                        execute_chow(self.gamestate, self.gamestate.action_array[69], self.gamestate.last_discard, [self.gamestate.last_discard-1, self.gamestate.last_discard+1])
                        self.agent_selection = self.rotate_selector_by_index(self._agent_selector, idx=self.gamestate.action_array[69])
                        self.gamestate.current_player = self.agent_name_mapping[self.agent_selection]
                        self.gamestate.phase = WAIT_TSUMO_ADD_KAN_AN_KAN
                    elif self.gamestate.action_array[68] < 4:  # chow (low)
                        execute_chow(self.gamestate, self.gamestate.action_array[68], self.gamestate.last_discard, [self.gamestate.last_discard+1, self.gamestate.last_discard+2])
                        self.agent_selection = self.rotate_selector_by_index(self._agent_selector, idx=self.gamestate.action_array[68])
                        self.gamestate.current_player = self.agent_name_mapping[self.agent_selection]
                        self.gamestate.phase = WAIT_TSUMO_ADD_KAN_AN_KAN
                    else:
                        # No one responded, pass to next player (the one who discarded)
                        self.agent_selection = self._agent_selector.next()
                        self.gamestate.current_player = self.agent_name_mapping[self.agent_selection]
                        self.gamestate.phase = WAIT_HUA_HU
                else:
                    # Not current player; generate mask for this agent
                    self.mask = self.generate_action_mask(self.agent_name_mapping[self.agent_selection], self.gamestate.last_discard)
                    if self.mask.sum() >= 2:
                        break
                    else:
                        # Auto-pass: set action_array[74] = player_idx
                        self.gamestate.action_array[74] = self.agent_name_mapping[self.agent_selection]
                        # Continue loop to next agent

            # Phase: WAIT_HUA_HU - draw a tile
            if self.gamestate.phase == WAIT_HUA_HU:
                if self.gamestate.wall_remaining == 0:
                    self.terminate_game(terminate_type="exhausted")
                    break
                drawn_tile = draw_tile(self.gamestate, self.gamestate.current_player, self.gamestate.wall)
                # Check flowers
                while is_flower(drawn_tile):
                    self.mask = self.generate_action_mask(self.gamestate.current_player, drawn_tile)
                    if self.mask.sum() >= 2:  # tsumo available
                        break
                    else:
                        execute_flower(self.gamestate, self.agent_name_mapping[self.agent_selection], drawn_tile)
                        if self.gamestate.wall_remaining == 0:
                            self.terminate_game(terminate_type="exhausted")
                            break
                        drawn_tile = draw_tile(self.gamestate, self.gamestate.current_player, self.gamestate.wall)
                # After flower handling, check if we broke due to tsumo possibility
                if self.mask.sum() >= 2:
                    pass
                # If no tsumo and no flowers, transition to WAIT_TSUMO_ADD_KAN_AN_KAN
                if self.gamestate.phase == WAIT_HUA_HU:
                    self.gamestate.phase = WAIT_TSUMO_ADD_KAN_AN_KAN
                    self.mask = self.generate_action_mask(self.gamestate.current_player, drawn_tile)
                    if self.mask.sum() >= 2:
                        break
                    else:
                        self.gamestate.phase = DISCARD
                        self.mask = self.generate_action_mask(self.gamestate.current_player)
                        break


from torchrl.envs import PettingZooWrapper
from torchrl.envs.utils import MarlGroupMapType
def make_env():
    return PettingZooWrapper(
        env=MahjongGameEnv(),
        use_mask=True,
        return_state=True,
        categorical_actions=True,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP
    )