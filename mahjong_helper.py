import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import random
import numpy.typing as npt

def is_flower(idx: int) -> bool:
    return idx >= 34

def is_19(tile: int) -> bool:
    return tile in [0, 9, 18, 8, 17, 26]

def is_wind(tile: int) -> bool:
    return tile >= 27 and tile <= 30

def is_dragon(tile: int) -> bool:
    return tile >= 31 and tile <= 33

def seat(player: int, game_wind: int):
    return (int(player) - int(game_wind) + 4) % 4

def is_zi(tile: int):
    return is_wind(tile) or is_dragon(tile)

action_name = {
    # Discard actions (tiles)
    0: "discard 1m", 1: "discard 2m", 2: "discard 3m", 3: "discard 4m", 4: "discard 5m", 
    5: "discard 6m", 6: "discard 7m", 7: "discard 8m", 8: "discard 9m",
    9: "discard 1p", 10: "discard 2p", 11: "discard 3p", 12: "discard 4p", 13: "discard 5p", 
    14: "discard 6p", 15: "discard 7p", 16: "discard 8p", 17: "discard 9p",
    18: "discard 1s", 19: "discard 2s", 20: "discard 3s", 21: "discard 4s", 22: "discard 5s", 
    23: "discard 6s", 24: "discard 7s", 25: "discard 8s", 26: "discard 9s",
    27: "discard E", 28: "discard S", 29: "discard W", 30: "discard N",
    31: "discard white", 32: "discard green", 33: "discard red",
    34: "discard red flower 1", 35: "discard red flower 2", 36: "discard red flower 3", 37: "discard red flower 4",
    38: "discard black flower 1", 39: "discard black flower 2", 40: "discard black flower 3", 41: "discard black flower 4",

    # Kan actions (tile + 34)
    42: "kan 1m", 43: "kan 2m", 44: "kan 3m", 45: "kan 4m", 46: "kan 5m", 47: "kan 6m", 
    48: "kan 7m", 49: "kan 8m", 50: "kan 9m",
    51: "kan 1p", 52: "kan 2p", 53: "kan 3p", 54: "kan 4p", 55: "kan 5p", 56: "kan 6p", 
    57: "kan 7p", 58: "kan 8p", 59: "kan 9p",
    60: "kan 1s", 61: "kan 2s", 62: "kan 3s", 63: "kan 4s", 64: "kan 5s", 65: "kan 6s", 
    66: "kan 7s", 67: "kan 8s",

    # Special melds and win actions
    68: "chow high",     # chow with +1,+2
    69: "chow mid",      # chow with -1,+1
    70: "chow low",      # chow with -2,-1
    71: "pon",           # pung (triplet)
    72: "ming kan",      # exposed kan
    73: "ron/tsumo",     # winning (ron or tsumo depending on phase)
    74: "pass"           # skip / no action
}

tiles_name = {
    0: "1m", 1: "2m", 2: "3m", 3: "4m", 4: "5m",
    5: "6m", 6: "7m", 7: "8m", 8: "9m",
    9: "1p", 10: "2p", 11: "3p", 12: "4p", 13: "5p",
    14: "6p", 15: "7p", 16: "8p", 17: "9p",
    18: "1s", 19: "2s", 20: "3s", 21: "4s", 22: "5s",
    23: "6s", 24: "7s", 25: "8s", 26: "9s",
    27: "E", 28: "S", 29: "W", 30: "N",
    31: "white", 32: "green", 33: "red",
    34: "red flower 1"
}

EAST = 0
SOUTH = 1
WEST = 2
NORTH = 3

# DRAW HERE =  -1
WAIT_TSUMO_ADD_KAN_AN_KAN = 0
DISCARD = 1
WAIT_RESPONSE = 2
WAIT_HUA_HU = 3

MAX_LOG_ENTRIES = 128  

def _default_wall():
    lst = list(range(34)) * 4 + list(range(34, 42))
    random.shuffle(lst)
    return lst

@dataclass
class GameState:
    round_wind: int
    game_wind: int
    current_player: int = EAST
    wall_remaining: int = 144
    phase: int = WAIT_TSUMO_ADD_KAN_AN_KAN

    # Player-specific data – now using np.ndarray
    hands: List[np.ndarray] = field(default_factory=lambda: [np.zeros(42, dtype=np.uint8) for _ in range(4)])
    flowers: List[np.ndarray] = field(default_factory=lambda: [np.zeros(42, dtype=np.uint8) for _ in range(4)])
    melds: List[List[np.ndarray]] = field(default_factory=lambda: [[] for _ in range(4)])
    
    # Log: each row is [tile one‑hot (42) + player one‑hot (4)]
    log: np.ndarray = field(default_factory=lambda: np.zeros((MAX_LOG_ENTRIES, 42 + 4), dtype=np.uint8))
    logline: int = 0

    # Convenience:
    last_discard: int = -1
    last_drawn: int = -1
    addkanable_tiles: List[Dict[int, int]] = field(default_factory=lambda: [{}, {}, {}, {}])
    men_qian_qing: List[bool] = field(default_factory=lambda: [True for _ in range(4)])
    action_array: np.ndarray  = field(default_factory=lambda: np.zeros(shape=(75,)))
    # Wall
    wall: List[int] = field(default_factory=_default_wall)
    
def melds_to_array(lst: List[np.ndarray]) -> np.ndarray:
    pad_array = np.zeros(42)
    n = len(lst)
    pads = [pad_array for _ in range(4 - n)]
    return np.stack([*lst, *pads]) if n < 4 else np.stack(lst)

def game_state_mask(game: GameState, player_idx: int) -> np.ndarray:
    # round wind, game wind, current player, wall remaining,
    metadata_array = np.zeros(shape=(4, 42 + 4), dtype=np.uint8)
    metadata_array[0, game.round_wind + 27] = 4
    metadata_array[1, game.game_wind + 27] = 4
    metadata_array[2, game.current_player + 42] = 1
    metadata_array[3, :] = game.wall_remaining

    hands_array = np.zeros(shape=(4, 42), dtype=np.uint8)
    hands_array[player_idx] = game.hands[player_idx]
    hands_array = np.hstack([hands_array, np.identity(4)])

    meld_player = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],

        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],

        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],

        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])
    melds_array = np.hstack([np.vstack([melds_to_array(game.melds[i]) for i in range(4)]), meld_player])
    flowers_array = np.hstack([np.stack(game.flowers), np.identity(4, dtype=np.uint8)])
    return np.vstack([
        metadata_array,
        hands_array,
        melds_array,
        flowers_array,
        game.log
    ])

def game_state_mask_simplified(game: GameState, player_idx: int):
    """
    Returns a simplified observation
    """
    # metadata
    metadata_array = np.zeros(shape=(4, 34), dtype=np.uint8)
    metadata_array[0, game.round_wind + 27] = 4
    metadata_array[1, seat(player_idx, game.game_wind) + 27] = 4
    metadata_array[2, game.current_player + 27] = 1
    metadata_array[3, :] = game.wall_remaining / 144
    # one hand
    hands_array = game.hands[player_idx][:34]
    # flowers, use ESWN
    flowers_array = np.hstack([np.zeros(shape=(4, 27)), np.stack(game.flowers)[:, 34:38] + np.stack(game.flowers)[:, 38:42], np.zeros(shape=(4, 3))])
    # melds
    melds_array = np.vstack([melds_to_array(game.melds[i]) for i in range(4)])[:, :34]
    # discard
    # from log add up all discards
    discards = np.zeros(shape=(4, 34))
    line_number = 0
    while game.log[line_number].sum() > 0:
        if game.log[line_number].sum() == 2 and not is_subsequently_called(game.log, line_number) and game.log[line_number][:34].sum() == 1:
            idxs = np.nonzero(game.log[line_number])[0]
            if idxs.size >= 2:
                tile = int(idxs[0])
                player_idx = int(idxs[1]) - 42
                # guard: ensure valid player index
                if 0 <= player_idx < 4:
                    discards[player_idx, tile] += 1
        line_number += 1
    return np.vstack([
        metadata_array,
        hands_array,
        melds_array,
        flowers_array,
        discards
    ]) 

def game_state_array(game: GameState) -> np.ndarray:
    # round wind, game wind, current player, wall remaining,
    metadata_array = np.zeros(shape=(4, 42 + 4), dtype=np.uint8)
    metadata_array[0, game.round_wind] = 4
    metadata_array[1, game.game_wind] = 4
    metadata_array[2, game.current_player + 42] = 1
    metadata_array[3, :] = game.wall_remaining

    hands_array = np.hstack([melds_to_array(game.hands), np.identity(4)])

    meld_player = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    melds_array = np.hstack([np.vstack([melds_to_array(game.melds[i]) for i in range(4)]), meld_player])
    flowers_array = np.hstack([np.stack(game.flowers), np.identity(4, dtype=np.uint8)])
    return np.vstack([
        metadata_array,
        hands_array,
        melds_array,
        flowers_array,
        game.log
    ])

def game_state_array_simplified(game: GameState):
    """
    Returns a simplified observation
    """
    # metadata
    metadata_array = np.zeros(shape=(4, 34), dtype=np.uint8)
    metadata_array[0, game.round_wind + 27] = 4
    metadata_array[1, game.game_wind + 27] = 4
    metadata_array[2, game.current_player + 27] = 1
    metadata_array[3, :] = game.wall_remaining / 144
    # four hand
    hands_array = melds_to_array(game.hands)[:, :34]
    # flowers, use ESWN
    flowers_array = np.hstack([np.zeros(shape=(4, 27)), np.stack(game.flowers)[:, 34:38] + np.stack(game.flowers)[:, 38:42], np.zeros(shape=(4, 3))])
    # melds
    melds_array = np.vstack([melds_to_array(game.melds[i]) for i in range(4)])[:, :34]
    # discard
    # from log add up all discards
    discards = np.zeros(shape=(4, 34))
    line_number = 0
    while game.log[line_number].sum() > 0:
        if game.log[line_number].sum() == 2 and not is_subsequently_called(game.log, line_number) and game.log[line_number][:34].sum() == 1:
            idxs = np.nonzero(game.log[line_number])[0]
            if idxs.size >= 2:
                tile = int(idxs[0])
                player_idx = int(idxs[1]) - 42
                # guard: ensure valid player index
                if 0 <= player_idx < 4:
                    discards[player_idx, tile] += 1
        line_number += 1
    return np.vstack([
        metadata_array,
        hands_array,
        melds_array,
        flowers_array,
        discards
    ]) 

def is_subsequently_called(log: np.ndarray, logline: int) -> bool:
    line = log[logline]
    subsequent_line = log[logline + 1]
    if subsequent_line.sum() == 0:
        return False
    # Convert np.nonzero to get first index of non‑zero element
    discarded_tile = np.nonzero(line)[0][0]
    # FIXED: previously both used 'line', now 'subsequent_line' for the called tile
    called_tile = np.nonzero(subsequent_line)[0][0]
    return True if (subsequent_line.sum() == 4 or (subsequent_line.sum() == 5 and discarded_tile == called_tile)) else False

def draw_tile(gamestate: GameState, player_idx: int, wall: List[int]) -> int:
    """
    Affected:
    wall_remaining, wall, hands
    """
    tile = gamestate.wall.pop()
    gamestate.hands[player_idx][tile] += 1
    gamestate.wall_remaining = len(gamestate.wall)
    gamestate.last_drawn = tile
    return tile

def discard(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, log, logline, last_discard
    """
    assert gamestate.hands[player_idx][tile] >= 1
    gamestate.hands[player_idx][tile] -= 1
    gamestate.last_discard = tile
    gamestate.log[gamestate.logline][tile] += 1
    gamestate.log[gamestate.logline][player_idx + 42] += 1
    gamestate.logline += 1


def execute_discard(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, log, logline, last_discard
    """
    assert gamestate.hands[player_idx][tile] >= 1
    gamestate.hands[player_idx][tile] -= 1
    gamestate.last_discard = tile
    gamestate.log[gamestate.logline][tile] += 1
    gamestate.log[gamestate.logline][player_idx + 42] += 1
    gamestate.logline += 1

def execute_pon(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, melds, log, logline, addkanable_tiles, men_qian_qing
    """
    assert len(gamestate.melds[player_idx]) < 4
    assert gamestate.hands[player_idx][tile] >= 2
    gamestate.hands[player_idx][tile] -= 2
    meld = np.zeros(42, dtype=np.uint8)
    meld[tile] += 3
    gamestate.melds[player_idx].append(meld)
    gamestate.log[gamestate.logline][tile] += 3
    gamestate.log[gamestate.logline][42 + player_idx] += 1
    gamestate.logline += 1
    gamestate.addkanable_tiles[player_idx][tile] = len(gamestate.melds[player_idx]) - 1
    gamestate.men_qian_qing[player_idx] = False


def execute_chow(gamestate: GameState, player_idx: int, tile: int, tiles_for_chow: List[int]) -> None:
    """
    Affected:
    hands, melds, log, logline, men_qian_qing
    """
    assert len(gamestate.melds[player_idx]) < 4
    assert len(tiles_for_chow) == 2
    for chow_tile in tiles_for_chow:
        assert gamestate.hands[player_idx][chow_tile] >= 1

    for chow_tile in tiles_for_chow:
        gamestate.hands[player_idx][chow_tile] -= 1


    meld = np.zeros(42, dtype=np.uint8)
    meld[tile] += 1
    for chow_tile in tiles_for_chow:
        meld[chow_tile] += 1

    gamestate.melds[player_idx].append(meld)
    for idx in [tile, *tiles_for_chow]:
        gamestate.log[gamestate.logline][idx] += 1
    gamestate.log[gamestate.logline][42 + player_idx] += 1
    gamestate.logline += 1
    gamestate.men_qian_qing[player_idx] = False


def execute_ming_kan(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, melds, log, logline, men_qian_qing
    """
    assert len(gamestate.melds[player_idx]) < 4
    assert gamestate.hands[player_idx][tile] >= 3
    for _ in range(3):
        gamestate.hands[player_idx][tile] -= 1

    meld = np.zeros(42, dtype=np.uint8)
    meld[tile] += 4
    gamestate.melds[player_idx].append(meld)
    gamestate.log[gamestate.logline][tile] += 4
    gamestate.log[gamestate.logline][42 + player_idx] += 1
    gamestate.logline += 1
    gamestate.men_qian_qing[player_idx] = False


def execute_an_kan(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, melds, log, logline
    """
    assert gamestate.hands[player_idx][tile] >= 4
    assert len(gamestate.melds[player_idx]) < 4
    for _ in range(4):
        gamestate.hands[player_idx][tile] -= 1

    meld = np.zeros(42, dtype=np.uint8)
    meld[tile] += 4
    gamestate.melds[player_idx].append(meld)
    gamestate.log[gamestate.logline][tile] += 4
    gamestate.log[gamestate.logline][42 + player_idx] += 1
    gamestate.logline += 1

def wall_sequence_array(wall: List[int], max_len: int = 92) -> np.ndarray:
    """
    Convert the remaining wall (list) into a fixed‑length one‑hot sequence.
    The sequence is in **reverse order**:
      - Row 0 corresponds to the last tile in `wall` (the next tile to be drawn).
      - Row 1 corresponds to the second‑last, etc.
    Missing tiles (if wall is shorter than max_len) are zero‑padded.
    Returns: (max_len, 46) np.ndarray, dtype=uint8
    """
    seq = np.zeros((max_len, 46), dtype=np.uint8)
    for i, tile in enumerate(reversed(wall)):
        if i >= max_len:
            break
        seq[i, tile] = 1
    return seq

def wall_sequence_array_simplified(wall: List[int], max_len: int = 20) -> np.ndarray:
    """
    Convert the remaining wall (list) into a fixed‑length one‑hot sequence.
    The sequence is in **reverse order**:
      - Row 0 corresponds to the last tile in `wall` (the next tile to be drawn).
      - Row 1 corresponds to the second‑last, etc.
    Missing tiles (if wall is shorter than max_len) are zero‑padded.
    Returns: (max_len, 46) np.ndarray, dtype=uint8
    """
    seq = np.zeros((max_len, 42), dtype=np.uint8)
    for i, tile in enumerate(reversed(wall)):
        if i >= max_len:
            break
        seq[i, tile] = 1
    return seq

def execute_add_kan(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, melds, log, logline
    """
    assert gamestate.hands[player_idx][tile] >= 1
    assert tile in gamestate.addkanable_tiles[player_idx]

    gamestate.hands[player_idx][tile] -= 1
    meld_index = gamestate.addkanable_tiles[player_idx][tile]
    gamestate.melds[player_idx][meld_index][tile] += 1
    gamestate.log[gamestate.logline][tile] += 1
    gamestate.log[gamestate.logline][42 + player_idx] += 1
    gamestate.logline += 1


def execute_flower(gamestate: GameState, player_idx: int, tile: int) -> None:
    """
    Affected:
    hands, flowers, log, logline
    """
    assert is_flower(tile)
    gamestate.hands[player_idx][tile] -= 1
    gamestate.flowers[player_idx][tile] += 1
    gamestate.log[gamestate.logline][tile] += 1
    gamestate.log[gamestate.logline][42 + player_idx] += 1
    gamestate.logline += 1

def hand_array_to_string(hand: npt.NDArray):
    res = []
    for tile, count in enumerate(hand[:34]):   
        [res.append(tiles_name[tile]) for _ in range(count)]
    return res