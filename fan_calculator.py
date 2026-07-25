from hand_divisor import divide_from_tensors, CHOW, PUNG, PAIR
from mahjong_helper import *
from typing import List, Tuple
import numpy as np


def flowers(flowers: np.ndarray, player: int, game_wind: int) -> int:
    def wu_hua(flowers: np.ndarray) -> int:
        return int((flowers[34:] == 0).all().item())

    def red_zheng_hua(flowers: np.ndarray, player: int, game_wind: int) -> int:
        return int((flowers[34 + seat(player, game_wind)] == 1).item())
    
    def black_zheng_hua(flowers: np.ndarray, player: int, game_wind: int) -> int:
        return int((flowers[38 + seat(player, game_wind)] == 1).item())
    
    def red_yi_tai_hua(flowers: np.ndarray) -> int:
        return int((flowers[34:37] == 1).all().item())
    
    def black_yi_tai_hua(flowers:np.ndarray) -> int:
        return int((flowers[38:41] == 1).all().item())
    
    return wu_hua(flowers) + red_zheng_hua(flowers, player, game_wind) + black_zheng_hua(flowers, player, game_wind) + red_yi_tai_hua(flowers) + black_yi_tai_hua(flowers)

# --- main hand ---

def ping_hu(division: List[Tuple[int, int]]) -> int:
    for pack_type, tile in division:
        if pack_type == PUNG:
            return 0
    return 1

def dui_dui_hu(division: List[Tuple[int, int]]) -> int:
    for pack_type, tile in division:
        if pack_type == CHOW:
            return 0
    return 3

def fan_pai(division: List[Tuple[int, int]], player: int, round_wind: int, game_wind: int) -> int:
    res = 0
    seat_wind = seat(player, game_wind)
    targets = [seat_wind + 27, round_wind + 27, 31, 32, 33]
    for pack_type, tile in division:
        if tile in targets and pack_type == PUNG:
            res += 1
    return res

def hua_yao(division: List[Tuple[int, int]]) -> int:
    for _, tile in division:
        if not (is_19(tile) or is_zi(tile)):
            return 0
    return 1

def qing_yao(division: List[Tuple[int, int]]) -> int:
    for _, tile in division:
        if not is_19(tile):
            return 0
    return 13

def da_xiao_san_yuan(division: List[Tuple[int, int]]) -> int:
    seen = 0
    pair_is_dragon = False
    for pack_type, tile in division:
        if is_dragon(tile):
            seen += 1
            if pack_type == PAIR:
                pair_is_dragon = True
        
    if seen != 3:
        return 0
    if pair_is_dragon:
        return 3 # 3+2=5
    return 13

def qing_hun_yi_se(division: List[Tuple[int, int]]) -> int:
    suit = -1
    zi_present = False
    for _, tile in division:
        if is_zi(tile):
            zi_present = True
            continue
        if suit == -1:
            suit = tile // 9
            continue
        if suit != tile // 9:
            return 0
    return 3 if zi_present else 7

def zi_yi_se(division: List[Tuple[int, int]]) -> int:
    for _, tile in division:
        if not is_zi(tile):
            return 0
    return 13

def da_xiao_si_xi(division: List[Tuple[int, int]]) -> int:
    seen = 0
    for _, tile in division:
        if is_wind(tile):
            seen += 1
    if seen != 4:
        return 0
    return 13

def jiu_zi_lian_huan(hand: np.ndarray) -> int:
    if hand.sum() != 14:
        return 0
    for base in (0, 9, 18):
        s = hand[base:base + 9]
        if s.sum() != 14:
            continue
        if s[0] >= 3 and (s[1:8] >= 1).all().item() and s[8] >= 3:
            return 13  # base is 0, 9, or 18, identifying the suit
    return 0

def si_gang_zi(calls: np.ndarray) -> int:
    return 13 * int((calls.sum(axis=1) == 4).all().item())

def shi_san_yao(hand: np.ndarray) -> int:
    indices = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    return int((hand[indices] >= 1).all().item()) * 13

def tsumo(game: GameState, player: int) -> int:
    return (game.current_player == player) * 1

def tian_hu(game: GameState, player: int) -> int:
    if player != EAST:
        return 0
    return int((game.log[:, :42] == 0).all().item()) * 13

def di_hu(game: GameState, player: int) -> int:
    if tsumo(game, player):
        return 0
    return int((game.log[:, :42].sum() == 1).item()) * 13

def men_qian_qing(game: GameState, player: int) -> int:
    return int(game.men_qian_qing[player]) * 1

def hai_di_lao_yue(game: GameState) -> int:
    return (game.wall_remaining == 0) * 1

def qiang_gang(game: GameState, player: int) -> int:
    return (game.log[game.logline, :42].sum().item() == 4 and game.current_player != player) * 1

def gang_shang_kai_hua(game: GameState, player: int) -> int:
    return (game.log[game.logline, :42].sum().item() == 4 and game.current_player == player) * 1

def kan_kan_hu(game: GameState, player: int, division: List[Tuple[int, int]], win_tile: int) -> int:
    if not dui_dui_hu(division) or not men_qian_qing(game, player):
        return 0
    if tsumo(game, player):
        return 13
    else:
        pair_tile: int = [tile for pack_type, tile in division if pack_type == PAIR][0]
        return (win_tile == pair_tile) * 13

def hua_hu(game: GameState, player: int, win_tile: int) -> int:
    if is_flower(win_tile):
        return int((game.flowers[player].sum() == 6) * 3 + (game.flowers[player].sum() == 7) * 13)
    return 0

def calculate_fan(


    game: GameState, player: int, win_tile: int, verbose: bool =False) -> int:

    # Special hands that immediately return (they override everything else)
    if hua_hu(game, player, win_tile):
        if verbose:
            print("hua_hu:", hua_hu(game, player, win_tile))
        return hua_hu(game, player, win_tile)
    if shi_san_yao(game.hands[player]):
        if verbose:
            print("shi_san_yao:", shi_san_yao(game.hands[player]))
        return 13

    success, divisions = divide_from_tensors(game.hands[player], game.melds[player])
    if not success or len(divisions) == 0:
        return 0

     
    # ---- Compute base yaku (independent of division) ----
    yaku = {}

    # Flower related yaku (we'll combine them, but you can break down further)
    flower_val = flowers(game.flowers[player], player, game.game_wind)
    if flower_val:
        yaku["flowers (combined)"] = flower_val

    val = jiu_zi_lian_huan(game.hands[player])
    if val:
        yaku["jiu_zi_lian_huan"] = val

    val = si_gang_zi(melds_to_array(game.melds[player]))
    if val:
        yaku["si_gang_zi"] = val

    val = tsumo(game, player)
    if val:
        yaku["tsumo"] = val

    val = tian_hu(game, player)
    if val:
        yaku["tian_hu"] = val

    val = di_hu(game, player)
    if val:
        yaku["di_hu"] = val

    val = men_qian_qing(game, player)
    if val:
        yaku["men_qian_qing"] = val

    val = hai_di_lao_yue(game)
    if val:
        yaku["hai_di_lao_yue"] = val

    val = qiang_gang(game, player)
    if val:
        yaku["qiang_gang"] = val

    val = gang_shang_kai_hua(game, player)
    if val:
        yaku["gang_shang_kai_hua"] = val

    # ---- Division-dependent yaku ----
    max_fan = 0
    best_yaku = {}
    for division in divisions:
        div_yaku = {}

        # Evaluate each division‑dependent yaku
        val = ping_hu(division)
        if val:
            div_yaku["ping_hu"] = val

        val = dui_dui_hu(division)
        if val:
            div_yaku["dui_dui_hu"] = val

        val = fan_pai(division, player, game.round_wind, game.game_wind)
        if val:
            div_yaku["fan_pai"] = val

        val = hua_yao(division)
        if val:
            div_yaku["hua_yao"] = val

        val = qing_yao(division)
        if val:
            div_yaku["qing_yao"] = val

        val = da_xiao_san_yuan(division)
        if val:
            div_yaku["da_xiao_san_yuan"] = val

        val = qing_hun_yi_se(division)
        if val:
            div_yaku["qing_hun_yi_se"] = val

        val = zi_yi_se(division)
        if val:
            div_yaku["zi_yi_se"] = val

        val = da_xiao_si_xi(division)
        if val:
            div_yaku["da_xiao_si_xi"] = val

        val = kan_kan_hu(game, player, division, win_tile)
        if val:
            div_yaku["kan_kan_hu"] = val

        # Total fan for this division (base + div)
        temp_fan = sum(yaku.values()) + sum(div_yaku.values())
        if temp_fan > max_fan:
            max_fan = temp_fan
            best_yaku = {**yaku, **div_yaku}   # merge dictionaries

    # ---- Print the positive yaku for the chosen division ----
    if best_yaku:
        if verbose:
            print("Yaku counted (with values):")
            for name, value in best_yaku.items():
                print(f"  {name}: {value}")
    else:
        if verbose:
            print("No yaku (only 0‑fan hands).")

    return min(13, max_fan)



if __name__ == '__main__':
    gamestate = GameState(
        round_wind=EAST,
        game_wind=SOUTH,
        current_player=WEST,
        wall_remaining=120,
        phase=WAIT_TSUMO_ADD_KAN_AN_KAN,
    )

    # Fill in some arbitrary values
    temp = np.array([
        21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 7, 9, 10, 11
    ])
    for thing in temp:
        gamestate.hands[0][thing] += 1
    gamestate.hands[1][10] = 1  # Player 1 has one tile index 10
    gamestate.flowers[2][40] = 1
    gamestate.melds[3] = [np.array([1,2,3], dtype=np.uint8)]
    gamestate.last_discard = 7
    gamestate.last_drawn = 12
    gamestate.addkanable_tiles[0] = {7: 1}
    gamestate.men_qian_qing = [False, False, False, False]
    gamestate.action_array[0] = 1
    gamestate.wall = list(range(100))  # shorter wall for demo
    gamestate.logline = 5
    gamestate.log[gamestate.logline] = np.ones(46, dtype=np.uint8)
    x = calculate_fan(gamestate, 0, gamestate.last_discard, verbose=True)
    print(x)