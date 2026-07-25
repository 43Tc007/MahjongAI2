import numpy as np
from typing import List, Optional, Tuple, cast
from collections import Counter

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TILE_COUNT = 34
ALL_TILES = list(range(TILE_COUNT))
PAIR, PUNG, CHOW = 0, 1, 2

TILE_NAMES = {
    0: "1m", 1: "2m", 2: "3m", 3: "4m", 4: "5m", 5: "6m", 6: "7m", 7: "8m", 8: "9m",
    9: "1p", 10: "2p", 11: "3p", 12: "4p", 13: "5p", 14: "6p", 15: "7p", 16: "8p", 17: "9p",
    18: "1s", 19: "2s", 20: "3s", 21: "4s", 22: "5s", 23: "6s", 24: "7s", 25: "8s", 26: "9s",
    27: "E", 28: "S", 29: "W", 30: "N",
    31: "white", 32: "green", 33: "red"
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def make_pack(pack_type: int, tile: int) -> Tuple[int, int]:
    return (pack_type, tile)

def is_numbered_suit(t: int) -> bool:
    return t < 27

def pack_key(p: Tuple[int, int]) -> Tuple[int, int]:
    return (p[1], p[0])

def pack_to_str(p: Tuple[int, int]) -> str:
    typ, tile = p
    if typ == PAIR:
        return f"pair({TILE_NAMES[tile]})"
    elif typ == PUNG:
        return f"pung({TILE_NAMES[tile]})"   # could also be a kan, but we don't distinguish
    elif typ == CHOW:
        return f"chow({TILE_NAMES[tile-1]}-{TILE_NAMES[tile]}-{TILE_NAMES[tile+1]})"
    else:
        return "unknown"

# ------------------------------------------------------------------
# Core recursive algorithm (unchanged)
# ------------------------------------------------------------------
def divide_tail_add_division(fixed_cnt: int, work_division: List[Optional[Tuple[int, int]]],
                             result: List[List[Tuple[int, int]]]) -> None:
    temp = work_division[:]
    assert all(entry is not None for entry in temp)
    melds = temp[fixed_cnt:4]
    melds_t = cast(List[Tuple[int, int]], melds)
    melds_sorted = sorted(melds_t, key=pack_key)
    temp[fixed_cnt:4] = melds_sorted
    for d in result:
        existing_melds = d[fixed_cnt:4]
        existing_sorted = sorted(existing_melds, key=pack_key)
        if existing_sorted == melds_sorted:
            return
    result.append(cast(List[Tuple[int, int]], temp))

def divide_tail(cnt_table: List[int], fixed_cnt: int,
                work_division: List[Optional[Tuple[int, int]]],
                result: List[List[Tuple[int, int]]]) -> bool:
    for t in ALL_TILES:
        if cnt_table[t] >= 2:
            cnt_table[t] -= 2
            if all(c == 0 for c in cnt_table):
                work_division[4] = make_pack(PAIR, t)
                divide_tail_add_division(fixed_cnt, work_division, result)
                cnt_table[t] += 2
                return True
            cnt_table[t] += 2
    return False

def is_division_branch_exist(fixed_cnt: int, step: int,
                             work_division: List[Optional[Tuple[int, int]]],
                             result: List[List[Tuple[int, int]]]) -> bool:
    if not result or step < 3:
        return False
    current_melds = work_division[fixed_cnt:fixed_cnt+step]
    assert all(entry is not None for entry in current_melds)
    current_sorted = sorted(cast(List[Tuple[int, int]], current_melds), key=pack_key)
    current_counter = Counter(current_sorted)
    for d in result:
        existing_melds = d[fixed_cnt:4]
        existing_sorted = sorted(existing_melds, key=pack_key)
        existing_counter = Counter(existing_sorted)
        if all(current_counter[k] <= existing_counter[k] for k in current_counter):
            return True
    return False

def divide_recursively(cnt_table: List[int], fixed_cnt: int, step: int,
                       work_division: List[Optional[Tuple[int, int]]],
                       result: List[List[Tuple[int, int]]]) -> bool:
    idx = step + fixed_cnt
    if idx == 4:
        return divide_tail(cnt_table, fixed_cnt, work_division, result)

    ret = False
    for t in ALL_TILES:
        if cnt_table[t] == 0:
            continue
        # Pung
        if cnt_table[t] >= 3:
            work_division[idx] = make_pack(PUNG, t)
            if not is_division_branch_exist(fixed_cnt, step+1, work_division, result):
                cnt_table[t] -= 3
                if divide_recursively(cnt_table, fixed_cnt, step+1, work_division, result):
                    ret = True
                cnt_table[t] += 3
        # Chow
        if is_numbered_suit(t):
            rank = t % 9
            if rank <= 6 and cnt_table[t+1] > 0 and cnt_table[t+2] > 0:
                work_division[idx] = make_pack(CHOW, t+1)
                if not is_division_branch_exist(fixed_cnt, step+1, work_division, result):
                    cnt_table[t] -= 1
                    cnt_table[t+1] -= 1
                    cnt_table[t+2] -= 1
                    if divide_recursively(cnt_table, fixed_cnt, step+1, work_division, result):
                        ret = True
                    cnt_table[t] += 1
                    cnt_table[t+1] += 1
                    cnt_table[t+2] += 1
    return ret

def divide_win_hand(cnt_table: List[int],
                    fixed_packs: Optional[List[Tuple[int, int]]] = None) -> Tuple[bool, List[List[Tuple[int, int]]]]:
    if fixed_packs is None:
        fixed_packs = []
    fixed_cnt = len(fixed_packs)
    result: List[List[Tuple[int, int]]] = []
    work_division: List[Optional[Tuple[int, int]]] = fixed_packs + [None] * (5 - fixed_cnt)
    success = divide_recursively(cnt_table, fixed_cnt, 0, work_division, result)
    return success, result

# ------------------------------------------------------------------
# PyTorch wrapper – now handles kan as a special pung
# ------------------------------------------------------------------
def divide_from_tensors(hand_array: np.ndarray,
                        fixed_melds_tensor: List[np.ndarray]) -> Tuple[bool, List[List[Tuple[int, int]]]]:
    """
    Parameters:
        hand_array       : shape (42,) – counts of standing tiles (only indices 0..33 used).
        fixed_melds_tensor: shape (4, 42) – each row is a count vector for a fixed meld.
                           A zero row is ignored.
                           Supported meld types:
                           - pung  : sum == 3 (one tile has count 3)
                           - kan   : sum == 4 (one tile has count 4) → treated as a pung
                           - chow  : three consecutive tiles each count = 1 (numbered suits only)
    Returns:
        (success, divisions)
    """
    # Standing hand count table (directly from the array)
    cnt_table: List[int] = [int(x) for x in hand_array[:TILE_COUNT]]

    fixed_packs: List[Tuple[int, int]] = []
    for row in fixed_melds_tensor:
        row_counts: List[int] = [int(x) for x in row[:TILE_COUNT]]
        total = sum(row_counts)
        if total == 0:
            continue  # empty row – ignore

        # Check for pung or kan: one tile count == 3 or 4
        meld_tile = None
        for tile, cnt in enumerate(row_counts):
            if cnt == 3 or cnt == 4:
                meld_tile = tile
                break
        if meld_tile is not None:
            # Treat both pung and kan as PUNG (the extra tile is ignored because it's already called)
            fixed_packs.append(make_pack(PUNG, meld_tile))
            continue

        # Check for chow: three consecutive 1s
        chow_found = False
        for tile in range(TILE_COUNT):
            if is_numbered_suit(tile):
                rank = tile % 9
                if rank <= 6:
                    if (row_counts[tile] == 1 and
                        row_counts[tile+1] == 1 and
                        row_counts[tile+2] == 1):
                        fixed_packs.append(make_pack(CHOW, tile+1))
                        chow_found = True
                        break
        if not chow_found:
            raise ValueError(f"Invalid fixed meld row: {row_counts}")

    # Call the core algorithm
    return divide_win_hand(cnt_table, fixed_packs)

