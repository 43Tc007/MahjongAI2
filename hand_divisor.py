import numpy as np
from typing import List, Optional, Tuple, cast
from collections import Counter
from functools import lru_cache

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TILE_COUNT = 34
ALL_TILES = list(range(TILE_COUNT))
PAIR, PUNG, CHOW = 0, 1, 2
DUO = 3

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
    elif typ == DUO:
        return f"duo({TILE_NAMES[tile]}-{TILE_NAMES[tile+1]})"
    else:
        return "unknown"


def orphan_shanten(hand_array: np.ndarray) -> int:
    if hand_array.sum() != 13:
        return 8
    counts = [int(x) for x in hand_array[:TILE_COUNT]]
    orphan_indices = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    present = [idx for idx in orphan_indices if counts[idx] > 0]
    distinct = len(present)
    has_pair = any(counts[idx] >= 2 for idx in orphan_indices)
    return 13 - distinct - int(has_pair)


def regular_shanten(hand_array: np.ndarray) -> int:
    """
    Estimate regular shanten by decomposing the hand into the best combination of
    melds, pairs and consecutive duos.

    A complete meld is either:
    - a pung (three identical tiles)
    - a chow (three consecutive tiles of the same suit)

    A four-of-a-kind is handled as a pung plus one leftover tile, which is still
    available for later pairing or chow construction.

    A pair is two identical tiles, a duo is two consecutive tiles of the same
    numbered suit, and a gap part is two tiles separated by one missing tile
    (for example 3s and 5s). Each meld is worth two points, while a pair, duo,
    or gap part is worth one. Only the best five parts contribute to the score.
    """
    # The hand array is a 34-entry count vector. Indices 0-26 are numbered tiles
    # (0-8 = man, 9-17 = pin, 18-26 = sou), while 27-33 are honours.
    counts = [int(x) for x in hand_array[:TILE_COUNT]]
    tile_count = int(sum(counts))
    if tile_count not in {13, 10, 7, 4, 1}:
        raise ValueError(f"hand size must be one of 13, 10, 7, 4, or 1 for this regular shanten computation. Got {tile_count}")

    fixed_melds = (13 - tile_count) // 3

    @lru_cache(maxsize=None)
    def best_part_score(counts_tuple: Tuple[int, ...], parts_used: int) -> int:
        if parts_used >= 5:
            return 0

        best_score = 0
        counts = list(counts_tuple)

        for tile in ALL_TILES:
            # A four-of-a-kind is treated as a pung plus one spare tile that can still
            # participate in a later duo or chow.
            if counts[tile] >= 3:
                counts[tile] -= 3
                best_score = max(best_score, 2 + best_part_score(tuple(counts), parts_used + 1))
                counts[tile] += 3

        for tile in range(TILE_COUNT):
            if not is_numbered_suit(tile):
                continue
            rank = tile % 9
            if rank <= 6 and counts[tile] > 0 and counts[tile + 1] > 0 and counts[tile + 2] > 0:
                counts[tile] -= 1
                counts[tile + 1] -= 1
                counts[tile + 2] -= 1
                best_score = max(best_score, 2 + best_part_score(tuple(counts), parts_used + 1))
                counts[tile] += 1
                counts[tile + 1] += 1
                counts[tile + 2] += 1

        for tile in ALL_TILES:
            if counts[tile] >= 2:
                counts[tile] -= 2
                best_score = max(best_score, 1 + best_part_score(tuple(counts), parts_used + 1))
                counts[tile] += 2

        for tile in range(TILE_COUNT):
            if not is_numbered_suit(tile):
                continue
            rank = tile % 9
            if rank <= 7 and counts[tile] > 0 and counts[tile + 1] > 0:
                counts[tile] -= 1
                counts[tile + 1] -= 1
                best_score = max(best_score, 1 + best_part_score(tuple(counts), parts_used + 1))
                counts[tile] += 1
                counts[tile + 1] += 1

            if rank <= 6 and counts[tile] > 0 and counts[tile + 2] > 0:
                counts[tile] -= 1
                counts[tile + 2] -= 1
                best_score = max(best_score, 1 + best_part_score(tuple(counts), parts_used + 1))
                counts[tile] += 1
                counts[tile + 2] += 1

        return best_score

    points = best_part_score(tuple(counts), 0)
    return 8 - 2 * fixed_melds - points


def shanten(hand_array: np.ndarray) -> int:
    """Return the minimum of regular shanten and orphan shanten."""
    if hand_array.sum() % 3 == 1:
        return min(regular_shanten(hand_array), orphan_shanten(hand_array))
    assert hand_array.sum() % 3 == 2
    return first_discard_shanten(hand_array)


def first_discard_shanten(hand_array: np.ndarray) -> int:
    """
    For a 14-tile hand, compute the minimum shanten value obtainable by discarding
    one tile and then evaluating the resulting 13-tile hand.
    """
    counts = [int(x) for x in hand_array[:TILE_COUNT]]
    tile_count = int(sum(counts))
    if tile_count != 14:
        raise ValueError("first_discard_shanten expects a 14-tile hand")

    best_value = None
    for tile in range(TILE_COUNT):
        if counts[tile] == 0:
            continue
        counts[tile] -= 1
        reduced = np.zeros(TILE_COUNT, dtype=np.int64)
        reduced[:TILE_COUNT] = counts
        value = shanten(reduced)
        if best_value is None or value < best_value:
            best_value = value
        counts[tile] += 1

    if best_value is None:
        raise ValueError("hand must contain at least one tile to discard")
    return best_value

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

