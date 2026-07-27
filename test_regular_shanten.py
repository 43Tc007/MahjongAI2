import numpy as np

from hand_divisor import regular_shanten, first_discard_shanten, orphan_shanten
import random
if __name__ == "__main__":
    arr = np.zeros(34, dtype=np.uint8)
    for _ in range(14):
        tile = random.randint(0, 33)
        arr[tile] += 1
    copy = np.copy(arr)
    from mahjong_helper import hand_array_to_string
    print(hand_array_to_string(arr))
    print('Regular shanten', first_discard_shanten(arr))
    print('Orphan shanten', orphan_shanten(arr))
    print(np.all(copy == arr))
