from mahjong_helper import GameState, is_subsequently_called
import pygame
import numpy as np
from typing import List, Tuple

tiles_unicode = {
    # Manzu (characters)
    0: "\U0001F007", 1: "\U0001F008", 2: "\U0001F009", 3: "\U0001F00A", 4: "\U0001F00B",
    5: "\U0001F00C", 6: "\U0001F00D", 7: "\U0001F00E", 8: "\U0001F00F",

    # Pinzu (dots / circles)
    9: "\U0001F019", 10: "\U0001F01A", 11: "\U0001F01B", 12: "\U0001F01C", 13: "\U0001F01D",
    14: "\U0001F01E", 15: "\U0001F01F", 16: "\U0001F020", 17: "\U0001F021",

    # Souzu (bamboo)
    18: "\U0001F010", 19: "\U0001F011", 20: "\U0001F012", 21: "\U0001F013", 22: "\U0001F014",
    23: "\U0001F015", 24: "\U0001F016", 25: "\U0001F017", 26: "\U0001F018",

    # Winds
    27: "\U0001F000",  # East
    28: "\U0001F001",  # South
    29: "\U0001F002",  # West
    30: "\U0001F003",  # North

    # Dragons
    31: "\U0001F006",  # White
    32: "\U0001F005",  # Green
    33: "🀄",  # Red

    # Flowers (red set)
    34: "\U0001F022",  # Plum
    35: "\U0001F023",  # Orchid
    36: "\U0001F024",  # Bamboo
    37: "\U0001F025",  # Chrysanthemum

    # Flowers (black set / seasons)
    38: "\U0001F026",  # Spring
    39: "\U0001F027",  # Summer
    40: "\U0001F028",  # Autumn
    41: "\U0001F029",  # Winter
}


def array_to_tile_string(tile_counter: np.ndarray) -> str:
    """
    Convert a torch.Tensor([42]) tile counter into a concatenated string of Mahjong Unicode tiles.
    """
    result: list[str] = []
    for idx, count_tensor in enumerate(tile_counter):
        count: int = int(count_tensor.item())  # explicit conversion for type safety
        if count > 0:
            result.extend([tiles_unicode[idx]] * count)
    return "".join(result)

def melds_to_string(melds: List[np.ndarray]) -> str:
    """
    Convert a list of meld tensors into a single string,
    with each meld separated by a space.
    """
    return " ".join(array_to_tile_string(meld) for meld in melds)

def string_to_surface_rect(s: str, center: Tuple[int, int], angle: int, font: pygame.font.Font):
    surface = font.render(s, False, 'black')
    surface = pygame.transform.rotozoom(surface, angle, 1)
    rect = surface.get_rect(center=center)
    return surface, rect

def hands_to_surface_rect(hand: np.ndarray, player_idx: int, font: pygame.font.Font):
    positions = [
        (400, 750),
        (750, 400),
        (400, 50),
        (50, 400)
    ]
    angles = [
        0, 
        90,
        180,
        270
    ]
    return string_to_surface_rect(
        array_to_tile_string(hand), 
        center=positions[player_idx], 
        angle=angles[player_idx],
        font=font
    )

def crop_surface(surface: pygame.Surface) -> pygame.Surface:
    """Return a new surface cropped to the non-transparent area."""
    bbox = surface.get_bounding_rect()
    if bbox.width == 0 or bbox.height == 0:
        return pygame.Surface((0, 0), pygame.SRCALPHA)
    return surface.subsurface(bbox).copy()

def discard_to_surf_rect(log: np.ndarray, font: pygame.font.Font):
    discards: List[List[str]] = [[] for _ in range(4)]
    line_number = 0
    while log[line_number].sum() > 0:
        if log[line_number].sum() == 2 and not is_subsequently_called(log, line_number) and log[line_number][:34].sum() == 1:
            idxs = np.nonzero(log[line_number])[0]
            if idxs.size >= 2:
                tile = int(idxs[0])
                player_idx = int(idxs[1]) - 42
                # guard: ensure valid player index
                if 0 <= player_idx < 4:
                    discards[player_idx].append(tiles_unicode[tile])
        line_number += 1
    discard_strings: List[str] = ["".join(discards[i]) for i in range(4)]

    # New positions & angles as requested
    discard_positions = [
        (300, 500),  # player 0 (bottom)
        (500, 500),  # player 1 (right)
        (500, 300),  # player 2 (top)
        (300, 300)   # player 3 (left)
    ]
    angles = [0, -90, -180, -270]          # rotation (negative for clockwise)
    anchors = ['topleft', 'bottomleft', 'bottomright', 'topright']

    surfaces_rects = []
    for i, dstr in enumerate(discard_strings):
        if not dstr:
            surf = pygame.Surface((0, 0), pygame.SRCALPHA)
            rect = surf.get_rect()
            setattr(rect, anchors[i], discard_positions[i])
            surfaces_rects.append((surf, rect))
            continue

        # Split into rows of 6 tiles
        chunk_size = 6
        chunks = [dstr[j:j+chunk_size] for j in range(0, len(dstr), chunk_size)]

        # Render each row, crop it tightly, and collect the cropped surfaces
        cropped_rows = []
        for chunk in chunks:
            row_surf = font.render(chunk, False, 'black')
            cropped_row = crop_surface(row_surf)
            cropped_rows.append(cropped_row)

        # Stack the cropped rows vertically with no extra space
        if not cropped_rows:
            continue
        max_width = max(r.get_width() for r in cropped_rows)
        total_height = sum(r.get_height() for r in cropped_rows)
        combined = pygame.Surface((max_width, total_height), pygame.SRCALPHA)
        y_offset = 0
        for row in cropped_rows:
            combined.blit(row, (0, y_offset))
            y_offset += row.get_height()

        # Rotate the combined surface (clockwise by the given angle)
        rotated = pygame.transform.rotate(combined, -angles[i])  # pygame rotates CCW

        # Crop the final rotated surface to remove any rotation‑added padding
        cropped = crop_surface(rotated)

        # Position the chosen corner exactly at the discard position
        rect = cropped.get_rect()
        setattr(rect, anchors[i], discard_positions[i])
        surfaces_rects.append((cropped, rect))

    return surfaces_rects

def melds_and_flowers_to_surface_rects(state: GameState, font: pygame.font.Font) -> List[Tuple[pygame.Surface, pygame.Rect]]:
    """Return a list of (surface, rect) for each player's melds and flowers combined."""
    results = []
    # Positions slightly inward from each player's hand
    positions = [
        (400, 700),   # player 0 (bottom)
        (700, 400),   # player 1 (right)
        (400, 100),   # player 2 (top)
        (100, 400)    # player 3 (left)
    ]
    angles = [0, 90, 180, 270]

    for player_idx in range(4):
        flower_tensor = state.flowers[player_idx]
        flower_str = array_to_tile_string(flower_tensor) if flower_tensor.sum() > 0 else ""

        melds = state.melds[player_idx]
        meld_str = " ".join(array_to_tile_string(m) for m in melds) if melds else ""

        combined = ""
        if flower_str:
            combined += flower_str
        if meld_str:
            if combined:
                combined += " "
            combined += meld_str

        if combined:
            surf, rect = string_to_surface_rect(
                combined,
                center=positions[player_idx],
                angle=angles[player_idx],
                font=font
            )
            results.append((surf, rect))
        # Skip players with nothing to display

    return results

def render_game_state(state: GameState, screen: pygame.Surface, font: pygame.font.Font):
   # game information
    square_rect = pygame.Rect(0, 0, 200, 200)
    square_rect.center = (400, 400)
    pygame.draw.rect(screen, 'grey', square_rect)
    game_info_string = \
    f"""{['E', 'S', 'W', 'N'][state.round_wind]} {state.game_wind + 1} {state.wall_remaining}"""
    game_info_surface = font.render(game_info_string, False, 'white')
    screen.blit(game_info_surface, square_rect)

    # discard
    for surf, rect in discard_to_surf_rect(state.log, font=font):
        screen.blit(surf, rect)

    # meld and flowers
    for surf, rect in melds_and_flowers_to_surface_rects(state, font=font):
        screen.blit(surf, rect)

    for player_idx, hand in enumerate(state.hands):
        hand_surf, hand_rect = hands_to_surface_rect(hand, player_idx, font=font)
        screen.blit(hand_surf, hand_rect)