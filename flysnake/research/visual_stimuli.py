"""Fixed-world visual stimuli, kept outside the neural input adapter.

World coordinates are used only to render RGB. These audit stimuli do not
deliver an answer, food vector, or direction label to the neural model.
"""
import numpy as np
from .projection import CoveredGame
from .optical_sampling import coverage_uv
from .viewport_candidates import find_viewport
from .vision import PALETTE

VERSION = 'fixed-world-RGB-only-blue-body-stimuli-v1'
BODY_COLOR = (90, 190, 245)


class VisualBoardGame(CoveredGame):
    def render_observation(self):
        if self._feedback_observation is not None:
            return self._feedback_observation.copy()
        rgb = super().render_observation()
        tile = self.fixed_geometry['tile_pixels']
        ox, oy = self.fixed_geometry['board_origin']
        for x, y in self.snake[1:]:
            rgb[oy+y*tile:oy+(y+1)*tile, ox+x*tile:ox+(x+1)*tile] = BODY_COLOR
        return rgb


def fixed_geometry(b, aperture, size=12):
    support = coverage_uv(np.r_[b.uv, b.r8_uv], aperture)
    return find_viewport(support, size, pixel_stride=1, tile_stride=1)


def audit_frames(geometry):
    if geometry['board_cells'] != 12:
        raise ValueError('This declared visual assay uses a fixed 12x12 board')
    game = VisualBoardGame(seed=613, size=12, geometry=geometry)
    game.snake = [(6, 6), (6, 7), (6, 8)]
    game.direction = 0
    game.food = None
    frames = {'board_no_food': game.render_observation()}
    game.food = (5, 6)
    frames['food_A'] = game.render_observation()
    game.food = (7, 6)
    frames['food_B'] = game.render_observation()
    game.food = (5, 6)
    game.direction = 1
    frames['head_rotated'] = game.render_observation()
    frames['head_as_body'] = frames['food_A'].copy()
    tile = geometry['tile_pixels']; ox, oy = geometry['board_origin']
    frames['head_as_body'][oy+6*tile:oy+7*tile, ox+6*tile:ox+7*tile] = BODY_COLOR
    game.snake = [(7, 6), (6, 6), (6, 7)]
    frames['pose_moved_food_fixed'] = game.render_observation()
    game.snake = []
    # A complete empty board is a pixel-level control, not a playable Snake.
    empty = frames['board_no_food'].copy()
    for x, y in ((6, 6), (6, 7), (6, 8)):
        empty[oy+y*tile:oy+(y+1)*tile, ox+x*tile:ox+(x+1)*tile] = PALETTE['background']
    frames['board_empty'] = empty
    frames['neutral_gray'] = np.full((180, 320, 3), 85, np.uint8)
    return frames
