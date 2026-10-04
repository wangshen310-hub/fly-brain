"""Label-free fixed display placement from anatomical sampling coverage.

The original receptor coordinates and neural graph remain unchanged. Only the
fixed viewport placement of the entire board is calibrated. No snake, food,
action or reward is available to this calibration function.
"""
import hashlib

import numpy as np
from .vision import ReadableGame
from .vision import PALETTE
from flysnake.game import DIRECTIONS

PROJECTION_VERSION = 'anatomical-coverage-fixed-board-v1'


def coverage_geometry(uv, size=4):
    uv = np.asarray(uv)
    if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) == 0
            or not np.isfinite(uv).all() or (uv<0).any() or (uv>1).any()
            or size not in (4, 6, 8, 12)):
        raise ValueError('Finite anatomical UV coordinates and supported board required')
    # Repeated photoreceptors at one position do not add spatial resolution.
    pixels = np.unique((uv*np.array([319, 179])).astype(int), axis=0)
    histogram = np.zeros((180, 320), dtype=int)
    histogram[pixels[:, 1], pixels[:, 0]] = 1
    integral = np.pad(histogram.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = None
    for tile in range(12, min(32, 180//(size+2))+1, 2):
        x, y = np.meshgrid(np.arange(tile, 320-(size+1)*tile+1, 2),
                           np.arange(tile, 180-(size+1)*tile+1, 2))
        if not x.size:
            continue
        x, y = x.ravel(), y.ravel()
        quarters = []
        for row in range(size):
            for col in range(size):
                for qy in range(2):
                    for qx in range(2):
                        x0 = x+col*tile+qx*(tile//2)
                        y0 = y+row*tile+qy*(tile//2)
                        x1, y1 = x0+tile//2, y0+tile//2
                        quarters.append(integral[y1, x1]-integral[y0, x1]
                                        -integral[y1, x0]+integral[y0, x0])
        counts = np.array(quarters)
        minimum = counts.min(axis=0)
        low_quantile = np.quantile(counts, .1, axis=0)
        total = counts.sum(axis=0)
        # Anatomical spatial coverage only, including all four parts of every
        # tile so heading marks are not hidden in an unobserved part of a cell.
        index = np.lexsort((-y, -x, np.full(len(x), tile), total,
                            low_quantile, minimum))[-1]
        score = (int(minimum[index]), float(low_quantile[index]),
                 int(total[index]), tile, -int(x[index]), -int(y[index]))
        if best is None or score > best[0]:
            quarter_counts = counts[:, index].reshape(size, size, 2, 2)
            best = (score, {'projection': 'fixed_world', 'version': PROJECTION_VERSION,
                            'tile_pixels': tile, 'board_cells': size,
                            'board_origin': [int(x[index]), int(y[index])],
                            'image_size': [320, 180], 'wall_cells': 1,
                            'unique_sample_pixels': len(pixels),
                            'quarter_sample_counts': quarter_counts.tolist(),
                            'cell_sample_counts': quarter_counts.sum((2, 3)).tolist(),
                            'calibration': 'Anatomical UV only; no objects, labels, actions or reward',
                            'uv_sha256': hashlib.sha256(np.ascontiguousarray(uv).tobytes()).hexdigest()})
    if best is None:
        raise ValueError('No complete board fits viewport')
    return best[1]


class CoveredGame(ReadableGame):
    def __init__(self, *, geometry, **kwargs):
        self.fixed_geometry = geometry
        super().__init__(**kwargs)
        if geometry['board_cells'] != self.size:
            raise ValueError('Calibrated geometry and board size differ')

    def observation_geometry(self):
        return self.fixed_geometry

    def render_observation(self):
        if self._feedback_observation is not None:
            return self._feedback_observation.copy()
        rgb = super().render_observation()
        tile = self.fixed_geometry['tile_pixels']
        ox, oy = self.fixed_geometry['board_origin']
        hx, hy = self.snake[0]
        region = rgb[oy+hy*tile:oy+(hy+1)*tile, ox+hx*tile:ox+(hx+1)*tile]
        region[:] = PALETTE['head']
        # A large front/back color split is observable at sparse anatomical
        # sample locations. This renders body orientation; it computes no move.
        rear = (235, 115, 30)
        mid = tile//2
        dx, dy = DIRECTIONS[self.direction]
        if dy < 0: region[mid:] = rear
        elif dy > 0: region[:mid] = rear
        elif dx > 0: region[:, :mid] = rear
        else: region[:, mid:] = rear
        return rgb
