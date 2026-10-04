"""Interface invariants; native spike measurements are a separate real audit."""
import numpy as np
import pytest
from flysnake.lab.protocol import sample_linear
from flysnake.research.vision import ReadableGame, RetinaConfig, PALETTE, light_to_current


def test_board_palette_has_ordered_primary_receptor_drives():
    currents=[]
    for name in ('wall','background','body','food','head'):
        rgb=np.full((180,320,3),PALETTE[name],dtype=np.uint8)
        light=sample_linear(rgb,np.array([[.5,.5]])) @ np.array([.2126,.7152,.0722])
        currents.append(float(light_to_current(light,7)[0]))
    assert np.isfinite(currents).all()
    assert np.all(np.diff(currents)>0)
    assert currents[0]>7


@pytest.mark.parametrize('size',[3,4,6,8,12])
def test_entire_wall_and_board_fit_and_do_not_follow_the_head(size):
    game=ReadableGame(seed=13,size=size)
    g=game.observation_geometry();tile=g['tile_pixels'];ox,oy=g['board_origin']
    assert ox-tile>=0 and oy-tile>=0
    assert ox+(size+1)*tile<=320 and oy+(size+1)*tile<=180
    before=game.render_observation()
    game.snake=[(size-1,size-1),(size-2,size-1),(size-3,size-1)]
    game.direction=1
    after=game.render_observation()
    assert g==game.observation_geometry()
    np.testing.assert_array_equal(before[oy-tile:oy,ox-tile:ox+(size+1)*tile],
                                  after[oy-tile:oy,ox-tile:ox+(size+1)*tile])
    assert tuple(after[oy-tile//2,ox-tile//2])==PALETTE['wall']


def test_heading_is_visible_without_moving_any_other_board_pixels():
    game=ReadableGame(seed=13,size=4)
    game.snake=[(2,2),(2,3),(1,3)];game.food=(0,0)
    game.direction=0; first=game.render_observation()
    game.direction=1; second=game.render_observation()
    changed=np.any(first!=second,axis=2)
    g=game.observation_geometry();tile=g['tile_pixels'];ox,oy=g['board_origin']
    inside=np.zeros_like(changed);inside[oy+2*tile:oy+3*tile,ox+2*tile:ox+3*tile]=True
    assert changed.any() and not (changed&~inside).any()


def test_invalid_retinal_dynamics_are_rejected():
    for cfg in (RetinaConfig(refractory_ms=0),RetinaConfig(membrane_ms=-1),RetinaConfig(white_hz=float('nan'))):
        with pytest.raises(ValueError):light_to_current([0,.5,1],7,cfg)
