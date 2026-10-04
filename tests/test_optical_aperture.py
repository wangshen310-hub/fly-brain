"""Physical radiance interpolation checks; no neural learning claims."""
import numpy as np
from flysnake.research.optical_sampling import sample_aperture_linear,ApertureConfig,footprint


def test_uniform_radiance_and_sensor_positions_are_preserved():
    uv=np.array([[0.,0.],[.3,.5],[1.,1.]])
    rgb=np.full((180,320,3),128,dtype=np.uint8)
    got=sample_aperture_linear(rgb,uv)
    expected=((128/255+.055)/1.055)**2.4
    assert np.allclose(got,expected)
    assert np.array_equal(uv,np.array([[0.,0.],[.3,.5],[1.,1.]]))
    assert np.allclose(sum(w for _,w in footprint(uv)),1.)


def test_bilinear_photons_are_interpolated_after_linearizing_rgb():
    rgb=np.zeros((180,320,3),dtype=np.uint8)
    rgb[20,11]=255
    uv=np.array([[10.5/319,20/179]])
    value=sample_aperture_linear(rgb,uv,ApertureConfig(radius_pixels=0,subdivisions=1))
    assert np.allclose(value,.5)
    # .5 linear radiance is NOT gamma-coded midpoint luminance (~.214).
    assert not np.allclose(value,((.5+.055)/1.055)**2.4)
