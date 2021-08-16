import pyrayt
import unittest
import matplotlib.pyplot as plt
import numpy as np

class TestThickLenses(unittest.TestCase):

    def setUp(self) -> None:
        self.focus = 5
        self.aperture = 1
        self.thickness = 0.1
        self.baffle = pyrayt.components.baffle((2*self.aperture, 2*self.aperture)).move_x(self.focus)
        self.source = pyrayt.components.LineOfRays(0.5*self.aperture).move_x(-1)
    
    def test_planar_lens(self) -> None:
        # make a lens with a focal length approximate to the focus
        lens = pyrayt.components.thick_lens(np.inf, np.inf, self.thickness, aperture=1)
        tracer = pyrayt.RayTracer(self.source, [lens, self.baffle])
        results = tracer.trace()
        
        # verify that the rays remain projected along the +x axis
        self.assertTrue(np.allclose(results['x_tilt'], 1.0))
        self.assertTrue(np.allclose(results['y_tilt'], 0.0))
        self.assertTrue(np.allclose(results['z_tilt'], 0.0))

    def test_positive_miniscus_lens(self) -> None:
        # a miniscus lens needs to be tested a bit differently since you need to account for the thickness
        # we just want to check that the lens power has the right sign
        r_lens = 1 # lens radius
        self.thickness = 1 # thickness of the lens
        self.focus = ((0.5**2)/1.5*(self.thickness/r_lens**2))**-1 # from the lensmakers equation
        print(self.focus)

        # move the baffle to the new focus
        self.baffle = pyrayt.components.baffle((2*self.aperture, 2*self.aperture)).move_x(self.focus)
        # make a lens with a focal length approximate to the focus
        lens = pyrayt.components.thick_lens(r_lens, r_lens, self.thickness, aperture=1)
        tracer = pyrayt.RayTracer(self.source, [lens, self.baffle])
        results = tracer.trace()
        # tracer.show()
        # plt.show()

        baffle_rays = results.loc[results['surface']==self.baffle.get_id()]
        x_tilt = np.asarray(baffle_rays['x_tilt'])
        y_tilt = np.asarray(baffle_rays['y_tilt'])
        y_0 = np.asarray(baffle_rays['y0'])
        power_sign = -self.focus*y_tilt/x_tilt*y_0
        
        self.assertTrue(np.all(power_sign > 0), f"expected all_positive, got {power_sign}")
    
    def test_biconvex_lens(self) -> None:
        # make a lens with a focal length approximate to the focus
        lens_radius = self.focus
        lens = pyrayt.components.thick_lens(lens_radius,-lens_radius, self.thickness, aperture=1)
        tracer = pyrayt.RayTracer(self.source, [lens, self.baffle])
        results = tracer.trace()
        
        # verify that the rays are being focused onto the focal point 
        baffle_rays = results.loc[results['surface']==self.baffle.get_id()]
        x_tilt = np.asarray(baffle_rays['x_tilt'])
        y_tilt = np.asarray(baffle_rays['y_tilt'])
        y_0 = np.asarray(baffle_rays['y0'])
        
        expected_elevation = self.focus*y_tilt/x_tilt
        self.assertTrue(np.allclose(expected_elevation, -y_0, rtol=0.01), f"expected {expected_elevation[:5]}, got {y_0[:5]}")

    def test_plano_convex_lens(self) -> None:
        # make a lens with a focal length approximate to the focus
        lens_radius = self.focus/2
        lens = pyrayt.components.thick_lens(np.inf, -lens_radius, self.thickness, aperture=1)
        tracer = pyrayt.RayTracer(self.source, [lens, self.baffle])
        results = tracer.trace()
        
        # verify that the rays are being focused onto the focal point 
        baffle_rays = results.loc[results['surface']==self.baffle.get_id()]
        x_tilt = np.asarray(baffle_rays['x_tilt'])
        y_tilt = np.asarray(baffle_rays['y_tilt'])
        y_0 = np.asarray(baffle_rays['y0'])
        
        expected_elevation = self.focus*y_tilt/x_tilt
        self.assertTrue(np.allclose(expected_elevation, -y_0, rtol=0.01), f"expected {expected_elevation[:5]}, got {y_0[:5]}")
    
    def test_biconcave_lens(self) -> None:
        # make a lens with a focal length approximate to the focus
        lens_radius = self.focus
        lens = pyrayt.components.thick_lens(-lens_radius, lens_radius, self.thickness, aperture=1)
        tracer = pyrayt.RayTracer(self.source, [lens, self.baffle])
        results = tracer.trace()

        # verify that the rays are being focused onto the focal point 
        baffle_rays = results.loc[results['surface']==self.baffle.get_id()]
        x_tilt = np.asarray(baffle_rays['x_tilt'])
        y_tilt = np.asarray(baffle_rays['y_tilt'])
        y_0 = np.asarray(baffle_rays['y0'])
        
        expected_elevation = self.focus*y_tilt/x_tilt
        self.assertTrue(np.allclose(expected_elevation, y_0, rtol=0.01), f"expected {expected_elevation[:]}, got {y_0[:]}")

    def test_plano_concave_lens(self) -> None:
        # make a lens with a focal length approximate to the focus
        lens_radius = self.focus/2
        lens = pyrayt.components.thick_lens(np.inf, lens_radius, self.thickness, aperture=1)
        tracer = pyrayt.RayTracer(self.source, [lens, self.baffle])
        results = tracer.trace()
        
        # verify that the rays are being focused onto the focal point 
        baffle_rays = results.loc[results['surface']==self.baffle.get_id()]
        x_tilt = np.asarray(baffle_rays['x_tilt'])
        y_tilt = np.asarray(baffle_rays['y_tilt'])
        y_0 = np.asarray(baffle_rays['y0'])
        
        expected_elevation = self.focus*y_tilt/x_tilt
        self.assertTrue(np.allclose(expected_elevation, y_0, rtol=0.02), f"expected {expected_elevation[:5]}, got {y_0[:5]}")


if __name__ == "__main__":
    unittest.main()