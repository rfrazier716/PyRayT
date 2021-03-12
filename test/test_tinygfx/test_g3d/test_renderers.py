import unittest

import tinygfx.g3d as cg


class TestEdgeRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.surfaces = (cg.Sphere(1).move_x(3).move_y(0.5), cg.Sphere(1).move_x(3).move_y(-0.5))
        self.camera = cg.OrthographicCamera(10, 10, 1)
        self.renderer = cg.renderers.EdgeRender(self.camera, self.surfaces)

    def test_render_results(self):
        results = self.renderer.render()
        self.assertEqual(results.shape, self.camera.get_resolution()[::-1])


if __name__ == '__main__':
    unittest.main()
