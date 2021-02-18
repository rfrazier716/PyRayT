import unittest
import adpd.packaging.renderers as render
from adpd.packaging.components.sources import LineOfRays
from adpd.packaging.analytic_surface import YZPlane
import adpd.packaging.shaders.analytic as mat


class TestAnalyticRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = (LineOfRays().rotate_y(-45), LineOfRays().rotate_y(-135))
        self.system = (YZPlane(material=mat.mirror).move_x(3), YZPlane(material = mat.mirror).move_x(-3))

    def test_creation(self):
        # make a basic system  with a single generation limit
        this_renderer = render.AnalyticRenderer(self.sources, self.system, 1, generation_limit=10)
        this_renderer.render()
        pass

    def test_full_render(self):
        # make a basic system  with a single generation limit
        this_renderer = render.AnalyticRenderer(self.sources, self.system, 1000, generation_limit=10)
        this_renderer.render()
        pass


if __name__ == '__main__':
    unittest.main()
