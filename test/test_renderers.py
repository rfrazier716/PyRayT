import unittest
import pyrayt.renderers as render
import pyrayt.designer as designer
from pyrayt.components.sources import LineOfRays
from pyrayt.surfaces import YZPlane
import pyrayt.shaders.analytic as mat


class TestAnalyticRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.system = designer.AnalyticSystem()
        self.sources = (LineOfRays().rotate_y(-45), LineOfRays().rotate_y(-135))
        self.surfaces = (YZPlane(material=mat.mirror).move_x(3), YZPlane(material=mat.mirror).move_x(-3))

        self.system.sources += self.sources
        self.system.components += self.surfaces
        self.renderer = render.AnalyticRenderer(self.system, generation_limit=20)

    def test_getters(self):
        self.renderer.set_generation_limit(1000)
        self.assertEqual(self.renderer.get_generation_limit(), 1000)

        new_system = designer.AnalyticSystem()
        new_system.sources.append("Hello World")
        self.renderer.load_system(new_system)
        self.assertEqual(self.renderer.get_system().sources[0], "Hello World")

    def test_full_render(self):
        # make a basic system  with a single generation limit
        print(self.system.sources)
        self.renderer.set_rays_per_source(1000)
        self.renderer.set_generation_limit(100)
        self.renderer.render()
        pass


if __name__ == '__main__':
    unittest.main()
