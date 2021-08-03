import pyrayt
from pyrayt.utils import lensmakers_equation


def main() -> None:

    crown_glass_coeffs = [1.03961212 , 0.231792344,  	1.01046945, 6.00069867E-3, 2.00179144E-2, 1.03560653E02]
    crown_glass = pyrayt.materials.SellmeierRefractor(*crown_glass_coeffs)

    # Create the Collimating Lens
    r1 = 100 # The radius of curvature for the first lens surface
    r2 = 100  # Radius of Curvature for the second lens surface
    thickness = 10  # the lens' maximum thickness
    aperture = 50  # the aperture of the lens, (a circular aperture with diameter == 1)
    focus = lensmakers_equation(r1, -r2, crown_glass._index(0.63), thickness)
    print(crown_glass._index(0.63))
    print(focus)
    lens = pyrayt.components.biconvex_lens(r1, r2, thickness, aperture=aperture, material = crown_glass)

    # Create a source and move it to the -focus
    sources = [pyrayt.components.ConeOfRays(cone_angle=5, wavelength=x).move_x(-focus) for x in [0.405, 0.530, 0.650]]

    # Create a baffle so we can view the collimated rays
    baffle = pyrayt.components.baffle((aperture, aperture)).move_x(2*focus)

    # load everything into a ray-trace object
    tracer = pyrayt.RayTracer(sources, [lens, baffle])
    tracer.set_rays_per_source(20)
    tracer.set_generation_limit(100)

    # Run the Ray trace and view results
    results = tracer.trace()
    tracer.show(color_function='wavelength', resolution=1080, ray_width=0.05)
    # uncomment this line to view the plot

if __name__ == "__main__":
    main()
