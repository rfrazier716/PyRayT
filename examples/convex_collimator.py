import pyrayt


def lensmakers_equation(r1: float, r2: float, n_lens: float, thickness: float) -> float:
    """
    Calculate the focal length of a thick spherical lens using the lensmaker's equation.

    :param r1: the first radius of curvature, positive for convex, negative for concave
    :param r2: the second radius of curvature, negative for convex, positive for concave
    :param n_lens: the refractive index of the lens
    :param thickness: The thickness of the lens

    :return: The focal length of the lens based on the paraxial approximation
    """

    p = (n_lens - 1) * (1 / r1 - 1 / r2 + (n_lens - 1) * thickness / (n_lens * r1 * r2))
    return 1 / p


def main() -> None:

    # Create the Collimating Lens
    r1 = 2  # The radius of curvature for the first lens surface
    r2 = 2  # Radius of Curvature for the second lens surface
    thickness = 0.25  # the lens' maximum thickness
    aperture = 1  # the aperture of the lens, (a circular aperture with diameter == 1)

    lens = pyrayt.components.biconvex_lens(r1, r2, thickness, aperture=aperture)
    focus = lensmakers_equation(
        r1, -r2, 1.5, thickness
    )  # calculate the focus of the lens

    # Create a source and move it to the -focus
    source = pyrayt.components.ConeOfRays(cone_angle=6).move_x(-focus)

    # Create a baffle so we can view the collimated rays
    baffle = pyrayt.components.baffle((1, 1)).move_x(1)

    # load everything into a ray-trace object
    tracer = pyrayt.RayTracer(source, [lens, baffle])
    tracer.set_rays_per_source(50)
    tracer.set_generation_limit(100)

    # Run the Ray trace and view results
    results = tracer.trace()
    tracer.show()
    # uncomment this line to view the plot


if __name__ == "__main__":
    main()
