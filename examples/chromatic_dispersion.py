import pyrayt
import numpy as np
import tinygfx.g3d as cg


def main() -> None:

    # Create an equilateral prism that will refract the light
    # The default material for prisms is BK7 Crown Glass
    prism_size = 1
    prism = pyrayt.components.equilateral_prism(prism_size, prism_size)
    prism.move_x(prism_size / 4)

    # A baffle is required to view the rays, otherwise they'll be terminated when they exit the prism
    baffle = pyrayt.components.baffle((1, 1)).rotate_y(90).move(1, 0, -0.5)

    # Sources are typically monochromatic, so to create a Rainbow of Rays we'll need to create a list of sources that will be simulated
    sources = [
        pyrayt.components.LineOfRays(spacing=0.1, wavelength=x)
        .move_x(-prism_size / 2)
        .rotate_y(-3)
        for x in np.linspace(0.44, 0.75, 11)
    ]

    # load everything into a ray tracer
    tracer = pyrayt.RayTracer(sources, [prism, baffle])
    tracer.set_rays_per_source(
        1
    )  # We only need one ray per source for this since we already have 11 sources

    # Run the Ray trace and view results
    tracer.trace()
    tracer.show(
        color_function="wavelength",  # Use the wavelength color function to shade rays by wavelength, otherwise they'll all be one color
        resolution=1080,  # Horizontal resolution of the render
        ray_width=0.005,  # The width to draw the rays
        view="xz",  # Projected view for the display, The prism is oriented in the XZ plane so we want to view that projection.
    )


if __name__ == "__main__":
    main()
