import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

def generate_map(height=1024, width=1024, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=0):
    """
    Generating a 2D heightmap with Perlin noise
    """
    np.random.seed(seed)

    xs = np.arange(height) / scale
    ys = np.arange(width) / scale
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # efficient population with np.frompyfunc
    noise_fn = np.frompyfunc(
        lambda x, y: pnoise2(
            x, y,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            repeatx=height,
            repeaty=width,
            base=seed
        ),
        nin=2, nout=1)

    heightmap = noise_fn(X, Y).astype(np.float32)

    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    return heightmap


def add_slope(heightmap, direction="x", strength=0.2):
    # adds overall slope to the map to break symmetry from the Perlin noise
    h, w = heightmap.shape
    if direction == "x":
        slope = np.linspace(0, 1, w)
        heightmap += strength * slope[None, :]
    else:
        slope = np.linspace(0, 1, h)
        heightmap += strength * slope[:, None]
    return heightmap


def generate_heightmap(height=1024, width=1024, p_slope=0.7, seed=0):
    """
    Not setting any map generation params apart from H/W to be configurable right now.
    Could be worth exploring in the future.
    """
    base = generate_map(height, width, scale=800, octaves=2, seed=seed)
    mid  = generate_map(height, width, scale=200, octaves=4, seed=seed)

    heightmap = (
        0.6 * base +
        0.4 * mid
    )

    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    # widens gap between large and small elevations
    heightmap = heightmap ** 1.5

    if np.random.rand() < p_slope:
        heightmap = add_slope(
            heightmap,
            direction=np.random.choice(["x", "y"]),
            strength=np.random.uniform(0.05, 0.2)
        )

    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    return heightmap