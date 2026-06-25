"""
plotting.py

Plotting utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# PROFILE NAME
# ==========================================================

def build_profile_name(
    profile_type,
    **kwargs
):
    """
    Create descriptive profile name.
    """

    if profile_type == "vertical":

        return (
            f"vertical_"
            f"{kwargs['x']:.2f}um"
        ).replace(".", "p")

    elif profile_type == "horizontal":

        return (
            f"horizontal_"
            f"{kwargs['y']:.2f}um"
        ).replace(".", "p")

    elif profile_type == "line":

        start = kwargs["start"]
        end = kwargs["end"]

        return (
            f"line_"
            f"x{start[0]:.1f}_y{start[1]:.1f}"
            f"_to_"
            f"x{end[0]:.1f}_y{end[1]:.1f}"
        ).replace(".", "p")

    elif profile_type == "curve":

        return "curve_profile"

    return "profile"


# ==========================================================
# SAVE FIGURE
# ==========================================================

def save_figure(
    fig,
    filename,
    dpi=300
):
    """
    Save figure.
    """

    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(f"Saved: {filename}")


# ==========================================================
# RECONSTRUCTED MAP
# ==========================================================

def plot_reconstructed_map(
    field,
    x_range,
    y_range,
    output_file=None,
    title="Reconstructed Field",
    cmap="jet"
):
    """
    Plot reconstructed heatmap.
    """

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        field,
        extent=[
            x_range[0],
            x_range[1],
            y_range[0],
            y_range[1]
        ],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label("Value")

    plt.tight_layout()

    if output_file:
        save_figure(
            fig,
            output_file
        )

    return fig, ax
    
# ==========================================================
# PROFILE PLOT
# ==========================================================

def plot_profile(
    profile,
    output_file=None,
    title="Profile"
):
    """
    Plot extracted profile.
    """

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        profile["distance"],
        profile["value"],
        linewidth=2
    )

    ax.set_xlabel(
        "Distance"
    )

    ax.set_ylabel(
        "Value"
    )

    ax.set_title(
        title
    )

    ax.grid(True)

    plt.tight_layout()

    if output_file:
        save_figure(
            fig,
            output_file
        )

    return fig, ax
    
# ==========================================================
# COMPARISON PLOT
# ==========================================================

def plot_comparison(
    original_image,
    heatmap_bbox,
    field,
    profile,
    x_range,
    y_range,
    output_file=None,
    cmap="jet"
):
    """
    Original image vs reconstructed map.
    """

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # ----------------------------------
    # Original image
    # ----------------------------------

    ax[0].imshow(
        original_image
    )

    hx0, hy0, hx1, hy1 = heatmap_bbox

    x_path = profile["x"]
    y_path = profile["y"]

    px = (
        hx0
        +
        (x_path - x_range[0])
        /
        (x_range[1] - x_range[0])
        *
        (hx1 - hx0)
    )

    py = (
        hy1
        -
        (
            (y_path - y_range[0])
            /
            (y_range[1] - y_range[0])
        )
        *
        (hy1 - hy0)
    )

    ax[0].plot(
        px,
        py,
        "w--",
        linewidth=2
    )

    ax[0].scatter(
        [px[0], px[-1]],
        [py[0], py[-1]],
        s=50
    )

    ax[0].set_title(
        "Original Image"
    )

    ax[0].axis("off")

    # ----------------------------------
    # Reconstructed map
    # ----------------------------------

    im = ax[1].imshow(
        field,
        extent=[
            x_range[0],
            x_range[1],
            y_range[0],
            y_range[1]
        ],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap
    )

    ax[1].plot(
        profile["x"],
        profile["y"],
        "w--",
        linewidth=4
    )

    ax[1].scatter(
        [
            profile["x"][0],
            profile["x"][-1]
        ],
        [
            profile["y"][0],
            profile["y"][-1]
        ],
        s=50
    )

    ax[1].set_title(
        "Reconstructed Field"
    )

    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    fig.colorbar(
        im,
        ax=ax[1]
    )

    plt.tight_layout()

    if output_file:
        save_figure(
            fig,
            output_file
        )

    return fig, ax
    
# ==========================================================
# OUTPUT FILES
# ==========================================================

def create_output_filenames(
    output_dir,
    profile_name,
    resolution
):
    """
    Standard output names.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    nx, ny = resolution

    return {

        "map_csv":
        output_dir /
        f"field_{nx}x{ny}.csv",

        "map_png":
        output_dir /
        f"field_{nx}x{ny}.png",

        "profile_csv":
        output_dir /
        f"profile_{profile_name}.csv",

        "profile_png":
        output_dir /
        f"{profile_name}_profile.png",

        "comparison_png":
        output_dir /
        f"{profile_name}_comparison.png"
    }
