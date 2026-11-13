"""Render main-paper Figure 1 from the first Table 1 replication."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ("spherical", "elliptical")
METHODS = ("cindes", "rfcde", "maf")
TITLES = ("True density", "CINDES", "RFCDE", "MAF")
ROW_LABELS = ("(a) Spherical Gaussian mixture", "(b) Elliptical Gaussian mixture")


def load_row(
    input_dir: Path, scenario: str, seed: int, mode: str
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    densities = []
    reference_grid = None
    reference_truth = None
    lower = upper = None
    for method in METHODS:
        path = input_dir / f"figure1_{scenario}_{method}_seed{seed}_{mode}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; rerun Table 1 with --figure-data-dir")
        with np.load(path) as data:
            grid = data["grid"]
            truth = data["true_density"]
            estimate = data["estimated_density"]
            size = int(data["grid_size"])
            if (grid.shape != (size * size, 2)
                    or truth.shape != (size * size,)
                    or estimate.shape != (size * size,)):
                raise ValueError(f"invalid density grid in {path}")
            if reference_grid is None:
                reference_grid, reference_truth = grid.copy(), truth.copy()
                lower, upper = data["lower"].copy(), data["upper"].copy()
            elif (not np.array_equal(grid, reference_grid)
                  or not np.array_equal(truth, reference_truth)):
                raise ValueError(f"{path} does not share its data/grid with the other methods")
            densities.append(estimate.reshape(size, size))
    if not all(np.isfinite(image).all() for image in densities):
        raise ValueError(f"nonfinite density in {scenario} row")
    return [reference_truth.reshape(size, size), *densities], lower, upper


def render(input_dir: Path, output: Path, seed: int, quick: bool, dpi: int) -> None:
    mode = "quick" if quick else "full"
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    for row_index, scenario in enumerate(SCENARIOS):
        images, lower, upper = load_row(input_dir, scenario, seed, mode)
        vmax = max(float(image.max()) for image in images)
        for column_index, (image, title) in enumerate(zip(images, TITLES)):
            ax = axes[row_index, column_index]
            artist = ax.imshow(
                image, extent=(lower[0], upper[0], lower[1], upper[1]),
                origin="lower", aspect="equal", cmap="Blues", vmin=0, vmax=vmax,
            )
            ax.set_title(title)
            ax.set_xlabel(r"$y_1$")
            if column_index == 0:
                ax.set_ylabel(ROW_LABELS[row_index] + "\n" + r"$y_2$")
        fig.colorbar(artist, ax=axes[row_index, :], shrink=0.75, label="Density")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Wrote main-paper Figure 1 to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/main/figure1"))
    parser.add_argument("--output", type=Path, default=Path("results/main/figure1.pdf"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick", action="store_true", help="Use quick-run grids; not a paper figure."
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    render(args.input_dir, args.output, args.seed, args.quick, args.dpi)


if __name__ == "__main__":
    main()
