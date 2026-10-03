"""
Create equation figures for the HGF environment and response model.

This script generates high-resolution PNG images that can be inserted into
PowerPoint, Word, or a meeting document.

Run from project root:

    python scripts/make_equation_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hgf_walk_outputs"
    / "figures"
    / "equations"
)


def draw_equation_figure(
    *,
    title: str,
    subtitle: str,
    equation_lines: list[str],
    output_path: Path,
    figsize: tuple[float, float] = (14, 8),
    line_spacing: float = 0.085,
    equation_fontsize: int = 18,
) -> None:
    """
    Draw a clean equation figure using matplotlib mathtext.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        title,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    ax.text(
        0.02,
        0.90,
        subtitle,
        fontsize=13,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    y = 0.80

    for line in equation_lines:
        ax.text(
            0.04,
            y,
            line,
            fontsize=equation_fontsize,
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
        y -= line_spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Create all equation figures.
    """

    environment_equations = [
        r"$s \in \{A,B\}, \quad t = 1,\ldots,T$",
        r"$z_t = 1 \;\mathrm{during\ A\ volatile\ blocks}, \quad z_t = 0 \;\mathrm{otherwise}$",
        r"$Q_{A,t} = Q_{\mathrm{low}} + z_t\left(Q_{\mathrm{high}} - Q_{\mathrm{low}}\right)$",
        r"$Q_{B,t} = Q_{\mathrm{low}}$",
        r"$x_{s,t} = \lambda_x x_{s,t-1} + \rho_x + \sqrt{Q_{s,t}}\,\epsilon_{s,t}$",
        r"$\epsilon_{s,t} \sim \mathcal{N}(0,1)$",
        r"$u_{s,t} = x_{s,t} + \sigma_u \eta_{s,t}$",
        r"$\eta_{s,t} \sim \mathcal{N}(0,1)$",
        r"$\lambda_x=0.95,\quad \rho_x=0,\quad Q_{\mathrm{low}}=0.0004,\quad Q_{\mathrm{high}}=0.01,\quad \sigma_u=0.10$",
    ]

    response_equations = [
        r"$\hat{\mu}_{A,t} = x_{2,t}^{\mathrm{expected\ mean}}, \quad \hat{\mu}_{B,t} = x_{3,t}^{\mathrm{expected\ mean}}$",
        r"$y_{A,t} \sim \mathcal{N}\left(\hat{\mu}_{A,t}, \sigma_{\mathrm{resp}}^2\right)$",
        r"$y_{B,t} \sim \mathcal{N}\left(\hat{\mu}_{B,t}, \sigma_{\mathrm{resp}}^2\right)$",
        r"$y_{s,t} = \hat{\mu}_{s,t} + \xi_{s,t}$",
        r"$\xi_{s,t} \sim \mathcal{N}\left(0,\sigma_{\mathrm{resp}}^2\right)$",
        r"$\sigma_{\mathrm{resp}} = 0.10$",
        r"$p(y \mid u,\theta) = \prod_t \prod_{s\in\{A,B\}} \mathcal{N}\left(y_{s,t};\hat{\mu}_{s,t}(\theta),\sigma_{\mathrm{resp}}^2\right)$",
    ]

    likelihood_equations = [
        r"$-\log p(y \mid u,\theta) =$",
        r"$\sum_t \sum_{s\in\{A,B\}} \left[" 
        r"\frac{1}{2}\log\left(2\pi\sigma_{\mathrm{resp}}^2\right)"
        r"+"
        r"\frac{\left(y_{s,t}-\hat{\mu}_{s,t}(\theta)\right)^2}{2\sigma_{\mathrm{resp}}^2}"
        r"\right]$",
        r"$\mathrm{Lower\ negative\ log\ likelihood} \;=\; \mathrm{better\ response\ model\ fit}$",
    ]

    learning_rate_equations = [
        r"$\eta_{A,t} = \frac{\pi_{u,A,t}}{\pi_{x,A,t}}$",
        r"$\eta_{B,t} = \frac{\pi_{u,B,t}}{\pi_{x,B,t}}$",
        r"$\pi_{u,s,t} = \mathrm{observation\ precision}$",
        r"$\pi_{x,s,t} = \mathrm{posterior\ precision\ of\ the\ value\ node}$",
        r"$\eta_{A,t} = \frac{x_{0,t}^{\mathrm{expected\ precision}}}{x_{2,t}^{\mathrm{expected\ precision}}}$",
        r"$\eta_{B,t} = \frac{x_{1,t}^{\mathrm{expected\ precision}}}{x_{3,t}^{\mathrm{expected\ precision}}}$",
    ]

    draw_equation_figure(
        title="Simulation environment",
        subtitle="Two-stream Gaussian random-walk environment used to generate observations.",
        equation_lines=environment_equations,
        output_path=OUTPUT_DIR / "environment_equations.png",
        figsize=(15, 8),
        line_spacing=0.075,
        equation_fontsize=18,
    )

    draw_equation_figure(
        title="Continuous Gaussian response model",
        subtitle="Response layer mapping HGF predictions to simulated participant predictions.",
        equation_lines=response_equations,
        output_path=OUTPUT_DIR / "response_model_equations.png",
        figsize=(15, 7),
        line_spacing=0.085,
        equation_fontsize=18,
    )

    draw_equation_figure(
        title="Response likelihood for later parameter fitting",
        subtitle="The response model can also score observed predictions by negative log-likelihood.",
        equation_lines=likelihood_equations,
        output_path=OUTPUT_DIR / "response_likelihood_equations.png",
        figsize=(15, 5),
        line_spacing=0.13,
        equation_fontsize=18,
    )

    draw_equation_figure(
        title="Model-implied learning rate",
        subtitle="Precision-ratio learning-rate measure extracted from pyhgf trajectories.",
        equation_lines=learning_rate_equations,
        output_path=OUTPUT_DIR / "learning_rate_equations.png",
        figsize=(13, 6),
        line_spacing=0.105,
        equation_fontsize=20,
    )

    print("Saved equation figures:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()