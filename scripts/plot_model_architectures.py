"""
Plot model architecture diagrams without requiring Graphviz/dot.

This script does not use pyhgf.plots.networkx.plot_network because that
function depends on Graphviz's external 'dot' executable.

Instead, it manually defines the node positions and edges for the three
architectures.

Outputs:
    outputs/figures/model_architectures/
        local_only_architecture.png
        global_only_architecture.png
        shared_volatility_architecture.png

Run from project root:
    python scripts/plot_model_architectures.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "config").is_dir())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "model_architectures"


def draw_architecture(
    *,
    model_name: str,
    node_labels: dict[int, str],
    edges: list[tuple[int, int, str]],
    positions: dict[int, tuple[float, float]],
    output_path: Path,
) -> None:
    """
    Draw one architecture as a directed node-edge graph.

    Parameters
    ----------
    model_name:
        Name shown in the figure title.

    node_labels:
        Mapping from integer node ID to display label.

    edges:
        List of edges as:
            (parent_node, child_node, edge_type)

        edge_type should be:
            "value"
            "volatility"

    positions:
        Manual 2D node positions.

    output_path:
        Where to save the PNG figure.
    """
    graph = nx.DiGraph()

    for node_id in node_labels:
        graph.add_node(node_id)

    for parent, child, edge_type in edges:
        graph.add_edge(parent, child, edge_type=edge_type)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{model_name} architecture", fontsize=15, fontweight="bold")

    value_edges = [
        (parent, child)
        for parent, child, edge_type in edges
        if edge_type == "value"
    ]

    volatility_edges = [
        (parent, child)
        for parent, child, edge_type in edges
        if edge_type == "volatility"
    ]

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_size=2800,
        linewidths=1.5,
        edgecolors="black",
        node_color="white",
    )

    nx.draw_networkx_labels(
        graph,
        pos=positions,
        labels=node_labels,
        ax=ax,
        font_size=9,
    )

    # Solid arrows for value coupling.
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=value_edges,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=2.0,
        style="solid",
        connectionstyle="arc3,rad=0.0",
    )

    # Dashed arrows for volatility coupling.
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        edgelist=volatility_edges,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=2.0,
        style="dashed",
        connectionstyle="arc3,rad=0.0",
    )

    # Edge labels
    edge_labels = {
        (parent, child): edge_type
        for parent, child, edge_type in edges
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos=positions,
        edge_labels=edge_labels,
        ax=ax,
        font_size=8,
        label_pos=0.5,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )

    ax.text(
        0.02,
        0.02,
        "solid = value coupling   |   dashed = volatility coupling",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_local_only() -> None:
    """
    Plot local-only architecture.
    """
    node_labels = {
        0: "0\ninput_A",
        1: "1\ninput_B",
        2: "2\nvalue_A",
        3: "3\nvalue_B",
        4: "4\nvol_A",
        5: "5\nvol_B",
    }

    edges = [
        (2, 0, "value"),
        (3, 1, "value"),
        (4, 2, "volatility"),
        (5, 3, "volatility"),
    ]

    positions = {
        4: (0, 2),
        2: (0, 1),
        0: (0, 0),
        5: (3, 2),
        3: (3, 1),
        1: (3, 0),
    }

    draw_architecture(
        model_name="local_only",
        node_labels=node_labels,
        edges=edges,
        positions=positions,
        output_path=OUTPUT_DIR / "local_only_architecture.png",
    )


def plot_global_only() -> None:
    """
    Plot global-only architecture.
    """
    node_labels = {
        0: "0\ninput_A",
        1: "1\ninput_B",
        2: "2\nvalue_A",
        3: "3\nvalue_B",
        4: "4\nshared_vol",
    }

    edges = [
        (2, 0, "value"),
        (3, 1, "value"),
        (4, 2, "volatility"),
        (4, 3, "volatility"),
    ]

    positions = {
        4: (1.5, 2),
        2: (0, 1),
        3: (3, 1),
        0: (0, 0),
        1: (3, 0),
    }

    draw_architecture(
        model_name="global_only",
        node_labels=node_labels,
        edges=edges,
        positions=positions,
        output_path=OUTPUT_DIR / "global_only_architecture.png",
    )


def plot_shared_volatility() -> None:
    """
    Plot shared-volatility / hybrid architecture.
    """
    node_labels = {
        0: "0\ninput_A",
        1: "1\ninput_B",
        2: "2\nvalue_A",
        3: "3\nvalue_B",
        4: "4\nlocal_vol_A",
        5: "5\nlocal_vol_B",
        6: "6\nshared_parent",
    }

    edges = [
        (2, 0, "value"),
        (3, 1, "value"),
        (4, 2, "volatility"),
        (5, 3, "volatility"),
        (6, 4, "value"),
        (6, 5, "value"),
    ]

    positions = {
        6: (1.5, 3),
        4: (0, 2),
        5: (3, 2),
        2: (0, 1),
        3: (3, 1),
        0: (0, 0),
        1: (3, 0),
    }

    draw_architecture(
        model_name="shared_volatility",
        node_labels=node_labels,
        edges=edges,
        positions=positions,
        output_path=OUTPUT_DIR / "shared_volatility_architecture.png",
    )


def main() -> None:
    """
    Save all architecture diagrams.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_local_only()
    plot_global_only()
    plot_shared_volatility()

    print("\nFinished plotting model architectures.")


if __name__ == "__main__":
    main()