"""Replicator dynamics simulation.

This script simulates a quasi-replicator model of market selection with
innovation, entry, and exit. It generates summary plots for market shares,
productivity deviations, turbulence, concentration, growth rates, and the
size-rank relationship.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for the simulation."""

    mu: float = 0.05  # Mean innovation draw
    sigma: float = 0.8  # Standard deviation of innovation draw
    selection_intensity: float = 1.0  # Intensity of market selection
    periods: int = 200  # Number of time periods
    firms: int = 150  # Number of firms
    seed: int | None = 42  # Random seed for reproducibility

    @property
    def minimum_market_share(self) -> float:
        """Minimum market share below which a firm exits and is replaced."""
        return 1 / (self.firms * 10)


def run_simulation(config: SimulationConfig) -> dict[str, np.ndarray]:
    """Run the replicator dynamics simulation.

    Returns a dictionary containing:
    - productivity: productivity levels by firm and period
    - shares: market shares by firm and period
    - productivity_deviation: log productivity deviations from weighted average
    - innovation: innovation draws by firm and period
    """
    rng = np.random.default_rng(config.seed)

    productivity = np.zeros((config.firms, config.periods))
    shares = np.zeros((config.firms, config.periods))
    productivity_deviation = np.zeros((config.firms, config.periods))
    innovation = np.zeros((config.firms, config.periods))

    productivity[:, 0] = 1.0
    shares[:, 0] = 1 / config.firms
    productivity_deviation[:, 0] = np.log(productivity[:, 0]) - np.log(
        np.sum(productivity[:, 0] * shares[:, 0])
    )
    innovation[:, 0] = config.mu

    for t in range(1, config.periods):
        # If a firm draws a negative innovation, it keeps its previous technique.
        innovation[:, t] = np.maximum(
            rng.normal(config.mu, config.sigma, config.firms), 0
        )

        productivity[:, t] = productivity[:, t - 1] * (1 + innovation[:, t])
        average_productivity = np.sum(productivity[:, t] * shares[:, t - 1])

        shares[:, t] = shares[:, t - 1] * (
            1
            + config.selection_intensity
            * ((productivity[:, t] / average_productivity) - 1)
        )

        productivity_deviation[:, t] = np.log(productivity[:, t]) - np.log(
            np.sum(productivity[:, t] * shares[:, t - 1])
        )

        # Entry-exit process: firms below the threshold are replaced by entrants.
        exit_index = np.where(shares[:, t] < config.minimum_market_share)[0]
        shares[exit_index, t] = config.minimum_market_share
        innovation[exit_index, t] = np.maximum(
            rng.normal(config.mu, config.sigma, len(exit_index)), 0
        )
        productivity[exit_index, t] = (1 + innovation[exit_index, t]) * average_productivity

        # Normalize shares after entry-exit to keep total market share equal to 1.
        shares[:, t] = shares[:, t] / shares[:, t].sum()

    return {
        "productivity": productivity,
        "shares": shares,
        "productivity_deviation": productivity_deviation,
        "innovation": innovation,
    }


def calculate_indicators(shares: np.ndarray) -> dict[str, np.ndarray | float]:
    """Calculate turbulence, concentration, and firm growth indicators."""
    turbulence = np.abs(np.diff(shares, axis=1))
    turbulence_index = np.sum(turbulence, axis=0)
    herfindahl_index = np.sum(shares**2, axis=0)

    safe_shares = np.clip(shares, 1e-12, None)
    log_share_growth = np.diff(np.log(safe_shares), axis=1)
    growth_volatility = float(np.std(log_share_growth))

    return {
        "turbulence_index": turbulence_index,
        "herfindahl_index": herfindahl_index,
        "log_share_growth": log_share_growth,
        "growth_volatility": growth_volatility,
    }


def save_plot(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save and close a matplotlib figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_market_shares(shares: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for firm in range(shares.shape[0]):
        ax.plot(shares[firm, :], linewidth=0.8, alpha=0.7)
    ax.set_title("Market Shares Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Market Share")
    save_plot(fig, output_dir, "market_shares_evolution.png")


def plot_productivity_deviation(productivity_deviation: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for firm in range(productivity_deviation.shape[0]):
        ax.plot(productivity_deviation[firm, :], linewidth=0.8, alpha=0.7)
    ax.set_title("Deviation from Average Productivity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Log Deviation")
    save_plot(fig, output_dir, "productivity_deviation_evolution.png")


def plot_productivity_distribution(productivity_deviation: np.ndarray, output_dir: Path) -> None:
    pooled = productivity_deviation.reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pooled, bins=20)
    ax.set_title("Productivity Distribution: Linear Scale")
    ax.set_xlabel("Deviation from Average Productivity")
    ax.set_ylabel("Frequency")
    save_plot(fig, output_dir, "productivity_distribution_linear.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pooled, bins=20)
    ax.set_yscale("log")
    ax.set_title("Productivity Distribution: Log Scale")
    ax.set_xlabel("Deviation from Average Productivity")
    ax.set_ylabel("Frequency")
    save_plot(fig, output_dir, "productivity_distribution_log.png")


def plot_turbulence(turbulence_index: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(turbulence_index)
    ax.set_title("Market Turbulence Index")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Turbulence Index")
    ax.grid(True)
    save_plot(fig, output_dir, "market_turbulence_index.png")


def plot_concentration(herfindahl_index: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(herfindahl_index)
    ax.set_title("Market Concentration: Herfindahl Index")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Herfindahl Index")
    ax.grid(True)
    save_plot(fig, output_dir, "herfindahl_index.png")


def plot_growth_distribution(log_share_growth: np.ndarray, output_dir: Path) -> None:
    pooled_growth = log_share_growth.reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pooled_growth, bins=20)
    ax.set_yscale("log")
    ax.set_title("Distribution of Firm Size Growth Rates")
    ax.set_xlabel("Firm Size Growth Rate")
    ax.set_ylabel("Frequency, log scale")
    save_plot(fig, output_dir, "firm_size_growth_distribution.png")


def plot_size_rank(shares: np.ndarray, output_dir: Path) -> None:
    pooled_shares = pd.DataFrame({"share": shares.reshape(-1)})
    ranked = pooled_shares.sort_values(by="share", ascending=False).reset_index(drop=True)

    log_size = np.log(np.clip(ranked["share"].to_numpy(), 1e-12, None))
    log_rank = np.log(np.arange(1, len(ranked) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_size, log_rank, ".", markersize=2)
    ax.set_title("Size-Rank Relationship, Logs")
    ax.set_xlabel("Log Size")
    ax.set_ylabel("Log Rank")
    ax.grid(True)
    save_plot(fig, output_dir, "size_rank_relationship.png")


def export_data(results: dict[str, np.ndarray], output_dir: Path) -> None:
    """Export core simulation matrices as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results["shares"]).to_csv(output_dir / "market_shares.csv", index=False)
    pd.DataFrame(results["productivity"]).to_csv(output_dir / "productivity.csv", index=False)
    pd.DataFrame(results["productivity_deviation"]).to_csv(
        output_dir / "productivity_deviation.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a replicator dynamics simulation.")
    parser.add_argument("--mu", type=float, default=0.05, help="Mean innovation draw.")
    parser.add_argument("--sigma", type=float, default=0.8, help="Innovation standard deviation.")
    parser.add_argument("--selection-intensity", type=float, default=1.0)
    parser.add_argument("--periods", type=int, default=200, help="Number of time periods.")
    parser.add_argument("--firms", type=int, default=150, help="Number of firms.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where plots and CSV files are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        mu=args.mu,
        sigma=args.sigma,
        selection_intensity=args.selection_intensity,
        periods=args.periods,
        firms=args.firms,
        seed=args.seed,
    )

    results = run_simulation(config)
    indicators = calculate_indicators(results["shares"])

    export_data(results, args.output_dir)
    plot_market_shares(results["shares"], args.output_dir)
    plot_productivity_deviation(results["productivity_deviation"], args.output_dir)
    plot_productivity_distribution(results["productivity_deviation"], args.output_dir)
    plot_turbulence(indicators["turbulence_index"], args.output_dir)
    plot_concentration(indicators["herfindahl_index"], args.output_dir)
    plot_growth_distribution(indicators["log_share_growth"], args.output_dir)
    plot_size_rank(results["shares"], args.output_dir)

    print("Simulation completed.")
    print(f"Growth volatility: {indicators['growth_volatility']:.6f}")
    print(f"Outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
