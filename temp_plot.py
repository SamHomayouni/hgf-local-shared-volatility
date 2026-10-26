from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path("outputs/hgf_walk_outputs/hybrid/environment_wide.csv")
df = pd.read_csv(path).sort_values("trial_index")

required = {
    "trial_index", "shared_state", "shared_mean",
    "local_volatility_A", "local_volatility_B",
    "deterministic_local_volatility_A",
    "deterministic_local_volatility_B",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing Hybrid columns: {sorted(missing)}")

t = df["trial_index"]

# 1. Shared signal: what drives the common transitions.
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t, df["shared_state"], label="Realized shared signal")
ax.plot(t, df["shared_mean"], "--", label="Prescribed pulse mean")
ax.set(
    title="Hybrid environment: shared signal",
    xlabel="Trial index", ylabel="Shared state",
)
ax.legend()
fig.tight_layout()

# 2. Actual local volatility versus the reference without random terms.
fig, ax = plt.subplots(figsize=(12, 4))
for stream in ("A", "B"):
    ax.plot(
        t, df[f"local_volatility_{stream}"],
        label=f"Local volatility {stream}",
    )
for stream in ("A", "B"):
    ax.plot(
        t, df[f"deterministic_local_volatility_{stream}"], "--",
        label=f"{stream}: shared pulses only, no random terms",
    )
ax.set(
    title="Hybrid environment: local volatility states",
    xlabel="Trial index", ylabel="Local volatility state",
)
ax.legend()
fig.tight_layout()

# 3. Accumulated randomness left after subtracting the pulse reference.
fig, ax = plt.subplots(figsize=(12, 3))
for stream in ("A", "B"):
    deviation = (
        df[f"local_volatility_{stream}"]
        - df[f"deterministic_local_volatility_{stream}"]
    )
    ax.plot(t, deviation, label=f"Stream {stream}")

ax.axhline(0, linestyle=":", linewidth=1)
ax.set(
    title="Hybrid environment: accumulated random component",
    xlabel="Trial index", ylabel="Actual minus no-noise reference",
)
ax.legend()
fig.tight_layout()

plt.show()