"""
sensitivity_analysis.py
The decision engine's expected-profit formula rests on three assumed
constants: interest rate, loss-given-default (LGD), and (for Model 1's
threshold) a 5:1 false-negative cost ratio. None of these are derived from
real bank data - they're stated, illustrative assumptions (see
decision_engine.py's docstring). This script shows how much the actual
recommendations move if those assumptions are varied within a realistic
range, so a reader can judge how much the final numbers actually depend on
them, rather than taking the headline example on faith.

Run with: python3 src/sensitivity_analysis.py
"""

from decision_engine import recommend_limit, breakeven_utilization

# A fixed example client, held constant across all scenarios below so only
# the assumptions change.
CLIENT = {"current_limit": 100000, "pd": 0.08, "expected_util": 0.45}


def interest_rate_sensitivity():
    print("=" * 70)
    print("Sensitivity to interest rate (LGD fixed at 0.75)")
    print("=" * 70)
    print(f"{'Interest rate':>15} {'Breakeven util':>16} {'Recommended limit':>19}")
    for rate in [0.12, 0.15, 0.18, 0.21, 0.24, 0.30]:
        breakeven = breakeven_utilization(CLIENT["pd"], interest_rate=rate)
        limit, profit, _ = recommend_limit(
            CLIENT["current_limit"], CLIENT["pd"], CLIENT["expected_util"], interest_rate=rate
        )
        print(f"{rate:>14.0%} {breakeven:>15.1%} {limit:>18,.0f}")


def lgd_sensitivity():
    print("\n" + "=" * 70)
    print("Sensitivity to loss-given-default (interest rate fixed at 0.18)")
    print("=" * 70)
    print(f"{'LGD':>15} {'Breakeven util':>16} {'Recommended limit':>19}")
    for lgd in [0.50, 0.60, 0.70, 0.75, 0.85, 0.95]:
        breakeven = breakeven_utilization(CLIENT["pd"], lgd=lgd)
        limit, profit, _ = recommend_limit(
            CLIENT["current_limit"], CLIENT["pd"], CLIENT["expected_util"], lgd=lgd
        )
        print(f"{lgd:>15.0%} {breakeven:>15.1%} {limit:>18,.0f}")


def pd_crossover_point():
    print("\n" + "=" * 70)
    print("At what PD does THIS client's fixed utilization (45%) stop")
    print("clearing breakeven, under default assumptions (18% / 75%)?")
    print("=" * 70)
    for pd in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        breakeven = breakeven_utilization(pd)
        limit, profit, _ = recommend_limit(CLIENT["current_limit"], pd, CLIENT["expected_util"])
        cleared = "above breakeven -> max limit" if CLIENT["expected_util"] > breakeven else "below breakeven -> floor"
        print(f"  PD={pd:>5.0%}  breakeven={breakeven:>6.1%}  {cleared:30s}  limit={limit:>10,.0f}")


if __name__ == "__main__":
    interest_rate_sensitivity()
    lgd_sensitivity()
    pd_crossover_point()
    print(
        "\nTakeaway: the recommended limit for a given client only changes "
        "when an assumption shift moves the breakeven utilization across "
        "that client's actual predicted utilization. Small assumption "
        "changes that don't cross that line leave the decision unchanged, "
        "which is a direct consequence of the bang-bang shape described in "
        "decision_engine.py."
    )
