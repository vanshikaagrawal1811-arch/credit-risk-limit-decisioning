"""
decision_engine.py
Turns the two models' outputs into a credit limit recommendation.

This module is deliberately NOT machine learning. There is no ground-truth
"correct limit" to fit against - the LIMIT_BAL column in the raw data
reflects a bank's past policy, not an objective target. Learning to predict
it would just imitate that historical policy (including its biases).

Instead, this is a decision rule built on expected profit:

expected_profit(L) = (1 - PD) * interest_rate * expected_util * L
                      - PD * loss_given_default * L

The first term is expected interest income if the client doesn't default;
the second is expected loss if they do. Both terms scale linearly with the
candidate limit L, so the sign of (marginal profit per dollar of limit) does
not depend on L at all - only on whether:

    expected_util  >  breakeven_utilization(PD)
    where breakeven_utilization(PD) = (PD * lgd) / ((1 - PD) * interest_rate)

If expected utilization clears that breakeven point, extending MORE limit is
always better (approve up to the policy ceiling); if it doesn't, extending
limit only adds risk with no offsetting income (extend the minimum). This
"bang-bang" shape isn't a bug - real underwriting behaves similarly: give
your most profitable, most engaged low-risk customers the most room, and
minimize exposure to the rest, rather than picking timid limits in between.

This is exactly why Model 2 (predicted utilization) matters: for clients
whose true utilization sits close to their breakeven point, a naive flat
utilization assumption (e.g. "assume everyone uses 50%") can land on the
wrong side of that line and flip the decision, whereas a per-client
prediction gets it right. See compare_naive_vs_model_utilization() below,
and the app's side-by-side comparison.

Assumptions (illustrative, not sourced from a real bank's loss data -
stated explicitly, same as the 5:1 cost ratio in train.py):
  - annual interest rate on carried balance: 18%   (typical revolving APR)
  - loss given default: 75%                        (common industry rule of thumb)
"""

DEFAULT_INTEREST_RATE = 0.18
DEFAULT_LGD = 0.75
DEFAULT_MIN_MULTIPLE = 0.5   # floor: don't recommend below 50% of current limit
DEFAULT_MAX_MULTIPLE = 1.5   # ceiling: don't recommend above 150% of current limit
DEFAULT_DECLINE_PD = 0.5     # above this PD, always floor regardless of utilization


def breakeven_utilization(pd, interest_rate=DEFAULT_INTEREST_RATE, lgd=DEFAULT_LGD):
    """
    The utilization level at which extending more credit is exactly
    profit-neutral for a client with this PD. Above it, more limit helps;
    below it, more limit only adds risk.
    """
    if pd >= 1:
        return float("inf")
    return (pd * lgd) / ((1 - pd) * interest_rate)


def expected_profit(limit, pd, expected_util, interest_rate=DEFAULT_INTEREST_RATE, lgd=DEFAULT_LGD):
    """Expected annual profit for a single candidate limit."""
    return (1 - pd) * interest_rate * expected_util * limit - pd * lgd * limit


def recommend_limit(
    current_limit,
    pd,
    expected_util,
    interest_rate=DEFAULT_INTEREST_RATE,
    lgd=DEFAULT_LGD,
    min_multiple=DEFAULT_MIN_MULTIPLE,
    max_multiple=DEFAULT_MAX_MULTIPLE,
    decline_pd=DEFAULT_DECLINE_PD,
):
    """
    Compares expected_util against the client's breakeven utilization and
    picks whichever end of the [min_multiple, max_multiple] * current_limit
    band maximizes expected profit. Very high-risk clients (pd >= decline_pd)
    are floored regardless, since a near-certain defaulter is never worth
    extending more credit to, however high their utilization prediction.

    Returns (recommended_limit, expected_profit_at_that_limit, breakeven_util).
    """
    breakeven = breakeven_utilization(pd, interest_rate, lgd)

    if pd >= decline_pd or expected_util <= breakeven:
        best_limit = current_limit * min_multiple
    else:
        best_limit = current_limit * max_multiple

    best_limit = round(float(best_limit), -3)
    best_profit = expected_profit(best_limit, pd, expected_util, interest_rate, lgd)

    return best_limit, float(best_profit), float(breakeven)


def compare_naive_vs_model_utilization(current_limit, pd, model_util, naive_util=0.5, **kwargs):
    """
    Demonstrates that Model 2 actually changes the decision: computes the
    recommended limit twice, once using a flat naive utilization assumption
    and once using the model's predicted utilization, and returns both so
    the difference can be shown side by side (see app.py). The interesting
    case is when naive_util and model_util fall on opposite sides of the
    client's breakeven utilization - that's where the decision itself flips,
    not just the profit estimate.
    """
    naive_limit, naive_profit, breakeven = recommend_limit(current_limit, pd, naive_util, **kwargs)
    model_limit, model_profit, _ = recommend_limit(current_limit, pd, model_util, **kwargs)
    return {
        "breakeven_utilization": breakeven,
        "naive_utilization_assumed": naive_util,
        "naive_recommended_limit": naive_limit,
        "naive_expected_profit": naive_profit,
        "model_predicted_utilization": model_util,
        "model_recommended_limit": model_limit,
        "model_expected_profit": model_profit,
        "limit_delta": model_limit - naive_limit,
        "decision_flipped": (naive_limit != model_limit),
    }


if __name__ == "__main__":
    print("Example A: naive assumption and model prediction land on the same")
    print("side of breakeven -> same decision, different profit estimate.\n")
    result_a = compare_naive_vs_model_utilization(current_limit=100000, pd=0.05, model_util=0.65)
    for k, v in result_a.items():
        print(f"  {k:28s}: {v}")

    print("\nExample B: a client with PD=0.10 (breakeven util ~0.46). Naive 0.5")
    print("assumption says 'approve max'; if the real predicted utilization is")
    print("only 0.30, Model 2 flips the decision to 'floor'.\n")
    result_b = compare_naive_vs_model_utilization(current_limit=100000, pd=0.10, model_util=0.30)
    for k, v in result_b.items():
        print(f"  {k:28s}: {v}")
