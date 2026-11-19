import math

import numpy as np
import pandas as pd

from monbio.metrics import aggregate_rotations, aurc, auroc, ece, option_probs, wilson_ci


def test_wilson_ci():
    lo, hi = wilson_ci(20, 50)
    assert abs(lo - 0.2760) < 1e-3 and abs(hi - 0.5382) < 1e-3
    lo, hi = wilson_ci(0, 10)
    assert lo == 0 or abs(lo) < 1e-12
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    assert math.isnan(auroc([0.3, 0.4], [1, 1]))
    # one tie between a correct and a wrong answer counts one half
    assert auroc([0.9, 0.5, 0.5], [1, 1, 0]) == 0.75


def test_ece_and_aurc():
    assert ece([1.0, 1.0, 0.0], [1, 1, 0]) == 0.0
    assert abs(ece([0.8] * 10, [1] * 8 + [0] * 2)) < 1e-12
    assert aurc([0.9, 0.8, 0.7], [1, 1, 1]) == 0.0
    # Confident error first is worse than confident error last.
    assert aurc([0.9, 0.5], [0, 1]) > aurc([0.9, 0.5], [1, 0])


def test_option_probs_maps_positions_back():
    lp = np.log(np.array([[0.7, 0.1, 0.1, 0.1]]))  # model picks position A
    out = option_probs(lp, ["1230"])  # position A shows original option 1
    assert np.allclose(out, [[0.1, 0.7, 0.1, 0.1]])


def _rows(position_probs_by_rotation):
    rows = []
    for r, probs in enumerate(position_probs_by_rotation):
        perm = "".join(str((p + r) % 4) for p in range(4))
        lp = np.log(probs)
        rows.append({"model": "m", "condition": "en", "variant": "instr_en", "item_id": "x", "rotation": r,
                     "perm": perm, "lp_A": lp[0], "lp_B": lp[1], "lp_C": lp[2], "lp_D": lp[3],
                     "first_lp_A": lp[3], "first_lp_B": lp[2], "first_lp_C": lp[1], "first_lp_D": lp[0],
                     "letter_mass": 1.0, "first_letter_mass": 0.5, "no_letter": False,
                     "n_tokens": 10, "nonfinite": False})
    return pd.DataFrame(rows)


def test_aggregate_rotations_readouts_differ():
    rotations = [np.array([0.1, 0.1, 0.7, 0.1])] * 4
    decision = aggregate_rotations(_rows(rotations), readout="decision").iloc[0]
    first = aggregate_rotations(_rows(rotations), readout="first").iloc[0]
    assert decision["letter_mass"] == 1.0 and first["letter_mass"] == 0.5
    assert decision["no_letter"] == 0.0


def test_aggregate_rotations_content_preference():
    # The model always prefers original option 2, wherever it is shown.
    rotations = []
    for r in range(4):
        pos_of_option2 = (2 - r) % 4
        probs = np.full(4, 0.1)
        probs[pos_of_option2] = 0.7
        rotations.append(probs)
    out = aggregate_rotations(_rows(rotations)).iloc[0]
    assert out["pred"] == 2
    assert abs(out["confidence"] - 0.7) < 1e-9
    assert out["consistency"] == 1.0


def test_aggregate_rotations_cancels_letter_bias():
    # The model always says "A" whatever the options are: pure position bias.
    out = aggregate_rotations(_rows([np.array([0.7, 0.1, 0.1, 0.1])] * 4)).iloc[0]
    assert np.allclose([out[f"p{j}"] for j in range(4)], 0.25)
    assert out["consistency"] == 0.25


def test_aggregate_rotations_skips_nonfinite():
    df = _rows([np.array([0.1, 0.1, 0.7, 0.1])] * 4)
    df.loc[0, "nonfinite"] = True
    df.loc[0, "lp_A"] = float("nan")
    out = aggregate_rotations(df).iloc[0]
    assert out["n_ok"] == 3
