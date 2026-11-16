"""How often does each bot version play exactly Tal's move? (held-out Tal positions from tal/study.py)

Uses the stored study data (Tal model candidates + Stockfish verdicts + features), and simulates each
version's move choice: Tal rules (fitted weights), Stockfish weight, Tal-traits boost, temperature,
instinct moves (lapses) and the sound-sacrifice hunter. Reports the expected % of exact Tal moves.
"""
import gzip
import json
import math
import os

FEATURES = None   # filled from tal.study at runtime
VERSIONS = {
    "Tal model alone": {"model_only": True, "model_temp": 0.75},
    "v1 (frozen live)": {"sfw": 0.5, "temp": 1.0, "boost": 1.0, "lapse": 0.10, "lapse_mode": "natural",
                         "model_temp": 0.75, "sac": 1.0},
    "v2": {"sfw": 0.4, "temp": 1.3, "boost": 1.0, "lapse": 0.12, "lapse_mode": "natural", "model_temp": 1.0, "sac": 1.0,
           "curve": {"start": 20, "per_move": 0.008, "ahead_cp": 300, "ahead_extra": 0.15, "ahead_sf": 0.5,
                     "late": 30, "late_extra": 0.08, "late_sf": 0.6, "max": 0.45}},
    "v3 (live now)": {"sfw": 0.4, "temp": 1.0, "boost": 1.6, "lapse": 0.12, "lapse_mode": "tal", "model_temp": 0.75, "sac": 1.0,
                      "curve": {"start": 20, "per_move": 0.008, "ahead_cp": 300, "ahead_extra": 0.15, "ahead_sf": 0.5,
                                "late": 30, "late_extra": 0.08, "late_sf": 0.6, "max": 0.45}},
    "v3 + Tal's real sacrifice rate (45%)": {"sfw": 0.4, "temp": 1.0, "boost": 1.6, "lapse": 0.12, "lapse_mode": "tal", "model_temp": 0.75, "sac": 0.45, "curve": "v3"},
    "v3, temp 0.6, sac 45%": {"sfw": 0.4, "temp": 0.6, "boost": 1.6, "lapse": 0.12, "lapse_mode": "tal", "model_temp": 0.75, "sac": 0.45, "curve": "v3"},
    "v3, temp 0.5, boost 1.0, sac 45%": {"sfw": 0.4, "temp": 0.5, "boost": 1.0, "lapse": 0.12, "lapse_mode": "tal", "model_temp": 0.75, "sac": 0.45, "curve": "v3"},
    "v3, temp 0.4, boost 1.3, sac 45%": {"sfw": 0.4, "temp": 0.4, "boost": 1.3, "lapse": 0.12, "lapse_mode": "tal", "model_temp": 0.75, "sac": 0.45, "curve": "v3"},
    "v3, temp 0.3, boost 1.3, sac 45%": {"sfw": 0.4, "temp": 0.3, "boost": 1.3, "lapse": 0.12, "lapse_mode": "tal", "model_temp": 0.75, "sac": 0.45, "curve": "v3"},
    "Tal rules, best-scored move (upper bound)": {"argmax": True, "sfw": 1.0, "boost": 1.0},
}


def softmax(xs, t):
    m = max(xs)
    e = [math.exp((x - m) / t) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def probs(r, v, rules, F, sfw_factor=1.0):
    names, w = rules["features"], rules["weights"]
    cands = r["cands"][:r["n_top"]]
    lg = []
    for c in cands:
        f = c["f"]
        lg.append(sum(w[n] * f[F[n]] * (v.get("sfw", 1.0) * sfw_factor if n == "sf_loss" else
                                          (1.0 if n == "logp" else v.get("boost", 1.0))) for n in names))
    return cands, lg


def tal_prob(r, v, rules, F):
    cands = r["cands"][:r["n_top"]]
    ti = next((i for i, c in enumerate(cands) if c["m"] == r["tal"]), None)
    nat = softmax([math.log(max(c["p"], 1e-9)) for c in cands], v.get("model_temp", 0.75))
    if v.get("model_only"):
        return nat[ti] if ti is not None else 0.0
    if v.get("argmax"):
        _, lg = probs(r, v, rules, F)
        return 1.0 if ti is not None and lg.index(max(lg)) == ti else 0.0
    # sound-sacrifice hunter: plays an available sound sacrifice (weighted by Maia)
    sacs = [i for i, c in enumerate(cands) if c["f"][F["sound_sac"]] > 0]
    p_sac = v.get("sac", 0) if sacs else 0.0
    sac_dist = [0.0] * len(cands)
    if sacs:
        tot = sum(cands[i]["p"] + 0.01 for i in sacs)
        for i in sacs:
            sac_dist[i] = (cands[i]["p"] + 0.01) / tot
    # lapse probability and Stockfish factors
    move_no = r["ply"] // 2 + 1
    lapse, sf_factor = v.get("lapse", 0.0), 1.0
    cv = v.get("curve")
    if cv == "v3":
        cv = VERSIONS["v3 (live now)"]["curve"]
    if cv:
        lapse += cv["per_move"] * max(0, move_no - cv["start"])
        if r["best"] > cv["ahead_cp"]:
            lapse += cv["ahead_extra"]
            sf_factor *= cv["ahead_sf"]
        if move_no > cv["late"]:
            lapse += cv["late_extra"]
            sf_factor *= cv["late_sf"]
        lapse = min(lapse, cv["max"])
    _, lg = probs(r, v, rules, F, sf_factor)
    normal = softmax(lg, v["temp"])
    if v["lapse_mode"] == "tal":
        _, lg0 = probs(r, v, rules, F, 0.0)
        lapse_dist = softmax(lg0, v["temp"])
    else:
        lapse_dist = nat
    dist = [(1 - lapse) * a + lapse * b for a, b in zip(normal, lapse_dist)]
    dist = [(1 - p_sac) * a + p_sac * s for a, s in zip(dist, sac_dist)]
    return dist[ti] if ti is not None else 0.0


def exp_loss(r, v, rules, F):
    cands = r["cands"][:r["n_top"]]
    out = 0.0
    for i, c in enumerate(cands):
        r2 = dict(r, tal=c["m"])
        out += tal_prob(r2, v, rules, F) * min(500, max(0, r["best"] - c["cp"]))
    return out


def main():
    import tal.study as study
    F = {n: i for i, n in enumerate(study.FEATURES)}
    rules = json.load(open("tal/tal-rules2.json"))
    with gzip.open(os.environ.get("STUDY_FILE", "study.jsonl.gz"), "rt") as f:
        hold = [r for r in (json.loads(l) for l in f) if r["split"] == "hold"]
    L = [f"# How often each version plays exactly Tal's move ({len(hold)} Tal positions the bot never trained on)", "",
         "| Version | Exact Tal move | Expected loss (cp, Tal himself ~17) |", "|---|---|---|"]
    for name, v in VERSIONS.items():
        acc = sum(tal_prob(r, v, rules, F) for r in hold) / len(hold)
        loss = sum(exp_loss(r, v, rules, F) for r in hold) / len(hold)
        L.append(f"| {name} | {acc:.1%} | {loss:.0f} |")
    text = "\n".join(L)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n")


if __name__ == "__main__":
    main()
