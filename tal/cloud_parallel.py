"""Pod-side parallel Tal plan. PLAN_MODE=pretrain or PLAN_MODE=variant (PLAN_VARIANT=name).

pretrain: similar-style players -> uploads pretrain-79m.pt + pretrain-info.json to release tal-runs.
variant:  downloads pretrain-79m.pt, fine-tunes on Tal -> uploads <name>.pt + <name>.json to tal-runs.
The workflow then picks the best variant and copies it to the tal-model release.
"""
import json
import os
import subprocess
import time

import requests
import torch

from tal.cloud_plan import GROUP, log, upload

REPO = os.environ["REPO"]
GH = {"Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json"}
TAG = "tal-runs"
VARIANTS = {   # blocks, lr, epochs, minutes
    "pre_b8_lr5e-6": ("8", "5e-6", "5", "20"),
    "pre_b4_lr1e-5": ("4", "1e-5", "6", "16"),
    "pre_all_lr3e-6": ("99", "3e-6", "4", "20"),
}


def run(env_extra):
    env = {**os.environ, "TAL_BATCH": "256", "TAL_WORKERS": "12", "TAL_EVAL": "3000", "TAL_EVAL_EVERY": "300", **env_extra}
    p = subprocess.run(["python", "-u", "-m", "tal.train"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout, flush=True)
    if p.returncode != 0:
        raise RuntimeError("training failed:\n" + "\n".join(p.stdout.splitlines()[-25:]))
    return json.load(open("tal-info.json"))


def download(name, path, tag=TAG):
    rel = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", headers=GH, timeout=30).json()
    a = next(a for a in rel["assets"] if a["name"] == name)
    with requests.get(a["url"], headers={**GH, "Accept": "application/octet-stream"}, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    log("downloaded", name, os.path.getsize(path))


def main():
    mode = os.environ["PLAN_MODE"]
    if os.environ.get("GAMES_ASSET"):     # games prefetched by the GitHub runner
        download(os.environ["GAMES_ASSET"], "games.pgn.gz")
        os.environ["GAMES_FILE"] = os.path.abspath("games.pgn.gz")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log("mode", mode, "gpu", gpu, "cpus", os.cpu_count())
    if mode == "pretrain":
        out = {"gpu": gpu}
        try:
            out.update(run({"PLAYER": GROUP, "PLAYER_MAX_GAMES": "0", "TAL_EPOCHS": "2", "TAL_TRAIN_BLOCKS": "12",
                            "TAL_LR": "1e-5", "TAL_MINUTES": os.environ.get("PRE_MINUTES", "50"),
                            "TAL_OUT": "pretrain-79m.pt", "TAL_EVAL": "1000"}))
            upload(TAG, "pretrain-79m.pt", "pretrain-79m.pt")
        except Exception as e:
            out["error"] = str(e)
        json.dump(out, open("pretrain-info.json", "w"), indent=1)
        upload(TAG, "pretrain-info.json", "pretrain-info.json")   # done signal
        return
    if mode == "imitate":   # player imitation (rules + own book) on the GPU: much faster than GitHub's CPU runner
        name = os.environ["PLAYER_NAME"]
        out = {"gpu": gpu}
        try:
            os.makedirs("prev", exist_ok=True)
            download(f"{name}-79m.pt", f"prev/{name}-79m.pt", tag=f"{name}-model")
            env = {**os.environ, "IMIT_NAME": name, "TAL_CKPT": f"prev/{name}-79m.pt", "GITHUB_STEP_SUMMARY": "imitate-summary.md"}
            p = subprocess.run(["python", "-u", "-m", "tal.imitate"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(p.stdout, flush=True)
            open(f"{name}-imitate.log", "w").write(p.stdout)
            upload(TAG, f"{name}-imitate.log", f"{name}-imitate.log")
            if p.returncode != 0:
                raise RuntimeError("\n".join(p.stdout.splitlines()[-25:]))
            upload(TAG, f"tal/{name}-rules.json", f"{name}-rules.json")
            upload(TAG, f"book-{name}.json", f"book-{name}.json")
            out["summary"] = open("imitate-summary.md").read() if os.path.exists("imitate-summary.md") else ""
        except Exception as e:
            out["error"] = str(e)
        json.dump(out, open(f"{name}-imitate.json", "w"), indent=1)
        upload(TAG, f"{name}-imitate.json", f"{name}-imitate.json")   # done signal
        return
    if mode == "player":    # any player (e.g. Carlsen): fine-tune on PLAYER -> <name>-79m.pt + <name>-info.json in tal-runs
        name = os.environ["PLAYER_NAME"]
        out = {"gpu": gpu}
        try:
            out.update(run({"PLAYER": os.environ["PLAYER"], "PLAYER_MAX_GAMES": os.environ.get("PLAYER_MAX_GAMES", "0"),
                            "TAL_EPOCHS": os.environ.get("TAL_EPOCHS", "3"), "TAL_TRAIN_BLOCKS": os.environ.get("TAL_TRAIN_BLOCKS", "8"),
                            "TAL_LR": os.environ.get("TAL_LR", "5e-6"), "TAL_MINUTES": os.environ.get("TAL_MINUTES", "100"),
                            "TAL_OUT": f"{name}-79m.pt", "TAL_RESUME": ""}))
            upload(TAG, f"{name}-79m.pt", f"{name}-79m.pt")
        except Exception as e:
            out["error"] = str(e)
        json.dump(out, open(f"{name}-info.json", "w"), indent=1)
        upload(TAG, f"{name}-info.json", f"{name}-info.json")   # done signal
        return
    if mode == "young":     # young, wild Tal (1949-1962): continue from the current Tal model
        out = {"gpu": gpu}
        try:
            download("tal-79m.pt", "tal-79m.pt", tag="tal-model")
            for name, blocks, lr, epochs in (("young_b4", "4", "1e-5", "6"), ("young_b8", "8", "5e-6", "5")):
                out[name] = run({"PLAYER": "tal", "PLAYER_MAX_GAMES": "0", "TAL_YEARS": os.environ.get("TAL_YEARS", "1949-1962"),
                                 "TAL_EPOCHS": epochs, "TAL_TRAIN_BLOCKS": blocks, "TAL_LR": lr, "TAL_MINUTES": "20",
                                 "TAL_OUT": f"{name}.pt", "TAL_RESUME": "tal-79m.pt", "TAL_EVAL": "2000"})
                upload(TAG, f"{name}.pt", f"{name}.pt")
        except Exception as e:
            out["error"] = str(e)
        json.dump(out, open("young-info.json", "w"), indent=1)
        upload(TAG, "young-info.json", "young-info.json")   # done signal
        return
    if mode == "tactics":    # tactics training: blunders removed from the model's instinct, Tal style kept
        out = {"gpu": gpu}
        try:
            download("tactics.jsonl.gz", "tactics.jsonl.gz")
            download("study.jsonl.gz", "study.jsonl.gz")
            download("tal-79m.pt", "tal-79m.pt", tag="tal-model")
            env = {**os.environ, "TACTICS_FILE": "tactics.jsonl.gz", "STUDY_FILE": "study.jsonl.gz",
                   "TAL_RESUME": "tal-79m.pt", "TT_OUT": "tactics.pt"}
            p = subprocess.run(["python", "-u", "-m", "tal.tactics"], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            print(p.stdout, flush=True)
            if p.returncode != 0:
                raise RuntimeError("tactics training failed:\n" + "\n".join(p.stdout.splitlines()[-25:]))
            out.update(json.load(open("tactics-info.json")))
            if os.path.exists("tactics.pt"):
                upload(TAG, "tactics.pt", "tactics.pt")
        except Exception as e:
            out["error"] = str(e)
        json.dump(out, open("tactics-info.json", "w"), indent=1)
        upload(TAG, "tactics-info.json", "tactics-info.json")   # done signal
        return
    if mode == "antiblunder":
        out = {"gpu": gpu}
        try:
            download("study.jsonl.gz", "study.jsonl.gz")
            download("tal-79m.pt", "tal-79m.pt", tag="tal-model")
            env = {**os.environ, "STUDY_FILE": "study.jsonl.gz", "TAL_RESUME": "tal-79m.pt", "AB_OUT": "antiblunder.pt"}
            p = subprocess.run(["python", "-u", "-m", "tal.antiblunder"], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            print(p.stdout, flush=True)
            if p.returncode != 0:
                raise RuntimeError("anti-blunder training failed:\n" + "\n".join(p.stdout.splitlines()[-25:]))
            out.update(json.load(open("antiblunder-info.json")))
            upload(TAG, "antiblunder.pt", "antiblunder.pt")
        except Exception as e:
            out["error"] = str(e)
        json.dump(out, open("antiblunder-info.json", "w"), indent=1)
        upload(TAG, "antiblunder-info.json", "antiblunder-info.json")   # done signal
        return
    name = os.environ["PLAN_VARIANT"]
    blocks, lr, epochs, minutes = VARIANTS[name]
    out = {"gpu": gpu}
    try:
        download("pretrain-79m.pt", "pretrain-79m.pt")
        out.update(run({"PLAYER": "tal", "PLAYER_MAX_GAMES": "0", "TAL_EPOCHS": epochs, "TAL_TRAIN_BLOCKS": blocks,
                        "TAL_LR": lr, "TAL_MINUTES": minutes, "TAL_OUT": f"{name}.pt",
                        "TAL_RESUME": "pretrain-79m.pt"}))
        upload(TAG, f"{name}.pt", f"{name}.pt")
    except Exception as e:
        out["error"] = str(e)
    json.dump(out, open(f"{name}.json", "w"), indent=1)
    upload(TAG, f"{name}.json", f"{name}.json")   # done signal


if __name__ == "__main__":
    main()
