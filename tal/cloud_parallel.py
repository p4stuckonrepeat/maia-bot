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
    "pre_b8_lr5e-6": ("8", "5e-6", "5", "25"),
    "pre_b4_lr1e-5": ("4", "1e-5", "6", "20"),
    "pre_all_lr3e-6": ("99", "3e-6", "4", "25"),
}


def run(env_extra):
    env = {**os.environ, "TAL_BATCH": "256", "TAL_EVAL": "3000", "TAL_EVAL_EVERY": "300", **env_extra}
    p = subprocess.run(["python", "-u", "-m", "tal.train"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout, flush=True)
    if p.returncode != 0:
        raise RuntimeError("training failed:\n" + "\n".join(p.stdout.splitlines()[-25:]))
    return json.load(open("tal-info.json"))


def download(name, path):
    rel = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", headers=GH, timeout=30).json()
    a = next(a for a in rel["assets"] if a["name"] == name)
    with requests.get(a["url"], headers={**GH, "Accept": "application/octet-stream"}, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    log("downloaded", name, os.path.getsize(path))


def main():
    mode = os.environ["PLAN_MODE"]
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
