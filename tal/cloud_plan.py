"""Paid GPU plan for Tal (runs on a rented RunPod GPU, started by the 'Tal: train on RunPod' workflow).

1. Pretrain on attacking players with a similar style (whole model, 1 pass).
2. Several Tal fine-tunes (different depth / learning rate), plus one without pretraining for comparison.
3. Every variant is measured on the same 3,000 positions from Tal games it never saw.
4. The best one is uploaded as the Tal model, if it beats the current one. Results are always uploaded.
"""
import json
import os
import subprocess
import time

import requests

REPO = os.environ["REPO"]
GH = {"Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json"}
GROUP = "group:Alekhine,Bronstein,Kasparov,Shirov,Topalov,Morozevich"
CURRENT_BEST = float(os.environ.get("CURRENT_BEST", "0.594"))   # tal-model so far: 59.4%

VARIANTS = [
    # name, resume from pretrain?, blocks, lr, epochs
    ("pre_b8_lr5e-6", True, 8, "5e-6", "4"),
    ("pre_b4_lr1e-5", True, 4, "1e-5", "5"),
    ("pre_b2_lr2e-5", True, 2, "2e-5", "6"),
    ("nopre_b8_lr5e-6", False, 8, "5e-6", "4"),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(env_extra):
    env = {**os.environ, "TAL_BATCH": "128", "TAL_EVAL": "3000", "TAL_EVAL_EVERY": "300", **env_extra}
    subprocess.run(["python", "-u", "-m", "tal.train"], check=True, env=env)
    return json.load(open("tal-info.json"))


def release(tag):
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", headers=GH, timeout=30)
    if r.status_code == 404:
        r = requests.post(f"https://api.github.com/repos/{REPO}/releases", headers=GH, timeout=30,
                          json={"tag_name": tag, "name": tag, "body": "Maia-3 79M fine-tuned on Tal"})
    return r.json()


def upload(tag, path, name):
    rel = release(tag)
    for a in rel.get("assets", []):
        if a["name"] == name:
            requests.delete(a["url"], headers=GH, timeout=30)
    with open(path, "rb") as f:
        r = requests.post(f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets",
                          params={"name": name}, data=f, timeout=600,
                          headers={**GH, "Content-Type": "application/octet-stream"})
    log("uploaded", name, r.status_code)


def main():
    results = {}
    try:
        log("stage 1: pretraining on", GROUP)
        pre = run({"PLAYER": GROUP, "PLAYER_MAX_GAMES": "0", "TAL_EPOCHS": "1", "TAL_TRAIN_BLOCKS": "8",
                   "TAL_LR": "1e-5", "TAL_MINUTES": "60", "TAL_OUT": "pretrain-79m.pt", "TAL_EVAL": "1000"})
        results["pretrain"] = pre
        for name, use_pre, blocks, lr, epochs in VARIANTS:
            log("stage 2:", name)
            env = {"PLAYER": "tal", "PLAYER_MAX_GAMES": "0", "TAL_EPOCHS": epochs, "TAL_TRAIN_BLOCKS": str(blocks),
                   "TAL_LR": lr, "TAL_MINUTES": "40", "TAL_OUT": f"{name}.pt", "TAL_RESUME": ""}
            if use_pre:
                env["TAL_RESUME"] = "pretrain-79m.pt"
            try:
                results[name] = run(env)
            except subprocess.CalledProcessError as e:
                results[name] = {"error": str(e)}
            log(name, results[name].get("before"), "->", results[name].get("after"))
        ok = {k: v for k, v in results.items() if k != "pretrain" and "after" in v}
        best = max(ok, key=lambda k: ok[k]["after"]) if ok else None
        results["best"] = best
        if best and ok[best]["after"] > CURRENT_BEST:
            results["saved"] = True
            upload("tal-model", f"{best}.pt", "tal-79m.pt")
        else:
            results["saved"] = False
    finally:
        json.dump(results, open("runpod-results.json", "w"), indent=1)
        if os.path.exists("/workspace/train.log"):
            upload("tal-model", "/workspace/train.log", "runpod-train.log")
        upload("tal-model", "runpod-results.json", "runpod-results.json")   # also the "done" signal


if __name__ == "__main__":
    main()
