"""Runner-side RunPod helper (used by the 'Tal: fast parallel training' workflow).

  python tools/runpod.py start NAME KEY=VAL ...   -> starts a GPU pod running tal.cloud_parallel, prints pod id
  python tools/runpod.py wait POD TAG ASSET MINUTES -> waits until ASSET in release TAG is newer than $STARTED
  python tools/runpod.py stop POD
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

API = "https://api.runpod.io/v2/pods"
KEY = os.environ["RUNPOD_API_KEY"]
REPO = os.environ["GH_REPO"]
GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA GeForce RTX 4090"]


def call(method, url, body=None, headers=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            t = r.read().decode()
            return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def rp(method, url, body=None):
    return call(method, url, body, {"Authorization": f"Bearer {KEY}"})


def start(name, extra):
    script = ("set -e; cd /workspace; git clone --depth 1 https://github.com/" + REPO + " repo; cd repo; "
              "pip install -q chess python-chess huggingface-hub numpy pyyaml requests; "
              "pip install -q git+https://github.com/CSSLab/maia3@1e13597c42d4858b7cfd7cfdae01e297263364b2; "
              "maia3-cache --model maia3-79m; "
              "python -u -m tal.cloud_parallel 2>&1 | tee /workspace/train.log; sleep infinity")
    env = {"GH_TOKEN": os.environ["GH_TOKEN"], "REPO": REPO, **extra}
    for cloud in ("SECURE", "COMMUNITY"):
        for gpu in GPUS:
            body = {"name": f"maia-tal-{name}", "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                    "gpu": {"id": gpu, "count": 1}, "cloud": cloud, "disk": 60, "env": env,
                    "args": "bash -c " + json.dumps(script)}
            st, r = rp("POST", API, body)
            if isinstance(r, dict) and r.get("id"):
                print(f"started {name}: {gpu} ({cloud}) cost/h={r.get('cost')}", file=sys.stderr)
                print(r["id"])
                return
            print(f"not available: {gpu} ({cloud})", file=sys.stderr)
    sys.exit("no GPU free right now")


def asset_time(tag, asset):
    st, r = call("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{tag}", None,
                 {"Authorization": f"Bearer {os.environ['GH_TOKEN']}"})
    if not isinstance(r, dict):
        return ""
    return next((a["updated_at"] for a in r.get("assets", []) if a["name"] == asset), "")


def pod_status(pod):
    st, r = rp("GET", f"{API}/{pod}")
    return r.get("desiredStatus") or r.get("status") if isinstance(r, dict) else f"http {st}"


def wait(pod, tag, asset, minutes):
    end = time.time() + float(minutes) * 60
    started = os.environ["STARTED"]
    while time.time() < end:
        up = asset_time(tag, asset)
        ps = pod_status(pod)
        print(time.strftime("%H:%M"), f"pod={ps} {asset}={up}", flush=True)
        if up and up > started:
            print("done")
            return
        if ps in ("ERROR", "TERMINATED", "EXITED"):
            sys.exit(f"pod stopped: {ps}")
        time.sleep(60)
    sys.exit("time limit reached")


def stop(pod):
    st, r = rp("DELETE", f"{API}/{pod}")
    print("terminated", pod, st)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "start":
        start(sys.argv[2], dict(a.split("=", 1) for a in sys.argv[3:]))
    elif cmd == "wait":
        wait(*sys.argv[2:6])
    elif cmd == "stop":
        stop(sys.argv[2])
