"""Start and stop this project's local background server."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
PIDFILE = RUNTIME / "server.pid"
PYTHON = ROOT / ".venv/bin/python"
URL = "http://127.0.0.1:8765"


def health():
    try:
        with urlopen(URL + "/api/health", timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def owned_pid():
    try:
        pid = int(PIDFILE.read_text().strip())
        command = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
        expected = f"{PYTHON} -m uvicorn jev_local.app:app "
        return pid if command.startswith(expected) else None
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / "control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "status":
            print(json.dumps(health() or {"status": "stopped"}, indent=2))
            return
        pid = owned_pid()
        if args.action == "stop":
            if pid:
                os.kill(pid, signal.SIGTERM)
                for _ in range(100):
                    time.sleep(0.1)
                    if not owned_pid():
                        PIDFILE.unlink(missing_ok=True)
                        print("JEV Local stopped.")
                        return
                print("Shutdown requested. The server is finishing its active request.")
            else:
                print("No server started by this project is running.")
            return
        current = health()
        if current:
            if current.get("revision") == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b":
                print(f"JEV Local is already running: {URL}")
                return
            sys.exit("Port 8765 is in use by another service.")
        if pid:
            sys.exit("JEV Local is already starting. Check .runtime/server.log.")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", PYTHONUNBUFFERED="1")
        with (RUNTIME / "server.log").open("w") as log:
            process = subprocess.Popen(
                [str(PYTHON), "-m", "uvicorn", "jev_local.app:app", "--host", "127.0.0.1",
                 "--port", "8765", "--no-access-log"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        PIDFILE.write_text(str(process.pid))
        print("Starting the local Winward lab…", flush=True)
        for _ in range(120):
            if process.poll() is not None:
                PIDFILE.unlink(missing_ok=True)
                print((RUNTIME / "server.log").read_text())
                sys.exit("Startup failed. If model files are missing, run setup.command.")
            if health():
                print(f"JEV Local is ready: {URL}")
                print("It runs in the background. Use Stop JEV.command to stop it.")
                return
            time.sleep(0.5)
        sys.exit("Model is still loading. See .runtime/server.log for progress.")


if __name__ == "__main__":
    main()
