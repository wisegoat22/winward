"""Try the local API: .venv/bin/python decide.py 'Your ticket here'."""
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jev_local.examples import EXAMPLES

case = EXAMPLES[0]
body = {k: case[k] for k in ("state", "question", "choices")}
if len(sys.argv) > 1:
    body["state"] = " ".join(sys.argv[1:])
request = Request("http://127.0.0.1:8765/api/decide", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
try:
    with urlopen(request, timeout=120) as response:
        print(json.dumps(json.load(response), indent=2))
except HTTPError as error:
    sys.exit(f"Request failed ({error.code}): {error.read().decode()}")
except URLError:
    sys.exit("JEV Local is not running. Open Start JEV.command first.")
