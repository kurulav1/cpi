#!/usr/bin/env python3
"""A logging pass-through in front of CPI, so the wire traffic is visible.

Two jobs, both of which came out of this week's failures.

It records what the FRAMEWORK actually sent: the system prompt it composed, the tool
schemas it generated, the message shapes it used across turns. Reading a final answer
tells you nothing about any of that, and every integration problem worth finding
lives in it.

And it proves the framework reached CPI at all. An SDK pointed at a bad base URL can
fall back to a default endpoint, or read a key from the environment and talk to
somebody else's server entirely; the run would look fine and would be measuring
another model. If this log is empty, nothing downstream means anything.

Usage: python tools/logging_proxy.py --listen 8081 --upstream http://127.0.0.1:8080 --log wire.jsonl
"""

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8080"
LOGPATH = "wire.jsonl"
_lock = threading.Lock()
_count = 0


def _log(record: dict) -> None:
    global _count
    with _lock:
        _count += 1
        record["seq"] = _count
        with open(LOGPATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quiet; the JSONL is the log
        pass

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        url = UPSTREAM + self.path
        req = urllib.request.Request(url, data=body if body else None, method=method)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "connection"):
                continue
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
                status = r.status
                ctype = r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            data = e.read()
            status = e.code
            ctype = e.headers.get("Content-Type", "application/json")
        except Exception as e:  # noqa: BLE001 - the client must see the failure
            data = json.dumps({"error": str(e)}).encode()
            status = 502
            ctype = "application/json"

        def parse(b: bytes):
            try:
                return json.loads(b.decode())
            except Exception:  # noqa: BLE001
                return b.decode(errors="replace")[:4000]

        _log({
            "path": self.path,
            "method": method,
            "status": status,
            "request": parse(body) if body else None,
            "response": parse(data),
        })

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


def main() -> int:
    global UPSTREAM, LOGPATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8081)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--log", default="wire.jsonl")
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    LOGPATH = args.log
    open(LOGPATH, "w", encoding="utf-8").close()
    srv = ThreadingHTTPServer(("127.0.0.1", args.listen), Handler)
    print("proxy on %d -> %s, logging to %s" % (args.listen, UPSTREAM, LOGPATH), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
