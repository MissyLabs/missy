#!/usr/bin/env python3
"""Isolated native-binary HTTP + Chromium acceptance, not a Python runtime bridge.

Requires the optional test dependency `playwright` and its Chromium browser.
Never reads production Missy configuration or uses real provider credentials.
"""
import argparse
import json
import os
import pathlib
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/debug/missy-rs")
    args = parser.parse_args()
    binary = str(pathlib.Path(args.binary).resolve())
    requests = []
    response = '<img src=x onerror="window.__missy_xss=true"> Native Rust console receipt.'

    class Fixture(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            size = int(self.headers["Content-Length"])
            assert 0 < size < 2_000_000
            requests.append(json.loads(self.rfile.read(size)))
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": response}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    fixture = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    process = None
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    try:
        with tempfile.TemporaryDirectory(prefix="missy-rust-console-") as directory:
            root = pathlib.Path(directory)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            token = secrets.token_urlsafe(48)
            origin = f"http://127.0.0.1:{fixture.server_port}"
            url = f"http://127.0.0.1:{port}"
            config = {"version": 1, "state_dir": str(root / "state"), "providers": [{"name": "fixture", "kind": "openai", "base_url": origin + "/v1", "model": "fixture-model", "api_key_env": "MISSY_FIXTURE_KEY", "input_microusd_per_million": 0, "output_microusd_per_million": 0}], "policy": {"allowed_origins": [origin], "allow_private_network": True}, "api": {"bind": f"127.0.0.1:{port}", "token_env": "MISSY_API_TOKEN", "tenant": "fixture", "user": "operator"}}
            path = root / "config.yaml"
            path.write_text(json.dumps(config))
            path.chmod(0o600)
            env = {"HOME": directory, "PATH": "/usr/bin:/bin", "MISSY_API_TOKEN": token, "MISSY_FIXTURE_KEY": "fixture-only"}
            process = subprocess.Popen([binary, "--config", str(path), "serve"], env=env, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("isolated gateway exited: " + process.communicate()[1].decode())
                try:
                    with urllib.request.urlopen(url, timeout=0.2) as result:
                        check(result.status == 200, "console served by native executable")
                        check("default-src 'none'" in result.headers["Content-Security-Policy"], "restrictive CSP")
                    break
                except urllib.error.URLError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("isolated gateway did not become ready")
            try:
                urllib.request.urlopen(url + "/api/v1/health")
                raise AssertionError("anonymous API accepted")
            except urllib.error.HTTPError as error:
                check(error.code == 401, "unauthenticated API denied")

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                page.locator("#token").fill(token)
                page.locator("#auth button").first.click()
                check(page.locator("#token").input_value() == "", "token input cleared after unlock")
                page.locator("#health").click()
                page.wait_for_function("() => document.querySelector('#notice').textContent === 'HTTP 200'")
                check('"status": "ok"' in page.locator("#output").inner_text(), "authenticated browser health")
                page.locator("#prompt").fill("Return a harmless literal markup canary")
                page.locator("#send").click()
                page.wait_for_function("() => document.querySelector('#turn').value.length > 10 && document.querySelector('#output').textContent.includes('Native Rust console receipt')")
                first = json.loads(page.locator("#output").inner_text())
                first = first.get("turn", first)
                check(first["status"] == "completed", "browser chat actually completed")
                check(first.get("content", first.get("response")) == response, "provider response reached browser")
                check(page.locator("#output img").count() == 0 and page.evaluate("window.__missy_xss !== true"), "model HTML rendered as inert text")
                page.locator("#history").click()
                page.wait_for_function("() => document.querySelector('#output').textContent.includes('Native Rust console receipt')")
                check("Native Rust console receipt" in page.locator("#output").inner_text(), "persisted browser history")
                page.locator("#session").fill("different-session")
                page.locator("#status").click()
                page.wait_for_function("() => document.querySelector('#notice').textContent === 'HTTP 404'")
                check("Native Rust console receipt" not in page.locator("#output").inner_text(), "foreign session cannot read turn by ID")
                check(page.evaluate("localStorage.length === 0 && sessionStorage.length === 0"), "bearer token never persisted to browser storage")
                page.reload()
                check(page.locator("#notice").inner_text() == "Locked", "reload forgets bearer token")
                check(page.locator("#token").input_value() == "", "reload does not recover token")
                browser.close()
            check(len(requests) == 1, "UI actions did not replay inference")
            check(not (root / ".missy").exists(), "legacy state untouched")
            print(json.dumps({"status": "pass", "assertions": len(checks), "checks": checks}, indent=2))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
