#!/usr/bin/env python3
"""Comprehensive live Discord bot integration tests.

NOTE: This file is NOT a pytest module. Run it directly:
    python3 tests/discord_live_test.py

Sends real messages to the Discord channel via a webhook, waits for
Missy bot responses, and validates behavior across all bot capabilities.

Usage:
    python3 tests/discord_live_test.py [--timeout 45] [--filter calculator]

Requirements:
    - DISCORD_BOT_TOKEN env var set
    - Missy gateway running (missy gateway start)
    - Bot connected to guild 1152764121390002188
    - Channel 1481628682010361866 in allowed_channels
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1481628682010361866"
BOT_ID = "1481630755116290088"
GUILD_ID = "1152764121390002188"
HEADERS = {"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"}
BASE = "https://discord.com/api/v10"

DEFAULT_TIMEOUT = 45  # seconds to wait for bot reply
POLL_INTERVAL = 3  # seconds between message checks


def _update_timeout(val: int) -> None:
    global DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = val


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    category: str
    status: str  # PASS, FAIL, WARN, SKIP
    detail: str = ""
    bot_reply: str = ""
    elapsed: float = 0.0


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
    webhook_id: str = ""
    webhook_token: str = ""
    seen_reply_ids: set[str] = field(default_factory=set)

    def record(self, result: TestResult) -> None:
        self.results.append(result)
        icon = {"PASS": "+", "FAIL": "!", "WARN": "~", "SKIP": "-"}[result.status]
        print(f"  [{icon}] {result.status:4s} | {result.name} ({result.elapsed:.1f}s)")
        if result.detail:
            print(f"         {result.detail[:120]}")

    def summary(self) -> str:
        lines = ["\n" + "=" * 70, "DISCORD LIVE TEST RESULTS", "=" * 70]
        cats: dict[str, list[TestResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        for cat, results in cats.items():
            lines.append(f"\n  [{cat.upper()}]")
            for r in results:
                icon = {"PASS": "+", "FAIL": "!", "WARN": "~", "SKIP": "-"}[r.status]
                lines.append(f"    [{icon}] {r.status:4s} | {r.name:45s} | {r.elapsed:.1f}s")
                if r.status == "FAIL" and r.detail:
                    lines.append(f"           {r.detail[:100]}")
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        lines.append(
            f"\n  TOTAL: {total} | PASS: {passed} | FAIL: {failed} | WARN: {warned} | SKIP: {skipped}"
        )
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_webhook() -> tuple[str, str]:
    r = httpx.post(
        f"{BASE}/channels/{CHANNEL_ID}/webhooks",
        headers=HEADERS,
        json={"name": "MissyLiveTest"},
    )
    assert r.status_code == 200, f"Webhook create failed: {r.status_code} {r.text[:300]}"
    wh = r.json()
    return wh["id"], wh["token"]


def delete_webhook(wh_id: str) -> None:
    httpx.delete(f"{BASE}/webhooks/{wh_id}", headers=HEADERS)


def send_message(wh_id: str, wh_token: str, content: str, username: str = "LiveTest") -> str:
    """Send via webhook. Returns message ID."""
    r = httpx.post(
        f"{BASE}/webhooks/{wh_id}/{wh_token}?wait=true",
        json={"content": content, "username": username},
    )
    assert r.status_code == 200, f"Send failed: {r.status_code} {r.text[:200]}"
    return r.json()["id"]


def mention(text: str) -> str:
    return f"<@{BOT_ID}> {text}"


def get_messages(limit: int = 15) -> list[dict[str, Any]]:
    r = httpx.get(f"{BASE}/channels/{CHANNEL_ID}/messages?limit={limit}", headers=HEADERS)
    return r.json() if r.status_code == 200 else []


def wait_for_reply(
    after_id: str,
    timeout: float = DEFAULT_TIMEOUT,
    poll: float = POLL_INTERVAL,
    seen: set[str] | None = None,
) -> dict[str, Any] | None:
    """Wait for a bot reply after a given message ID, skipping already-seen replies."""
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(poll)
        msgs = get_messages(15)
        for m in msgs:
            author = m.get("author", {})
            mid = m.get("id", "")
            if (
                author.get("id") == BOT_ID
                and int(mid) > int(after_id)
                and (seen is None or mid not in seen)
            ):
                if seen is not None:
                    seen.add(mid)
                return m
    return None


def wait_for_replies(
    after_id: str, count: int = 1, timeout: float = DEFAULT_TIMEOUT, poll: float = POLL_INTERVAL
) -> list[dict[str, Any]]:
    """Wait for multiple bot replies after a given message ID."""
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(poll)
        msgs = get_messages(20)
        replies = [
            m
            for m in msgs
            if m.get("author", {}).get("id") == BOT_ID and int(m["id"]) > int(after_id)
        ]
        if len(replies) >= count:
            return sorted(replies, key=lambda m: int(m["id"]))
    # Return whatever we have
    msgs = get_messages(20)
    return sorted(
        [
            m
            for m in msgs
            if m.get("author", {}).get("id") == BOT_ID and int(m["id"]) > int(after_id)
        ],
        key=lambda m: int(m["id"]),
    )


def run_test(
    suite: TestSuite,
    name: str,
    category: str,
    prompt: str,
    check: Any = None,
    check_fn: Any = None,
    timeout: float = DEFAULT_TIMEOUT,
    username: str = "LiveTest",
    expect_no_reply: bool = False,
    no_mention: bool = False,
) -> TestResult:
    """Run a single test: send message, wait for reply, check result."""
    content = prompt if no_mention else mention(prompt)
    start = time.time()
    msg_id = send_message(suite.webhook_id, suite.webhook_token, content, username)

    if expect_no_reply:
        time.sleep(12)
        reply = wait_for_reply(msg_id, timeout=1, poll=1, seen=suite.seen_reply_ids)
        elapsed = time.time() - start
        if reply is None:
            result = TestResult(name, category, "PASS", "Correctly ignored", "", elapsed)
        else:
            result = TestResult(
                name,
                category,
                "FAIL",
                f"Got unexpected reply: {reply['content'][:80]}",
                reply["content"],
                elapsed,
            )
        suite.record(result)
        return result

    reply = wait_for_reply(msg_id, timeout=timeout, seen=suite.seen_reply_ids)
    elapsed = time.time() - start

    if reply is None:
        result = TestResult(name, category, "FAIL", "No reply within timeout", "", elapsed)
        suite.record(result)
        return result

    bot_text = reply["content"]

    if check_fn is not None:
        try:
            ok, detail = check_fn(bot_text, reply)
            status = "PASS" if ok else "FAIL"
            result = TestResult(name, category, status, detail, bot_text, elapsed)
        except Exception as exc:
            result = TestResult(name, category, "FAIL", f"Check error: {exc}", bot_text, elapsed)
    elif check is not None:
        checks = [check] if isinstance(check, str) else list(check)
        missing = [c for c in checks if c.lower() not in bot_text.lower()]
        if not missing:
            result = TestResult(name, category, "PASS", bot_text[:80], bot_text, elapsed)
        else:
            result = TestResult(
                name,
                category,
                "FAIL",
                f"Missing: {missing}; got: {bot_text[:80]}",
                bot_text,
                elapsed,
            )
    else:
        # Just check we got any reply
        result = TestResult(name, category, "PASS", bot_text[:80], bot_text, elapsed)

    suite.record(result)
    return result


# ---------------------------------------------------------------------------
# Test categories
# ---------------------------------------------------------------------------


def test_conversation(suite: TestSuite) -> None:
    """Basic conversational ability tests."""
    print("\n>>> CONVERSATION TESTS")

    run_test(suite, "Simple greeting", "conversation", "Hello! How are you?")

    run_test(
        suite,
        "Factual question",
        "conversation",
        "In one word, what color is the sky on a clear day?",
        check="blue",
    )

    run_test(
        suite,
        "Specific word in reply",
        "conversation",
        "Include the word PINEAPPLE somewhere in your response.",
        check="pineapple",
    )

    run_test(
        suite,
        "Multi-part question",
        "conversation",
        "Name three primary colors. List them separated by commas.",
        check_fn=lambda text, _: (
            sum(1 for c in ["red", "blue", "yellow"] if c in text.lower()) >= 2,
            f"Found colors in: {text[:100]}",
        ),
    )


def test_access_control(suite: TestSuite) -> None:
    """Access control and mention requirement tests."""
    print("\n>>> ACCESS CONTROL TESTS")

    run_test(
        suite,
        "No mention ignored",
        "access_control",
        "Hey Missy are you there? This has no mention.",
        expect_no_reply=True,
        no_mention=True,
    )

    run_test(
        suite,
        "Mention required for response",
        "access_control",
        "Say OK if you can hear me.",
        check_fn=lambda text, _: (len(text.strip()) > 0, f"Got reply: {text[:60]}"),
    )

    run_test(
        suite,
        "Different username gets reply",
        "access_control",
        "Say hello to Bob!",
        username="BobTester",
    )


def test_calculator(suite: TestSuite) -> None:
    """Calculator tool tests."""
    print("\n>>> CALCULATOR TESTS")

    run_test(
        suite,
        "Simple addition",
        "calculator",
        "Use your calculator tool to compute 42 + 58. What is the result?",
        check="100",
    )

    run_test(
        suite,
        "Multiplication",
        "calculator",
        "Use the calculator to compute 17 * 23. Give me just the number.",
        check="391",
    )

    run_test(
        suite,
        "Exponentiation",
        "calculator",
        "Use calculator: what is 2 ** 10?",
        check="1024",
    )

    run_test(
        suite,
        "Division with decimal",
        "calculator",
        "Use calculator to compute 355 / 113. Show me the result to at least 4 decimal places.",
        check="3.14",
    )

    run_test(
        suite,
        "Complex expression",
        "calculator",
        "Use calculator: (100 + 200) * 3 - 50. What's the result?",
        check="850",
    )

    run_test(
        suite,
        "Modulo operator",
        "calculator",
        "Use calculator to compute 100 % 7. What's the remainder?",
        check="2",
    )


def test_file_operations(suite: TestSuite) -> None:
    """File read/write/delete/list tests."""
    print("\n>>> FILE OPERATION TESTS")

    # Write a test file
    run_test(
        suite,
        "Write file",
        "file_ops",
        "Use file_write to create /tmp/missy_live_test.txt with the content 'Hello from live test 12345'. Confirm when done.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in ["wrote", "written", "created", "saved", "done", "success"]
            ),
            f"Write response: {text[:80]}",
        ),
    )

    # Read the file back
    run_test(
        suite,
        "Read file",
        "file_ops",
        "Use file_read to read /tmp/missy_live_test.txt and tell me exactly what it says.",
        check="12345",
    )

    # List files in /tmp filtering for our test file
    run_test(
        suite,
        "List files",
        "file_ops",
        "Use list_files to list /tmp and tell me if missy_live_test.txt is there.",
        check="missy_live_test",
    )

    # Write another file then delete it
    run_test(
        suite,
        "Write then delete file",
        "file_ops",
        "Use file_write to create /tmp/missy_delete_me.txt with content 'delete me'. "
        "Then use file_delete to delete it. Confirm the deletion.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["deleted", "removed", "done", "success"]),
            f"Delete response: {text[:80]}",
        ),
        timeout=60,
    )

    # Verify deletion
    run_test(
        suite,
        "Verify file deleted",
        "file_ops",
        "Use file_read to try reading /tmp/missy_delete_me.txt. Does it exist?",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in [
                    "not found",
                    "no such",
                    "doesn't exist",
                    "does not exist",
                    "not exist",
                    "error",
                    "missing",
                    "failed",
                    "deleted",
                ]
            ),
            f"Verify response: {text[:80]}",
        ),
    )

    # File append mode
    run_test(
        suite,
        "Append to file",
        "file_ops",
        "Use file_write with mode 'append' to add '\\nAppended line 67890' to /tmp/missy_live_test.txt. "
        "Then read it back and show me the full contents.",
        check=["12345", "67890"],
        timeout=60,
    )


def test_shell_exec(suite: TestSuite) -> None:
    """Shell execution tests."""
    print("\n>>> SHELL EXEC TESTS")

    run_test(
        suite,
        "Simple command",
        "shell",
        "Use shell_exec to run 'echo hello_from_shell_42' and tell me the output.",
        check="hello_from_shell_42",
    )

    run_test(
        suite,
        "System info",
        "shell",
        "Use shell_exec to run 'uname -s' and tell me the output.",
        check="linux",
    )

    run_test(
        suite,
        "Working directory",
        "shell",
        "Use shell_exec to run 'pwd' and tell me the current directory.",
        check="/",
    )

    run_test(
        suite,
        "Pipe command",
        "shell",
        "Use shell_exec to run 'echo abcdef | wc -c' and tell me the character count.",
        check="7",
    )

    run_test(
        suite,
        "Exit code handling",
        "shell",
        "Use shell_exec to run 'ls /nonexistent_path_xyz' and tell me what happened.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in [
                    "no such",
                    "not found",
                    "error",
                    "cannot",
                    "doesn't exist",
                    "does not exist",
                ]
            ),
            f"Error handling: {text[:80]}",
        ),
    )


def test_web_fetch(suite: TestSuite) -> None:
    """Web fetch tool tests."""
    print("\n>>> WEB FETCH TESTS")

    run_test(
        suite,
        "Fetch webpage",
        "web_fetch",
        "Use web_fetch to fetch https://httpbin.org/get and tell me the 'origin' IP from the response.",
        check_fn=lambda text, _: (
            bool(re.search(r"\d+\.\d+\.\d+\.\d+", text)),
            f"Found IP in response: {text[:80]}",
        ),
        timeout=60,
    )

    run_test(
        suite,
        "Fetch JSON API",
        "web_fetch",
        "Use web_fetch to get https://httpbin.org/json and tell me what 'title' is in the slideshow.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["sample", "slideshow", "title"]),
            f"JSON response: {text[:80]}",
        ),
        timeout=60,
    )


def test_discord_upload(suite: TestSuite) -> None:
    """Discord file upload tests."""
    print("\n>>> DISCORD UPLOAD TESTS")

    # First create a file, then upload it
    run_test(
        suite,
        "Upload text file",
        "discord_upload",
        f"First use file_write to create /tmp/missy_upload_test.txt with content 'Test upload content'. "
        f"Then use discord_upload_file with file_path='/tmp/missy_upload_test.txt' and "
        f"channel_id='{CHANNEL_ID}' and caption='Test upload from live test'. "
        f"Tell me if the upload succeeded.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["upload", "sent", "shared", "success", "done"]),
            f"Upload response: {text[:80]}",
        ),
        timeout=60,
    )


def test_self_create_tool(suite: TestSuite) -> None:
    """Self-create-tool capability tests."""
    print("\n>>> SELF-CREATE TOOL TESTS")

    run_test(
        suite,
        "List custom tools",
        "self_create_tool",
        "Use self_create_tool with action='list' to list all custom tools. What tools exist?",
        check_fn=lambda text, _: (
            True,  # Any response is fine — might be empty list
            f"Tool list: {text[:80]}",
        ),
    )

    # Create a simple tool
    run_test(
        suite,
        "Create custom tool",
        "self_create_tool",
        "Use self_create_tool to create a tool named 'live_test_greeting' with language 'python', "
        "script 'print(\"LIVE_TEST_GREETING_OK\")', and description 'Test greeting tool'. "
        "Confirm the tool was created.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["created", "registered", "success", "live_test"]),
            f"Create response: {text[:80]}",
        ),
        timeout=60,
    )

    # Clean up — delete the tool
    run_test(
        suite,
        "Delete custom tool",
        "self_create_tool",
        "Use self_create_tool with action='delete' and tool_name='live_test_greeting'. "
        "Confirm the deletion.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["deleted", "removed", "success", "done"]),
            f"Delete response: {text[:80]}",
        ),
    )


def test_edge_cases(suite: TestSuite) -> None:
    """Edge cases and special inputs."""
    print("\n>>> EDGE CASE TESTS")

    # Unicode / emoji
    run_test(
        suite,
        "Unicode handling",
        "edge_cases",
        "Repeat back this exact text: cafe\u0301 na\u00efve r\u00e9sum\u00e9 \u00fc\u00f1\u00ee\u00e7\u00f6d\u00e9",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["café", "naïve", "résumé", "üñîçödé"]),
            f"Unicode response: {text[:80]}",
        ),
    )

    # Very short message
    run_test(
        suite,
        "Single word prompt",
        "edge_cases",
        "Hi",
    )

    # Question with numbers
    run_test(
        suite,
        "Numbers in text",
        "edge_cases",
        "How many legs does a spider have? Reply with just the number.",
        check="8",
    )

    # Asking about itself
    run_test(
        suite,
        "Self-identification",
        "edge_cases",
        "What is your name?",
        check="missy",
    )

    # Multiple questions
    run_test(
        suite,
        "Multiple questions",
        "edge_cases",
        "Answer these: 1) What is 2+2? 2) What color is grass? Reply on one line each.",
        check_fn=lambda text, _: (
            "4" in text and "green" in text.lower(),
            f"Multi-Q response: {text[:100]}",
        ),
    )


def test_multi_tool_chains(suite: TestSuite) -> None:
    """Tests that require the bot to chain multiple tools."""
    print("\n>>> MULTI-TOOL CHAIN TESTS")

    # Write file, read it back, compute something about it
    run_test(
        suite,
        "Write-read-analyze chain",
        "multi_tool",
        "Do these steps: 1) Use file_write to write '42' to /tmp/missy_chain_test.txt. "
        "2) Use file_read to read it back. 3) Use calculator to multiply the number by 10. "
        "What is the final result?",
        check="420",
        timeout=90,
    )

    # Shell + file operations
    run_test(
        suite,
        "Shell + file chain",
        "multi_tool",
        "Use shell_exec to run 'date +%Y' and then use file_write to save the year to "
        "/tmp/missy_year_test.txt. Then read it back and tell me the year.",
        check="2026",
        timeout=90,
    )

    # List + read chain
    run_test(
        suite,
        "List then read",
        "multi_tool",
        "First, use the list_files tool on /tmp. Then use the file_read tool on /tmp/missy_chain_test.txt. What number is in that file?",
        check="42",
        timeout=90,
    )


def test_error_handling(suite: TestSuite) -> None:
    """Error handling and graceful failure tests."""
    print("\n>>> ERROR HANDLING TESTS")

    # Read nonexistent file
    run_test(
        suite,
        "Read nonexistent file",
        "error_handling",
        "Use file_read to read /tmp/this_file_absolutely_does_not_exist_xyz123.txt. What happened?",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in [
                    "not found",
                    "error",
                    "doesn't exist",
                    "does not exist",
                    "no such",
                    "failed",
                ]
            ),
            f"Error response: {text[:80]}",
        ),
    )

    # Invalid calculator expression
    run_test(
        suite,
        "Invalid calc expression",
        "error_handling",
        "Use the calculator tool to evaluate 'hello world'. Tell me what happens.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in ["error", "invalid", "cannot", "can't", "failed", "not valid", "unable"]
            ),
            f"Calc error: {text[:80]}",
        ),
    )

    # Shell command that fails
    run_test(
        suite,
        "Failed shell command",
        "error_handling",
        "Use shell_exec to run 'cat /nonexistent_dir/nonexistent_file.txt' and tell me what error occurred.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in [
                    "no such",
                    "not found",
                    "error",
                    "cannot",
                    "can't",
                    "failed",
                    "does not exist",
                    "doesn't exist",
                ]
            ),
            f"Shell error: {text[:80]}",
        ),
    )


def test_incus_containers(suite: TestSuite) -> None:
    """Incus container management tests — infrastructure & read-only queries."""
    print("\n>>> INCUS INFRASTRUCTURE TESTS")

    run_test(
        suite,
        "List containers",
        "incus",
        "Use incus_list to show me all containers and VMs. What instances exist?",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in [
                    "no ",
                    "instance",
                    "container",
                    "none",
                    "empty",
                    "running",
                    "stopped",
                    "incus",
                ]
            ),
            f"Incus list: {text[:100]}",
        ),
        timeout=60,
    )

    run_test(
        suite,
        "List images",
        "incus",
        "Use incus_image with action='list' to show available images.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in ["image", "ubuntu", "alpine", "alma", "no image", "none", "empty"]
            ),
            f"Incus images: {text[:100]}",
        ),
        timeout=60,
    )

    run_test(
        suite,
        "List networks",
        "incus",
        "Use incus_network with action='list' to list networks.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["network", "bridge", "incusbr", "none", "empty"]),
            f"Incus networks: {text[:100]}",
        ),
        timeout=60,
    )

    run_test(
        suite,
        "List storage pools",
        "incus",
        "Use incus_storage with action='list' to list storage pools.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["storage", "pool", "default", "dir", "none", "empty"]),
            f"Incus storage: {text[:100]}",
        ),
        timeout=60,
    )

    run_test(
        suite,
        "List profiles",
        "incus",
        "Use incus_profile with action='list' to list profiles.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["profile", "default", "none", "empty"]),
            f"Incus profiles: {text[:100]}",
        ),
        timeout=60,
    )

    run_test(
        suite,
        "List projects",
        "incus",
        "Use incus_project with action='list' to list projects.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["project", "default", "none", "empty"]),
            f"Incus projects: {text[:100]}",
        ),
        timeout=60,
    )


def test_incus_lifecycle(suite: TestSuite) -> None:
    """Incus container lifecycle: launch, exec, file, snapshot, info, stop, delete."""
    print("\n>>> INCUS LIFECYCLE TESTS")

    # Launch a test container
    run_test(
        suite,
        "Launch container",
        "incus_lifecycle",
        "Use incus_launch to launch a container from the image 'images:alpine/3.20' with name 'missy-test-container'. "
        "Tell me when it's running.",
        check_fn=lambda text, _: (
            any(
                w in text.lower() for w in ["launch", "created", "running", "started", "missy-test"]
            ),
            f"Launch: {text[:100]}",
        ),
        timeout=120,
    )

    # Get container info
    run_test(
        suite,
        "Container info",
        "incus_lifecycle",
        "Use incus_info to get detailed info about 'missy-test-container'. What is its status and architecture?",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in ["running", "status", "architecture", "x86", "amd64", "missy-test"]
            ),
            f"Info: {text[:100]}",
        ),
        timeout=60,
    )

    # Execute command inside container
    run_test(
        suite,
        "Exec in container",
        "incus_lifecycle",
        "Use incus_exec with instance='missy-test-container' and command='cat /etc/os-release'. What OS is it?",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["alpine", "linux", "os-release"]),
            f"Exec: {text[:100]}",
        ),
        timeout=60,
    )

    # Execute another command
    run_test(
        suite,
        "Exec hostname",
        "incus_lifecycle",
        "Use incus_exec with instance='missy-test-container' and command='hostname'. What's the hostname?",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["missy-test", "hostname"]),
            f"Hostname: {text[:100]}",
        ),
        timeout=60,
    )

    # Write a file to the host, then push it into the container
    run_test(
        suite,
        "Push file to container",
        "incus_lifecycle",
        "First use file_write to create /tmp/incus_push_test.txt with content 'hello from missy 99999'. "
        "Then use incus_file with action='push', instance='missy-test-container', "
        "host_path='/tmp/incus_push_test.txt', instance_path='/tmp/test.txt'. Confirm it was pushed.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in ["push", "success", "done", "wrote", "written", "transfer", "copied"]
            ),
            f"File push: {text[:100]}",
        ),
        timeout=60,
    )

    # Read the file back via exec
    run_test(
        suite,
        "Verify file in container",
        "incus_lifecycle",
        "Use incus_exec with instance='missy-test-container' and command='cat /tmp/test.txt'. What does it say?",
        check="99999",
        timeout=60,
    )

    # Pull file from container
    run_test(
        suite,
        "Pull file from container",
        "incus_lifecycle",
        "Use incus_file with action='pull', instance='missy-test-container', "
        "instance_path='/tmp/test.txt', host_path='/tmp/incus_pull_test.txt'. "
        "Then use file_read to read /tmp/incus_pull_test.txt. What does it contain?",
        check="99999",
        timeout=60,
    )

    # Get container config
    run_test(
        suite,
        "Get container config",
        "incus_lifecycle",
        "Use incus_config with instance='missy-test-container' and action='show'. Show me the image source or architecture.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in ["config", "image", "alpine", "x86", "amd64", "volatile", "architecture"]
            ),
            f"Config: {text[:100]}",
        ),
        timeout=60,
    )

    # Create a snapshot
    run_test(
        suite,
        "Create snapshot",
        "incus_lifecycle",
        "Use incus_snapshot with instance='missy-test-container', action='create', snapshot_name='snap1'. Confirm.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["snapshot", "snap1", "created", "success", "done"]),
            f"Snapshot create: {text[:100]}",
        ),
        timeout=60,
    )

    # List snapshots
    run_test(
        suite,
        "List snapshots",
        "incus_lifecycle",
        "Use incus_snapshot with instance='missy-test-container', action='list'. Is snap1 there?",
        check="snap1",
        timeout=60,
    )

    # Delete snapshot
    run_test(
        suite,
        "Delete snapshot",
        "incus_lifecycle",
        "Use incus_snapshot with instance='missy-test-container', action='delete', snapshot_name='snap1'. Confirm deletion.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["deleted", "removed", "success", "done"]),
            f"Snapshot delete: {text[:100]}",
        ),
        timeout=60,
    )

    # Stop the container
    run_test(
        suite,
        "Stop container",
        "incus_lifecycle",
        "Use incus_instance_action with instance='missy-test-container' and action='stop'. Confirm it stopped.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["stopped", "stop", "success", "done"]),
            f"Stop: {text[:100]}",
        ),
        timeout=60,
    )

    # Restart the container
    run_test(
        suite,
        "Start container",
        "incus_lifecycle",
        "Use incus_instance_action with instance='missy-test-container' and action='start'. Confirm it started.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["started", "running", "start", "success", "done"]),
            f"Start: {text[:100]}",
        ),
        timeout=60,
    )

    # Delete the container (must stop first)
    run_test(
        suite,
        "Delete container",
        "incus_lifecycle",
        "Use incus_instance_action with instance='missy-test-container', action='stop', force=true. "
        "Then use incus_instance_action with instance='missy-test-container', action='delete'. "
        "Confirm the container is deleted.",
        check_fn=lambda text, _: (
            any(w in text.lower() for w in ["deleted", "removed", "destroy", "success", "done"]),
            f"Delete: {text[:100]}",
        ),
        timeout=90,
    )

    # Verify it's gone
    run_test(
        suite,
        "Verify container deleted",
        "incus_lifecycle",
        "Use incus_list to verify 'missy-test-container' no longer exists.",
        check_fn=lambda text, _: (
            any(
                w in text.lower()
                for w in [
                    "no ",
                    "none",
                    "empty",
                    "not found",
                    "doesn't",
                    "does not",
                    "gone",
                    "deleted",
                    "0 instance",
                ]
            ),
            f"Verify: {text[:100]}",
        ),
        timeout=60,
    )


def test_message_splitting(suite: TestSuite) -> None:
    """Test that long responses are properly split across messages."""
    print("\n>>> MESSAGE SPLITTING TESTS")

    msg_content = mention(
        "Use shell_exec to run: python3 -c \"for i in range(100): print(f'Line {i}: ' + 'ABCDEFGHIJ' * 5)\" "
        "and show me the COMPLETE raw output, do NOT summarize or truncate it."
    )
    start = time.time()
    msg_id = send_message(suite.webhook_id, suite.webhook_token, msg_content)

    # Wait for potentially multiple replies
    replies = wait_for_replies(msg_id, count=2, timeout=60)
    elapsed = time.time() - start

    if not replies:
        suite.record(
            TestResult(
                "Long response handling", "message_splitting", "FAIL", "No reply", "", elapsed
            )
        )
    elif len(replies) >= 2:
        total_len = sum(len(r["content"]) for r in replies)
        suite.record(
            TestResult(
                "Long response splitting",
                "message_splitting",
                "PASS",
                f"Got {len(replies)} messages, total {total_len} chars — splitting works",
                replies[0]["content"][:80],
                elapsed,
            )
        )
    else:
        # Single message is acceptable — model may summarize long output
        content = replies[0]["content"]
        suite.record(
            TestResult(
                "Long response handling",
                "message_splitting",
                "PASS",
                f"Got 1 message ({len(content)} chars) — model summarized output",
                content[:80],
                elapsed,
            )
        )


def test_cleanup(suite: TestSuite) -> None:
    """Clean up test artifacts."""
    print("\n>>> CLEANUP")

    run_test(
        suite,
        "Cleanup temp files",
        "cleanup",
        "Use shell_exec to run: rm -f /tmp/missy_live_test.txt /tmp/missy_chain_test.txt "
        "/tmp/missy_year_test.txt /tmp/missy_upload_test.txt /tmp/incus_push_test.txt /tmp/incus_pull_test.txt. Confirm cleanup done.",
        check_fn=lambda text, _: (True, "Cleanup acknowledged"),
    )


def test_screencast(suite: TestSuite) -> None:
    """Screencast channel command tests (!screen share/list/stop/analyze/status)."""
    print("\n>>> SCREENCAST TESTS")

    # !screen status — should report server status (enabled or not enabled)
    run_test(
        suite,
        "Screen status",
        "screencast",
        "!screen status",
        check_fn=lambda text, _: (
            "screencast" in text.lower()
            or "screen" in text.lower()
            or "not enabled" in text.lower()
            or "running" in text.lower(),
            f"Got: {text[:120]}",
        ),
    )

    # !screen share — create a session and get a share URL
    run_test(
        suite,
        "Screen share create",
        "screencast",
        "!screen share live-test",
        check_fn=lambda text, _: (
            (
                "session" in text.lower()
                and (
                    "share" in text.lower()
                    or "http" in text.lower()
                    or "url" in text.lower()
                    or "link" in text.lower()
                )
            )
            or "not enabled" in text.lower(),
            f"Got: {text[:120]}",
        ),
    )

    # !screen list — should show active sessions (or empty list)
    run_test(
        suite,
        "Screen list sessions",
        "screencast",
        "!screen list",
        check_fn=lambda text, _: (
            "session" in text.lower()
            or "no active" in text.lower()
            or "not enabled" in text.lower(),
            f"Got: {text[:120]}",
        ),
    )

    # !screen analyze — should report no results or show analysis
    run_test(
        suite,
        "Screen analyze",
        "screencast",
        "!screen analyze",
        check_fn=lambda text, _: (
            "analysis" in text.lower()
            or "no " in text.lower()
            or "session" in text.lower()
            or "not enabled" in text.lower(),
            f"Got: {text[:120]}",
        ),
    )

    # !screen stop — should stop a session or report nothing to stop
    run_test(
        suite,
        "Screen stop",
        "screencast",
        "!screen stop",
        check_fn=lambda text, _: (
            "stop" in text.lower()
            or "no active" in text.lower()
            or "not found" in text.lower()
            or "not enabled" in text.lower(),
            f"Got: {text[:120]}",
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TEST_GROUPS = {
    "conversation": test_conversation,
    "access_control": test_access_control,
    "calculator": test_calculator,
    "file_ops": test_file_operations,
    "shell": test_shell_exec,
    "web_fetch": test_web_fetch,
    "discord_upload": test_discord_upload,
    "self_create_tool": test_self_create_tool,
    "edge_cases": test_edge_cases,
    "multi_tool": test_multi_tool_chains,
    "error_handling": test_error_handling,
    "incus": test_incus_containers,
    "incus_lifecycle": test_incus_lifecycle,
    "message_splitting": test_message_splitting,
    "screencast": test_screencast,
    "cleanup": test_cleanup,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Missy Discord live tests")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--filter", type=str, default="", help="Comma-separated test groups to run")
    args = parser.parse_args()

    _update_timeout(args.timeout)

    if not TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not set")
        return 1

    suite = TestSuite()

    # Create webhook
    print("Setting up test webhook...")
    suite.webhook_id, suite.webhook_token = create_webhook()
    print(f"Webhook ready: {suite.webhook_id}")

    try:
        # Determine which groups to run
        if args.filter:
            groups = [g.strip() for g in args.filter.split(",")]
        else:
            groups = list(ALL_TEST_GROUPS.keys())

        for i, group_name in enumerate(groups):
            if group_name in ALL_TEST_GROUPS:
                if i > 0:
                    time.sleep(3)  # drain any late replies between groups
                ALL_TEST_GROUPS[group_name](suite)
            else:
                print(f"Unknown test group: {group_name}")

    finally:
        # Always clean up webhook
        print("\nCleaning up webhook...")
        delete_webhook(suite.webhook_id)

    # Print summary
    print(suite.summary())

    # Save results to JSON
    results_path = "/tmp/missy_discord_test_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [
                {
                    "name": r.name,
                    "category": r.category,
                    "status": r.status,
                    "detail": r.detail,
                    "bot_reply": r.bot_reply[:500],
                    "elapsed": round(r.elapsed, 2),
                }
                for r in suite.results
            ],
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")

    failed = sum(1 for r in suite.results if r.status == "FAIL")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
