"""Headless Chromium only: replaces display capture with a synthetic canvas.

No physical display, microphone, camera, account, extension, or desktop is used.
Input pairing is supplied through stdin, never through process arguments/URLs.
"""
import asyncio
import json
import sys

from playwright.async_api import async_playwright


async def main():
    payload = json.load(sys.stdin)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        await context.add_init_script("""(() => {
          window.fixtureCaptureCalls = 0;
          window.fixtureStreams = [];
          const canvas = document.createElement('canvas');
          canvas.width = canvas.height = 64;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = '#587acd'; ctx.fillRect(0,0,64,64);
          Object.defineProperty(navigator.mediaDevices, 'getDisplayMedia', {
            value: async () => {
              window.fixtureCaptureCalls++;
              const stream = canvas.captureStream(1);
              const repaint = setInterval(() => {
                if (stream.getVideoTracks()[0].readyState === 'ended') clearInterval(repaint);
                else ctx.fillRect(0,0,64,64);
              }, 50);
              window.fixtureStreams.push(stream);
              return stream;
            }
          });
        })();""")
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(payload["url"])
        assert await page.evaluate("window.fixtureCaptureCalls") == 0
        assert await page.evaluate("localStorage.length + sessionStorage.length") == 0
        assert "token" not in page.url
        await page.locator("#pairing").fill(json.dumps(payload["pairing"]))
        await page.locator("#start").click()
        for _ in range(100):
            if "analyzed" in await page.locator("#status").inner_text():
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError(await page.locator("#status").inner_text())
        assert await page.evaluate("window.fixtureCaptureCalls") == 1
        assert await page.locator("#pairing").input_value() == ""
        assert await page.evaluate("localStorage.length + sessionStorage.length") == 0
        await page.locator("#stop").click()
        assert await page.evaluate("window.fixtureStreams.every(s => s.getTracks().every(t => t.readyState === 'ended'))")
        assert await page.locator("#start").is_enabled()
        assert await page.locator("#stop").is_disabled()
        assert await page.evaluate("document.querySelector('#preview').srcObject === null")
        assert not errors, errors
        await browser.close()
    print("headless Chromium fixture: explicit click, synthetic frame, no ambient capture, no storage, stop cleanup PASS")


asyncio.run(main())
