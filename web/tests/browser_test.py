"""Math, independent Python-fixture parity, and real-worker browser tests.

Run from any directory: python3 web/tests/browser_test.py
Requires Playwright and either installed Google Chrome or Playwright Chromium.
An ephemeral HTTP server is started unless --url is supplied.
"""
import argparse
import functools
import http.server
from pathlib import Path
import re
import tempfile
import threading

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="An already running prototype URL")
    parser.add_argument("--artifacts", default=str(Path(tempfile.gettempdir()) / "rl-litmus-artifacts"))
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    server = None
    url = args.url
    if not url:
        # Serve beneath a /web/ prefix to exercise GitHub project-site paths.
        repo = Path(__file__).resolve().parents[2]
        handler = functools.partial(QuietHandler, directory=str(repo))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/web/"

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="chrome", headless=True)
            except Exception:
                browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
            page = context.new_page()
            # Observe the actual worker boundary without adding test-only exports
            # to the demo. This also distinguishes worker progress from the
            # historical step selected in the figure.
            page.add_init_script("""(() => {
              const NativeWorker = window.Worker;
              window.__trainingWorkers = [];
              window.Worker = class extends NativeWorker {
                constructor(url, options) {
                  super(url, options);
                  if (!new URL(url, location.href).pathname.endsWith('/worker.js')) return;
                  const record = {
                    config: null, iterations: null, history: [], messages: [], extensions: [], complete: false,
                    initialView: {
                      value: Number(document.getElementById('timeline').value),
                      max: Number(document.getElementById('timeline').max),
                      step: document.getElementById('step-label').textContent,
                    },
                  };
                  window.__trainingWorkers.push(record);
                  const nativePostMessage = this.postMessage.bind(this);
                  this.postMessage = (data, ...rest) => {
                    record.messages.push(data.type);
                    if (data.type === 'init') {
                      record.config = structuredClone(data.config);
                      record.iterations = data.config.iterations;
                    }
                    return nativePostMessage(data, ...rest);
                  };
                  this.addEventListener('message', ({data}) => {
                    if (data.type === 'ready') record.history = [data.snapshot];
                    if (data.type === 'extended') {
                      record.iterations = data.iterations;
                      record.extensions.push(data.iterations);
                      record.complete = false;
                    }
                    if (data.type === 'update') {
                      record.history.push(...data.history);
                      record.complete = data.complete;
                    }
                  });
                }
              };
            })();""")
            errors = []
            failed_requests = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: failed_requests.append(request.url))
            page.goto(url)

            def state(label):
                page.wait_for_function(
                    "expected => document.getElementById('run-status').textContent === expected",
                    arg=label, timeout=180000,
                )

            def current_return():
                text = page.locator("#current-return").inner_text().replace("−", "-").replace(",", "")
                match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
                assert match, f"Missing expected return: {text}"
                return float(match.group())

            def worker_value(expression):
                return page.evaluate(f"() => {{ const record = window.__trainingWorkers.at(-1); return {expression}; }}")

            def move_slider(selector, value):
                page.locator(selector).evaluate("""(input, value) => {
                  input.value = String(value);
                  input.dispatchEvent(new Event('input', {bubbles: true}));
                }""", value)

            def canvas_image(selector):
                # Let a queued draw and its resulting paint complete.
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                return page.locator(selector).evaluate("canvas => canvas.toDataURL()")

            def step_is(iteration, total=10000):
                page.wait_for_function("""({expected, total}) => {
                  const input = document.getElementById('timeline');
                  return Number(input.value) === expected &&
                    document.getElementById('step-label').textContent ===
                      `Step ${expected.toLocaleString('en-US')} / ${total.toLocaleString('en-US')}`;
                }""", arg={"expected": iteration, "total": total})

            state("Ready to train")
            math = page.evaluate("async () => (await import('./tests/engine.test.js')).runTests()")
            assert math["failed"] == 0, math
            print(f"PASS: {math['passed']} mathematical test groups")

            parity = page.evaluate("""async () => {
              const {transition, createExperiment} = await import('./engine.js');
              const fixtures = await (await fetch('./tests/reference.json')).json();
              let count = 0;
              function check(actual, expected, keys, name) {
                for (const key of keys) {
                  if (!Number.isFinite(actual[key]) || Math.abs(actual[key] - expected[key]) > 1e-9)
                    throw Error(`${name}.${key}: ${actual[key]} != ${expected[key]}`);
                }
                count++;
              }
              for (const f of fixtures.transitions) {
                const a=f.input;
                check(transition(a.buffer,a.action,a.bandwidth,a.config), f.expected,
                  ['buffer','reward','stall','bitrate','download'], f.name);
              }
              for (const f of fixtures.policies)
                check(createExperiment(f.config).evaluate(f.params.w,f.params.b),f.expected,
                  ['return','gradW','gradB','bitrate','stall'], f.name);
              for (const f of fixtures.baselines)
                check(createExperiment(f.config).baseline,f.expected,
                  ['return','bitrate','stall'], f.name);
              return count;
            }""")
            print(f"PASS: {parity} independent Python reference cases")

            config = worker_value("record.config")
            assert {key: config[key] for key in ("iterations", "learningRate", "batchSize", "seed", "lambda", "horizon")} == {
                "iterations": 10000, "learningRate": 0.05, "batchSize": 16,
                "seed": 42, "lambda": 4.3, "horizon": 8,
            }
            assert config["bandwidths"] == [3, 3.5, 4, 4.5, 5, 5.5, 6]
            assert page.get_by_text("FIGURE 2 · INTERACTIVE", exact=True).count() == 0
            assert not page.locator("#method-details").evaluate("details => details.open")
            assert page.locator("#train-button").inner_text().split() == ["▶", "Train"]
            assert page.locator("#penalty").get_attribute("min") == "0"
            assert page.locator("#penalty").get_attribute("max") == "10"
            assert float(page.locator("#penalty-value").inner_text()) == 4.3
            assert page.locator("#learning-rate").get_attribute("type") == "range"
            assert page.locator("#learning-rate").get_attribute("min") == "0.001"
            assert page.locator("#learning-rate").get_attribute("max") == "0.2"
            assert page.locator("#learning-rate").get_attribute("step") == "0.001"
            assert float(page.locator("#learning-rate").input_value()) == 0.05
            assert float(page.locator("#learning-rate-value").inner_text()) == 0.05
            assert page.locator("#reward-formula").is_visible()
            assert "bitrate / 4.3" in page.locator("#reward-formula").inner_text()
            assert "stall seconds / 4" in page.locator("#reward-formula").inner_text()
            step_is(0)
            initial = current_return()
            assert abs(initial - worker_value("record.history[0].return")) < 0.0051
            initial_policy = canvas_image("#policy-chart")
            initial_landscape = canvas_image("#landscape-chart")

            page.locator("#train-button").click()
            state("Training")
            assert page.locator("#train-button").inner_text().split() == ["Ⅱ", "Pause"]
            page.wait_for_function("window.__trainingWorkers.at(-1).history.length > 8")
            # Scrubbing a live run must finish pausing before it selects the
            # requested step. A late worker update must not move the slider.
            move_slider("#timeline", 0)
            state("Paused")
            step_is(0)
            assert current_return() == initial
            paused_iteration = worker_value("record.history.at(-1).iteration")
            page.wait_for_timeout(180)
            assert worker_value("record.history.at(-1).iteration") == paused_iteration
            step_is(0)
            assert canvas_image("#policy-chart") == initial_policy

            page.locator("#train-button").click()
            state("Training")
            page.wait_for_function(
                "previous => window.__trainingWorkers.at(-1).history.at(-1).iteration > previous",
                arg=paused_iteration,
            )
            page.locator("#train-button").click()
            state("Paused")
            assert page.locator("#train-button").inner_text().split() == ["▶", "Resume"]
            paused_iteration = worker_value("record.history.at(-1).iteration")
            page.wait_for_timeout(180)
            assert worker_value("record.history.at(-1).iteration") == paused_iteration
            page.locator("#train-button").click()
            state("Complete")
            step_is(10000)
            assert page.locator("#train-button").inner_text().split() == ["▶", "Train", "again"]
            assert worker_value("record.complete && record.history.length === 10001")
            assert worker_value("record.history.every((point, index) => point.iteration === index)")
            page.wait_for_timeout(240)
            assert worker_value("record.complete && record.history.length === 10001 && record.history.at(-1).iteration === 10000")
            assert worker_value("record.extensions") == []
            step_is(10000)
            final_return = current_return()
            assert final_return > initial
            assert abs(final_return - worker_value("record.history.at(-1).return")) < 0.0051
            assert page.locator("#policy-summary").inner_text().strip()
            final_policy = canvas_image("#policy-chart")
            assert final_policy != initial_policy
            assert canvas_image("#landscape-chart") != initial_landscape
            page.evaluate("window.__firstHistory = JSON.stringify(window.__trainingWorkers.at(-1).history)")
            print("PASS: full 10,000-step worker training, pause/resume, live scrub pause, and linked figure updates", flush=True)

            move_slider("#timeline", 0)
            step_is(0)
            assert current_return() == initial
            assert canvas_image("#policy-chart") == initial_policy
            move_slider("#timeline", 10000)
            step_is(10000)
            assert current_return() == final_return
            assert canvas_image("#policy-chart") == final_policy
            print("PASS: timeline restores the initial and final policy and exact expected return")

            page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(150)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile horizontal overflow"
            for selector in ("#landscape-chart", "#policy-chart"):
                box = page.locator(selector).bounding_box()
                assert box and box["width"] > 200 and box["x"] + box["width"] <= 391, box
            page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
            page.locator("#method-details summary").click()
            assert page.locator("#method-details").evaluate("details => details.open")
            page.locator("#method-details summary").click()
            assert not page.locator("#method-details").evaluate("details => details.open")
            page.set_viewport_size({"width": 1440, "height": 1100})
            print("PASS: both plots fit mobile and method details stay optional")

            worker_count = page.evaluate("window.__trainingWorkers.length")
            page.locator("#train-button").click()
            state("Training")
            page.wait_for_function("window.__trainingWorkers.at(-1).history.at(-1).iteration > 100")
            page.locator("#train-button").click()
            state("Paused")
            replayed_iteration = worker_value("record.history.at(-1).iteration")
            assert page.evaluate("window.__trainingWorkers.length") == worker_count + 1, "Train again did not create a fresh worker"
            assert worker_value("record.config.iterations") == 10000
            assert worker_value("record.iterations") == 10000
            assert worker_value("record.extensions") == []
            assert worker_value("record.initialView") == {"value": 0, "max": 0, "step": "Step 0 / 10,000"}
            assert worker_value("!record.complete && record.history.length < 10001 && record.history.every((point, index) => point.iteration === index)")
            assert page.evaluate("""() => {
              const history = window.__trainingWorkers.at(-1).history;
              return JSON.stringify(history) === JSON.stringify(JSON.parse(window.__firstHistory).slice(0, history.length));
            }"""), "Train again did not replay the same seeded trajectory"
            step_is(replayed_iteration)
            replayed_return = current_return()
            replayed_policy = canvas_image("#policy-chart")
            page.wait_for_timeout(180)
            assert worker_value("record.history.at(-1).iteration") == replayed_iteration
            move_slider("#timeline", 0)
            step_is(0)
            assert current_return() == initial
            assert canvas_image("#policy-chart") == initial_policy
            move_slider("#timeline", replayed_iteration)
            step_is(replayed_iteration)
            assert current_return() == replayed_return
            assert canvas_image("#policy-chart") == replayed_policy
            print("PASS: training stops at 10,000 and Train again starts a fresh seeded run with a reset timeline", flush=True)

            page.locator("#reset-button").click()
            state("Ready to train")
            step_is(0)
            assert current_return() == initial
            assert worker_value("record.history.length") == 1
            assert page.evaluate("window.__trainingWorkers.length") == worker_count + 2
            page.locator("#train-button").click()
            state("Training")
            page.wait_for_function("window.__trainingWorkers.at(-1).history.at(-1).iteration > 100")
            page.locator("#train-button").click()
            state("Paused")
            assert page.evaluate("""() => {
              const history = window.__trainingWorkers.at(-1).history;
              return JSON.stringify(history) === JSON.stringify(JSON.parse(window.__firstHistory).slice(0, history.length));
            }"""), "Explicit Reset did not replay the same seeded trajectory"
            print("PASS: explicit Reset restores the 10,000-step budget and seeded starting trajectory", flush=True)

            page.locator("#reset-button").click()
            state("Ready to train")
            step_is(0)
            map_before_penalty = canvas_image("#landscape-chart")
            move_slider("#penalty", 0)
            state("Ready to train")
            assert worker_value("record.config.lambda") == 0
            assert float(page.locator("#penalty-value").inner_text()) == 0
            assert current_return() > initial
            assert abs(current_return() - worker_value("record.history[0].return")) < 0.0051
            page.wait_for_function(
                "previous => document.getElementById('landscape-chart').toDataURL() !== previous",
                arg=map_before_penalty,
            )
            for learning_rate in (0.001, 0.073, 0.2):
                move_slider("#learning-rate", learning_rate)
                state("Ready to train")
                assert worker_value("record.config.learningRate") == learning_rate
                assert float(page.locator("#learning-rate-value").inner_text()) == learning_rate
                assert worker_value("record.history.length") == 1
                page.locator("#train-button").click()
                state("Training")
                page.wait_for_function("window.__trainingWorkers.at(-1).history.at(-1).iteration >= 16")
                page.locator("#train-button").click()
                state("Paused")
                assert worker_value("record.history.every(point => Number.isFinite(point.return) && Number.isFinite(point.w) && Number.isFinite(point.b))")
            print("PASS: penalty changes recompute the map and learning-rate slider min/intermediate/max train successfully")

            before_click = worker_value("({w: record.config.w, b: record.config.b})")
            policy_before_click = canvas_image("#policy-chart")
            map_before_click = canvas_image("#landscape-chart")
            bounds = page.locator("#landscape-chart").bounding_box()
            page.locator("#landscape-chart").click(position={"x": bounds["width"] * 0.7, "y": bounds["height"] * 0.45})
            state("Ready to train")
            step_is(0)
            after_click = worker_value("({w: record.config.w, b: record.config.b})")
            assert after_click != before_click, "Heatmap click did not choose new initial parameters"
            assert abs(current_return() - worker_value("record.history[0].return")) < 0.0051
            assert canvas_image("#policy-chart") != policy_before_click
            assert canvas_image("#landscape-chart") != map_before_click
            assert worker_value("record.history.length") == 1
            print("PASS: a heatmap click selects a new starting policy and updates both plots")

            assert not errors, errors
            assert not failed_requests, failed_requests
            print(f"PASS: no JavaScript errors or failed requests; screenshots in {artifacts}")
            context.close()
            browser.close()
    finally:
        if server:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
