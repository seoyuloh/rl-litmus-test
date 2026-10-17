# RL Litmus Test — interactive Figure 2

A browser prototype of the paper's controlled adaptive bitrate (ABR) experiment. Two linked plots show where a policy moves during training and what it does at each buffer level:

- **Return heatmap:** color shows expected total reward, and the training path shows changes to the two policy parameters, `w` and `b`.
- **Policy:** two curves show the probability of choosing 4.3 Mbps or 2.85 Mbps as the buffer fills from 0% to 100%.

Training and evaluation run locally in a Web Worker. The site is static HTML, CSS, and JavaScript; it needs no build step, backend, API key, or AI service.

## Run locally

From the repository root:

```sh
python3 -m http.server 4173 --directory web
```

Open <http://localhost:4173>. Serve the files over HTTP so ES modules and the worker can load.

1. Start from `w = 0, b = 0` (each bitrate has probability 50% at every buffer level), or click the heatmap to choose another starting policy.
2. Select a stall penalty and learning rate, then start training.
3. Watch the path and policy curves change. Pause and resume as needed.
4. Move the training-step slider to inspect recorded policies. Every run stops at exactly 10,000 steps. **Train again** starts a fresh 10,000-step run with the selected settings and starting policy; **Reset** returns to step 0 without starting training.

The demo does not extend completed runs. Every update remains available in the timeline; paths are subsampled only for drawing to keep the interface responsive. The worker also enforces the 10,000-step limit. Versioned module URLs keep the app, worker, and training engine on the same release.

## Experiment

The demo uses eight four-second chunks and seven constant-bandwidth traces from 3 to 6 Mbps in 0.5 Mbps increments. Every episode starts with an empty buffer; the buffer is capped at 60 seconds.

The high-bitrate probability is `sigmoid(w × buffer_seconds / 60 + b)`, clipped to `[1e-6, 1 - 1e-6]`. The low-bitrate probability is its complement. The policy chart displays buffer percentage, while the engine uses buffer seconds divided by 60. The heatmap initially spans `w ∈ [-60, 80]` and `b ∈ [-12, 12]`, matching the paper's axes, and expands as the training path approaches an edge.

Per-chunk reward is `bitrate / 4.3 - stall_penalty × stall_seconds / 4`. The initial download counts as stall. Heatmap values are exact expected total episode returns, averaging all action sequences over the seven traces. The path records actual training updates; grid colors approximate the landscape between sampled parameter points.

Training uses sampled, undiscounted REINFORCE with return-to-go, averaged over batch timesteps, and Adam. The default budget is 10,000 updates, with batch size 16, seed 42, and entropy coefficient 0.01. The learning-rate slider ranges from 0.001 to 0.2 in increments of 0.001; the default is 0.05. Changing either setting starts a fresh run. Changing the stall penalty changes the reward objective; the reward formula is shown below the plots.

The demo trains and evaluates on the same traces. It illustrates this controlled experiment; it does not establish performance on unseen bandwidths or production video streams.

## Fidelity to the source

The mathematical port is based on `processing/abr_env.py`, `processing/policy.py`, and `processing/train_vpg_batch_2D.py` at commit `12dced39cf6cd0b44ecbbeac3e8f30392f0471bb`. It preserves buffer normalization, startup stall, reward calculation, and probability clipping.

- The interface fixes the paper's eight-chunk, seven-trace setting. The checked-in source YAML instead specifies four chunks at 4 Mbps.
- The browser defaults to learning rate 0.05 and 10,000 updates. The Python configuration uses learning rate 0.01 and 20,000 updates.
- Constant entropy coefficient 0.01 matches the source's effective minimum; the demo does not reproduce its entropy schedule.
- JavaScript float64 and seeded Mulberry32 replace PyTorch float32 and its RNG. Identical browser configurations reproduce their own runs, but not PyTorch's exact trajectory.

## Validation

`tests/engine.test.js` contains 13 mathematical test groups covering independent trajectory enumeration, finite-difference gradients, sampled/exact update agreement, reproducibility, clipping, monotonicity, input validation, and the engine's threshold-oracle API. The oracle remains an engine test target and is not displayed in the interface. `tests/reference.json` contains 17 independent Python reference cases; regenerate them with `python3 web/tests/reference.py`.

The Playwright runner starts a temporary local server, runs the mathematical suite in a browser, and checks the interface:

```sh
python3 -m pip install playwright
python3 web/tests/browser_test.py
```

It uses installed Google Chrome, falling back to Playwright Chromium. If needed, install Chromium with `python3 -m playwright install chromium`. To check an existing server, pass `--url http://localhost:4173`. This tooling is required only for validation.

## Publish with GitHub Pages

The personal demo is hosted at **https://seoyuloh.github.io/rl-litmus-test/** from **seoyuloh/rl-litmus-test**.

`.github/workflows/pages.yml` deploys runtime files from `web/`, excluding tests and this README. It runs on relevant pushes to `main` or manually through **Actions → Deploy RL Litmus Test to Pages → Run workflow**. GitHub Pages uses **GitHub Actions** as its source.

The static files use relative paths, so the same package can later be hosted at `https://illinois-nsai.github.io/rl-litmus-test/` in an organization-owned repository.
