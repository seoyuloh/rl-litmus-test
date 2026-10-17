# RL Litmus Test

Temporary browser demo of the controlled adaptive bitrate (ABR) experiment.

**Website:** https://seoyuloh.github.io/rl-litmus-test/

Choose a starting policy on the return heatmap, adjust stall penalty and learning rate, then train in your browser. The heatmap and bitrate policy update together. Each run lasts 10,000 updates; Continue adds another 10,000 without resetting training.

## Run locally

```sh
python3 -m http.server 4173 --directory web
```

Open http://localhost:4173/. This is a static HTML/CSS/JavaScript site with no backend, API key, or build step.

## Source and validation

This prototype is based on [illinois-nsai/Demystify_ML](https://github.com/illinois-nsai/Demystify_ML) at commit `12dced39cf6cd0b44ecbbeac3e8f30392f0471bb`. The simulation and training code are a JavaScript port of its controlled ABR experiment. The browser implementation is a prototype, not an exact reproduction of the paper's numerical results.

See [web/README.md](web/README.md) for the reward definition, experiment assumptions, source differences, and validation commands.

## Deployment

GitHub Pages publishes the eight runtime files from `web/` using `.github/workflows/pages.yml`. Changes to `web/` on `main` deploy automatically; the workflow can also be run manually. Tests and documentation are excluded from the published artifact.
