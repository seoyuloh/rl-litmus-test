import { createExperiment } from './engine.js';

let trainer;
let experiment;
let iterations;
let timer;
let running = false;

function stop() {
  running = false;
  clearTimeout(timer);
}

function tick() {
  if (!running) return;
  try {
    const history = [];
    const started = performance.now();
    for (let i = 0; i < 8; i += 1) {
      const snapshot = trainer.step();
      history.push(snapshot);
      if (snapshot.iteration >= iterations || performance.now() - started > 25) break;
    }
    const snapshot = history.at(-1);
    const complete = snapshot.iteration >= iterations;
    postMessage({ type: 'update', snapshot, history, complete });
    if (complete) stop();
    else timer = setTimeout(tick, 35);
  } catch (error) {
    stop();
    postMessage({ type: 'error', message: error.message });
  }
}

self.onmessage = ({ data }) => {
  try {
    if (data.type === 'init') {
      stop();
      experiment = createExperiment(data.config);
      trainer = experiment.createTrainer();
      iterations = experiment.config.iterations;
      const landscape = data.bounds ? experiment.landscape(data.bounds) : null;
      postMessage({ type: 'ready', snapshot: trainer.snapshot(), landscape });
    } else if (data.type === 'start' && trainer && !running) {
      if (trainer.snapshot().iteration >= iterations) return;
      running = true;
      tick();
    } else if (data.type === 'pause') {
      stop();
      postMessage({ type: 'paused' });
    } else if (data.type === 'extend' && trainer && !running) {
      iterations = trainer.extend(data.iterations);
      postMessage({ type: 'extended', iterations });
      running = true;
      tick();
    } else if (data.type === 'landscape' && experiment) {
      postMessage({ type: 'landscape', landscape: experiment.landscape(data.bounds), bounds: data.bounds });
    }
  } catch (error) {
    stop();
    postMessage({ type: 'error', message: error.message });
  }
};
