import { DEFAULT_CONFIG, validateConfig, sigmoid } from './engine.js?v=20261016-origin0';
import { drawLandscape, hitLandscape, drawPolicy } from './charts.js?v=20261016-origin0';

const $ = (id) => document.getElementById(id);
const BOUNDS = { wMin: -60, wMax: 80, bMin: -12, bMax: 12, width: 85, height: 65 };
let initial = { w: DEFAULT_CONFIG.w, b: DEFAULT_CONFIG.b };
let worker;
let state = 'preparing';
let config;
let history = [];
let selected = 0;
let following = true;
let landscape;
let landscapePenalty;
let mapBounds = { ...BOUNDS };
let mapPending = false;
let debounce;
let renderQueued = false;

function setState(next) {
  state = next;
  const labels = {
    preparing: ['Getting ready…', 'Preparing…'],
    ready: ['▶ Train', 'Ready to train'],
    running: ['Ⅱ Pause', 'Training'],
    pausing: ['Pausing…', 'Pausing…'],
    paused: ['▶ Resume', 'Paused'],
    completed: ['▶ Train again', 'Complete'],
    error: ['Unable to train', 'Error'],
  };
  $('train-button').textContent = labels[next][0];
  $('run-status').textContent = labels[next][1];
  $('train-button').disabled = ['preparing', 'pausing', 'error'].includes(next);
  $('train-button').dataset.running = next === 'running';
  $('reset-button').disabled = next === 'preparing' || next === 'pausing';
  $('timeline').disabled = history.length < 2 || next === 'preparing' || next === 'error';
}

function fail(message) {
  worker?.terminate();
  worker = undefined;
  $('error-message').textContent = message;
  $('error-message').hidden = false;
  $('map-loading').hidden = true;
  setState('error');
}

function probability(snapshot, occupancy) {
  return Math.max(1e-6, Math.min(1 - 1e-6, sigmoid(snapshot.w * occupancy + snapshot.b)));
}

function updateView() {
  const snapshot = history[selected];
  if (!snapshot) return;
  $('current-return').textContent = snapshot.return.toFixed(3);
  $('step-label').value = `Step ${snapshot.iteration.toLocaleString()} / ${config.iterations.toLocaleString()}`;
  $('timeline').max = history.length - 1;
  $('timeline').value = selected;
  $('timeline').disabled = history.length < 2;
  $('parameters').textContent = `w = ${snapshot.w.toFixed(2)} · b = ${snapshot.b.toFixed(2)}`;
  let summary;
  let kind;
  if (snapshot.w === 0 && snapshot.b === 0) {
    summary = 'Both bitrates are equally likely at every buffer level.';
    kind = 'constant';
  } else if (!snapshot.greedyMonotonic) {
    summary = 'Reversed rule: more buffer can make this policy prefer a lower bitrate.';
    kind = 'reversed';
  } else if (probability(snapshot, 0) <= 0.5 && probability(snapshot, 1) <= 0.5) {
    summary = 'This policy always prefers the lower bitrate, even with a full buffer.';
    kind = 'constant';
  } else if (probability(snapshot, 0) > 0.5 && probability(snapshot, 1) > 0.5) {
    summary = 'This policy always prefers the higher bitrate, even with an empty buffer.';
    kind = 'constant';
  } else {
    summary = `Follows the buffer rule: it prefers the higher bitrate above ${(snapshot.threshold / 60 * 100).toFixed(1)}% buffer occupancy.`;
    kind = 'learned';
  }
  $('policy-summary').textContent = summary;
  document.querySelector('.policy-note').dataset.kind = kind;
  $('landscape-chart').setAttribute('aria-label', `Return landscape. Selected step ${snapshot.iteration}, weight ${snapshot.w.toFixed(2)}, bias ${snapshot.b.toFixed(2)}, expected return ${snapshot.return.toFixed(3)}. Click or use arrow keys to choose a new starting policy.`);
  $('policy-chart').setAttribute('aria-label', `Bitrate probabilities at step ${snapshot.iteration}. ${summary}`);
  queueCharts();
}

function queueCharts() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    const snapshot = history[selected];
    // Preserve every step for scrubbing, but bound canvas work for long runs.
    const path = [];
    const stride = Math.max(1, Math.ceil((selected + 1) / 2500));
    for (let i = 0; i <= selected; i += stride) path.push(history[i]);
    if (snapshot && path.at(-1) !== snapshot) path.push(snapshot);
    drawLandscape($('landscape-chart'), landscape, path.filter(Boolean), snapshot);
    if (snapshot) drawPolicy($('policy-chart'), snapshot.w, snapshot.b);
  });
}

function expandMap(snapshot) {
  if (mapPending || !landscape) return;
  const spanW = mapBounds.wMax - mapBounds.wMin;
  const spanB = mapBounds.bMax - mapBounds.bMin;
  const next = { ...mapBounds };
  if (snapshot.w < next.wMin + spanW * 0.06) next.wMin = Math.floor((snapshot.w - spanW * 0.25) / 20) * 20;
  if (snapshot.w > next.wMax - spanW * 0.06) next.wMax = Math.ceil((snapshot.w + spanW * 0.25) / 20) * 20;
  if (snapshot.b < next.bMin + spanB * 0.06) next.bMin = Math.floor((snapshot.b - spanB * 0.25) / 4) * 4;
  if (snapshot.b > next.bMax - spanB * 0.06) next.bMax = Math.ceil((snapshot.b + spanB * 0.25) / 4) * 4;
  if (['wMin', 'wMax', 'bMin', 'bMax'].some(key => next[key] !== mapBounds[key])) {
    mapPending = true;
    worker.postMessage({ type: 'landscape', bounds: next });
  }
}

function prepare(autoStart = false) {
  clearTimeout(debounce);
  worker?.terminate();
  worker = undefined;
  mapPending = false;
  $('error-message').hidden = true;
  try {
    config = validateConfig({ ...DEFAULT_CONFIG, ...initial, iterations: 10000, learningRate: Number($('learning-rate').value), lambda: Number($('penalty').value) });
  } catch (error) { fail(error.message); return; }
  const needLandscape = !landscape || landscapePenalty !== config.lambda;
  if (needLandscape) { landscape = undefined; $('map-loading').hidden = false; }
  history = [];
  selected = 0;
  following = true;
  $('timeline').max = 0;
  $('timeline').value = 0;
  $('step-label').value = `Step 0 / ${config.iterations.toLocaleString()}`;
  setState('preparing');
  const activeWorker = new Worker(new URL('./worker.js?v=20261016-origin0', import.meta.url), { type: 'module' });
  worker = activeWorker;
  activeWorker.onerror = (error) => {
    if (worker === activeWorker) fail(error.message || 'The training worker could not start. Please reload the page.');
  };
  activeWorker.onmessage = ({ data }) => {
    if (worker !== activeWorker) return;
    if (data.type === 'error') { fail(data.message); return; }
    if (data.type === 'ready') {
      history = [data.snapshot];
      if (data.landscape) { landscape = data.landscape; landscapePenalty = config.lambda; }
      $('map-loading').hidden = true;
      setState('ready');
      updateView();
      expandMap(data.snapshot);
      if (autoStart) start();
    } else if (data.type === 'update') {
      history.push(...data.history);
      if (following) selected = history.length - 1;
      updateView();
      expandMap(data.snapshot);
      if (data.complete) setState('completed');
    } else if (data.type === 'landscape') {
      landscape = data.landscape;
      mapBounds = data.bounds;
      mapPending = false;
      queueCharts();
      if (history.length) expandMap(history.at(-1));
    } else if (data.type === 'paused' && state === 'pausing') setState('paused');
  };
  activeWorker.postMessage({ type: 'init', config, bounds: needLandscape ? mapBounds : null });
}

function start() {
  if (!['ready', 'paused'].includes(state)) return;
  following = true;
  selected = history.length - 1;
  updateView();
  setState('running');
  worker.postMessage({ type: 'start' });
}

function pause() {
  if (state !== 'running') return;
  setState('pausing');
  worker.postMessage({ type: 'pause' });
}

function restartAfterEdit() {
  worker?.terminate();
  worker = undefined;
  setState('preparing');
  clearTimeout(debounce);
  debounce = setTimeout(() => prepare(), 160);
}

function updateSettings() {
  const penalty = Number($('penalty').value);
  const rate = Number($('learning-rate').value);
  $('penalty-value').value = penalty.toFixed(1);
  $('learning-rate-value').value = String(rate);
}

$('train-button').addEventListener('click', () => {
  if (state === 'running') pause();
  else if (state === 'completed') prepare(true);
  else start();
});
$('reset-button').addEventListener('click', () => prepare());
for (const id of ['penalty', 'learning-rate']) {
  $(id).addEventListener('input', () => {
    updateSettings();
    restartAfterEdit();
  });
}
$('timeline').addEventListener('input', () => {
  selected = Math.max(0, Math.min(history.length - 1, Number($('timeline').value)));
  following = false;
  pause();
  updateView();
});
$('landscape-chart').addEventListener('click', (event) => {
  if (!landscape || state === 'preparing') return;
  const point = hitLandscape($('landscape-chart'), event, landscape);
  if (!point) return;
  initial = { w: point.w, b: point.b };
  prepare();
});
$('landscape-chart').addEventListener('keydown', (event) => {
  const moves = { ArrowLeft: [-5, 0], ArrowRight: [5, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  const delta = moves[event.key];
  if (!delta || !landscape || state === 'preparing') return;
  event.preventDefault();
  initial = { w: Math.max(mapBounds.wMin, Math.min(mapBounds.wMax, initial.w + delta[0])), b: Math.max(mapBounds.bMin, Math.min(mapBounds.bMax, initial.b + delta[1])) };
  prepare();
});
new ResizeObserver(queueCharts).observe(document.querySelector('.plots'));
window.addEventListener('beforeunload', () => worker?.terminate());
updateSettings();
if (location.protocol === 'file:') fail('Open this demo through a local HTTP server. The setup command is in web/README.md.');
else prepare();
