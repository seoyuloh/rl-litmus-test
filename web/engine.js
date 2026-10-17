/**
 * Controlled ABR, ported from processing/abr_env.py and train_vpg_batch_2D.py.
 * Browser arithmetic is float64; the original policy uses float32 tensors.
 * The manuscript's 8 chunks / 7 traces are intentional, rather than the
 * committed config.yaml's 4 chunks / one trace. No network or ML dependency.
 */
export const DEFAULT_CONFIG = Object.freeze({
  learningRate: 0.01,
  batchSize: 16,
  iterations: 10000,
  seed: 42,
  lambda: 4.3,
  horizon: 8,
  bandwidths: Object.freeze([3, 3.5, 4, 4.5, 5, 5.5, 6]),
  w: 0.1,
  b: -0.1,
  optimizer: 'adam',
  gradient: 'sampled',
  // The original max(schedule, .01) has this floor even when YAML says 0.
  entropy: 0.01,
});

const BITRATES = [2.85, 4.3];
const CHUNK = 4;
const BUFFER_MAX = 60;
const EPS = 1e-6;
export const MAX_ITERATIONS = 1_000_000;

function boundedNumber(value, name, min, max, integer = false) {
  if (typeof value !== 'number' || !Number.isFinite(value)
      || value < min || value > max || (integer && !Number.isInteger(value))) {
    throw new RangeError(`${name} must be ${integer ? 'an integer' : 'a finite number'} from ${min} to ${max}.`);
  }
  return value;
}

export function validateConfig(input = {}) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('Training configuration must be an object.');
  }
  const c = { ...DEFAULT_CONFIG, ...input };
  const cleaned = {
    learningRate: boundedNumber(c.learningRate, 'Learning rate', 0.00001, 5),
    batchSize: boundedNumber(c.batchSize, 'Batch size', 1, 256, true),
    iterations: boundedNumber(c.iterations, 'Iterations', 1, MAX_ITERATIONS, true),
    seed: boundedNumber(c.seed, 'Seed', 0, 0xffffffff, true),
    lambda: boundedNumber(c.lambda, 'Stall penalty', 0, 20),
    horizon: boundedNumber(c.horizon, 'Episode length', 1, 10, true),
    w: boundedNumber(c.w, 'Initial weight', -10000, 10000),
    b: boundedNumber(c.b, 'Initial bias', -10000, 10000),
    entropy: boundedNumber(c.entropy, 'Entropy coefficient', 0, 1),
    optimizer: c.optimizer,
    gradient: c.gradient,
  };
  if (!Array.isArray(c.bandwidths) || c.bandwidths.length < 1 || c.bandwidths.length > 16) {
    throw new RangeError('Select 1 to 16 constant bandwidth traces.');
  }
  cleaned.bandwidths = c.bandwidths.map((v) => boundedNumber(v, 'Bandwidth', 0.25, 20));
  if (!['adam', 'sgd'].includes(c.optimizer)) throw new RangeError('Optimizer must be adam or sgd.');
  if (!['sampled', 'exact'].includes(c.gradient)) throw new RangeError('Gradient must be sampled or exact.');
  return cleaned;
}

export function sigmoid(z) {
  // Stable on both saturated plateaus.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function policy(z) {
  const raw = sigmoid(z);
  const p = Math.max(EPS, Math.min(1 - EPS, raw));
  // torch.clamp blocks the derivative outside its interval.
  const derivative = raw < EPS || raw > 1 - EPS ? 0 : raw * (1 - raw);
  return { p, derivative };
}

function advance(buffer, action, bandwidth, lambda) {
  const bitrate = BITRATES[action];
  const download = bitrate * CHUNK / bandwidth;
  const stall = Math.max(download - buffer, 0);
  return {
    buffer: Math.min(BUFFER_MAX, Math.max(buffer - download, 0) + CHUNK),
    reward: bitrate / BITRATES[1] - lambda * stall / CHUNK,
    stall,
    bitrate,
    download,
  };
}

export function transition(buffer, action, bandwidth, config = DEFAULT_CONFIG) {
  boundedNumber(buffer, 'Buffer', 0, BUFFER_MAX);
  boundedNumber(action, 'Action', 0, 1, true);
  boundedNumber(bandwidth, 'Bandwidth', 0.25, 20);
  const lambda = boundedNumber(config.lambda ?? DEFAULT_CONFIG.lambda, 'Stall penalty', 0, 20);
  return advance(buffer, action, bandwidth, lambda);
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function checkParameters(w, b) {
  if (!Number.isFinite(w) || !Number.isFinite(b)) throw new RangeError('Policy parameters must be finite.');
}

export function createExperiment(input = {}) {
  const config = validateConfig(input);
  Object.freeze(config.bandwidths);
  Object.freeze(config);
  const { horizon, bandwidths, lambda } = config;
  const size = 2 ** horizon - 1;
  const lastStart = 2 ** (horizon - 1) - 1;
  const reachableBuffers = new Set([0, BUFFER_MAX]);

  // A complete binary tree stores states once; probability and value depend on
  // the current parameters. This enumerates every sequence without sampling.
  const trees = bandwidths.map((bandwidth) => {
    const buffers = new Float64Array(size);
    const x = new Float64Array(size);
    const r0 = new Float64Array(size);
    const r1 = new Float64Array(size);
    const s0 = new Float64Array(size);
    const s1 = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      const low = advance(buffers[i], 0, bandwidth, lambda);
      const high = advance(buffers[i], 1, bandwidth, lambda);
      x[i] = buffers[i] / BUFFER_MAX;
      reachableBuffers.add(buffers[i]);
      r0[i] = low.reward;
      r1[i] = high.reward;
      s0[i] = low.stall;
      s1[i] = high.stall;
      if (i < lastStart) {
        buffers[2 * i + 1] = low.buffer;
        buffers[2 * i + 2] = high.buffer;
      }
    }
    return { x, r0, r1, s0, s1 };
  });

  // Synchronous calls reuse scratch arrays. No persistent RNG is involved in
  // diagnostics, so drawing a heatmap cannot change the optimization path.
  const value = new Float64Array(size);
  const gradW = new Float64Array(size);
  const gradB = new Float64Array(size);
  const bitrate = new Float64Array(size);
  const stall = new Float64Array(size);
  const entropy = new Float64Array(size);
  const entropyW = new Float64Array(size);
  const entropyB = new Float64Array(size);

  function evaluate(w, b) {
    checkParameters(w, b);
    const result = { return: 0, gradW: 0, gradB: 0, bitrate: 0, stall: 0,
      entropy: 0, entropyGradW: 0, entropyGradB: 0 };
    for (const tree of trees) {
      for (let i = size - 1; i >= 0; i -= 1) {
        const { p, derivative } = policy(w * tree.x[i] + b);
        const p0 = 1 - p;
        const left = 2 * i + 1;
        const right = left + 1;
        const hasChildren = i < lastStart;
        const q0 = tree.r0[i] + (hasChildren ? value[left] : 0);
        const q1 = tree.r1[i] + (hasChildren ? value[right] : 0);
        const direct = derivative * (q1 - q0);
        value[i] = p0 * q0 + p * q1;
        gradW[i] = direct * tree.x[i] + (hasChildren ? p0 * gradW[left] + p * gradW[right] : 0);
        gradB[i] = direct + (hasChildren ? p0 * gradB[left] + p * gradB[right] : 0);
        bitrate[i] = p0 * BITRATES[0] + p * BITRATES[1]
          + (hasChildren ? p0 * bitrate[left] + p * bitrate[right] : 0);
        stall[i] = p0 * tree.s0[i] + p * tree.s1[i]
          + (hasChildren ? p0 * stall[left] + p * stall[right] : 0);
        entropy[i] = -p0 * Math.log(p0) - p * Math.log(p)
          + (hasChildren ? p0 * entropy[left] + p * entropy[right] : 0);
        // Expected local entropy derivative, holding the sampled states fixed,
        // exactly as in the source training loss (not d/dθ of state occupancy).
        const localEntropyGradient = derivative * Math.log(p0 / p);
        entropyW[i] = localEntropyGradient * tree.x[i]
          + (hasChildren ? p0 * entropyW[left] + p * entropyW[right] : 0);
        entropyB[i] = localEntropyGradient
          + (hasChildren ? p0 * entropyB[left] + p * entropyB[right] : 0);
      }
      result.return += value[0];
      result.gradW += gradW[0];
      result.gradB += gradB[0];
      result.bitrate += bitrate[0] / horizon;
      result.stall += stall[0];
      result.entropy += entropy[0] / horizon;
      result.entropyGradW += entropyW[0] / horizon;
      result.entropyGradB += entropyB[0] / horizon;
    }
    for (const key of Object.keys(result)) result[key] /= bandwidths.length;
    return result;
  }

  function thresholdPerformance(threshold) {
    let rewardSum = 0;
    let bitrateSum = 0;
    let stallSum = 0;
    for (const bandwidth of bandwidths) {
      let buffer = 0;
      for (let t = 0; t < horizon; t += 1) {
        const out = advance(buffer, buffer > threshold ? 1 : 0, bandwidth, lambda);
        buffer = out.buffer;
        rewardSum += out.reward;
        bitrateSum += out.bitrate;
        stallSum += out.stall;
      }
    }
    return { return: rewardSum / bandwidths.length, threshold,
      bitrate: bitrateSum / (bandwidths.length * horizon), stall: stallSum / bandwidths.length };
  }

  // Every threshold decision can change only at a reachable buffer. Testing
  // each breakpoint and one below zero exhausts this policy class (ties low).
  let baseline = thresholdPerformance(-1);
  for (const threshold of [...reachableBuffers].sort((a, b) => a - b)) {
    const candidate = thresholdPerformance(threshold);
    if (candidate.return > baseline.return + 1e-12) baseline = candidate;
  }
  Object.freeze(baseline);

  function landscape({ wMin, wMax, bMin, bMax, width, height }) {
    checkParameters(wMin, bMin);
    checkParameters(wMax, bMax);
    if (wMin >= wMax || bMin >= bMax) throw new RangeError('Landscape axis bounds must increase.');
    boundedNumber(width, 'Landscape width', 2, 128, true);
    boundedNumber(height, 'Landscape height', 2, 128, true);
    const values = new Float64Array(width * height);
    let min = Infinity;
    let max = -Infinity;
    for (let row = 0; row < height; row += 1) {
      const b = bMin + (bMax - bMin) * row / (height - 1);
      for (let col = 0; col < width; col += 1) {
        const w = wMin + (wMax - wMin) * col / (width - 1);
        // Heatmaps need only return; skip derivative / entropy / metric work.
        let returnSum = 0;
        for (const tree of trees) {
          for (let i = size - 1; i >= 0; i -= 1) {
            const p = Math.max(EPS, Math.min(1 - EPS, sigmoid(w * tree.x[i] + b)));
            const left = 2 * i + 1;
            const q0 = tree.r0[i] + (i < lastStart ? value[left] : 0);
            const q1 = tree.r1[i] + (i < lastStart ? value[left + 1] : 0);
            value[i] = (1 - p) * q0 + p * q1;
          }
          returnSum += value[0];
        }
        const expectedReturn = returnSum / bandwidths.length;
        values[row * width + col] = expectedReturn;
        min = Math.min(min, expectedReturn);
        max = Math.max(max, expectedReturn);
      }
    }
    return { values, min, max, wMin, wMax, bMin, bMax, width, height };
  }

  function rollout(w, b, bandwidth, deterministic = true) {
    checkParameters(w, b);
    boundedNumber(bandwidth, 'Bandwidth', 0.25, 20);
    const random = seededRandom(config.seed ^ 0x9e3779b9);
    const steps = [];
    let buffer = 0;
    for (let index = 0; index < horizon; index += 1) {
      const { p } = policy(w * buffer / BUFFER_MAX + b);
      const action = deterministic ? Number(p > 0.5) : Number(random() < p);
      const out = advance(buffer, action, bandwidth, lambda);
      steps.push({ index, bufferBefore: buffer, ...out, action, bandwidth, probability: p });
      buffer = out.buffer;
    }
    return steps;
  }

  function createTrainer() {
    const random = seededRandom(config.seed);
    let w = config.w;
    let b = config.b;
    let iteration = 0;
    let iterationLimit = config.iterations;
    let sampledReturn = null;
    let mw = 0;
    let mb = 0;
    let vw = 0;
    let vb = 0;
    let evaluated = evaluate(w, b);

    function snapshot() {
      const crossing = w === 0 ? null : -BUFFER_MAX * b / w;
      return {
        iteration, w, b,
        return: evaluated.return,
        gradNorm: Math.hypot(evaluated.gradW, evaluated.gradB),
        sampledReturn,
        bitrate: evaluated.bitrate,
        stall: evaluated.stall,
        // The affine sigmoid is monotone in one direction. Clipping can make a
        // negative-slope policy constant over the entire buffer domain.
        monotonic: policy(b).p <= policy(w + b).p,
        slopeNonnegative: w >= 0,
        greedyMonotonic: !(w < 0 && b > 0 && w + b <= 0),
        threshold: crossing !== null && crossing >= 0 && crossing <= BUFFER_MAX ? crossing : null,
      };
    }

    // Extend the same run, retaining its random stream and optimizer moments.
    function extend(additionalIterations = config.iterations) {
      boundedNumber(additionalIterations, 'Additional iterations', 1, MAX_ITERATIONS, true);
      const nextLimit = boundedNumber(iterationLimit + additionalIterations,
        'Total iterations', 1, MAX_ITERATIONS, true);
      iterationLimit = nextLimit;
      return iterationLimit;
    }

    function step() {
      if (iteration >= iterationLimit) return snapshot();
      let gw = 0;
      let gb = 0;
      if (config.gradient === 'exact') {
        gw = evaluated.gradW / horizon + config.entropy * evaluated.entropyGradW;
        gb = evaluated.gradB / horizon + config.entropy * evaluated.entropyGradB;
      } else {
        let rewardSum = 0;
        for (let episode = 0; episode < config.batchSize; episode += 1) {
          const bandwidth = bandwidths[Math.floor(random() * bandwidths.length)];
          const states = [];
          let buffer = 0;
          for (let t = 0; t < horizon; t += 1) {
            const x = buffer / BUFFER_MAX;
            const { p, derivative } = policy(w * x + b);
            const action = Number(random() < p);
            const out = advance(buffer, action, bandwidth, lambda);
            // d log π(a|s) / dz; zero after probability clipping.
            const score = derivative === 0 ? 0 : (action ? derivative / p : -derivative / (1 - p));
            states.push({ x, score, reward: out.reward,
              entropyGradient: derivative * Math.log((1 - p) / p) });
            buffer = out.buffer;
          }
          let returnToGo = 0;
          for (let t = horizon - 1; t >= 0; t -= 1) {
            const state = states[t];
            returnToGo += state.reward;
            const local = state.score * returnToGo + config.entropy * state.entropyGradient;
            gw += local * state.x;
            gb += local;
          }
          rewardSum += returnToGo;
        }
        const denominator = config.batchSize * horizon;
        gw /= denominator;
        gb /= denominator;
        sampledReturn = rewardSum / config.batchSize;
      }
      iteration += 1;
      if (config.optimizer === 'adam') {
        mw = 0.9 * mw + 0.1 * gw;
        mb = 0.9 * mb + 0.1 * gb;
        vw = 0.999 * vw + 0.001 * gw * gw;
        vb = 0.999 * vb + 0.001 * gb * gb;
        const correction1 = 1 - 0.9 ** iteration;
        const correction2 = 1 - 0.999 ** iteration;
        w += config.learningRate * (mw / correction1) / (Math.sqrt(vw / correction2) + 1e-8);
        b += config.learningRate * (mb / correction1) / (Math.sqrt(vb / correction2) + 1e-8);
      } else {
        w += config.learningRate * gw;
        b += config.learningRate * gb;
      }
      checkParameters(w, b);
      evaluated = evaluate(w, b);
      return snapshot();
    }
    return { step, snapshot, extend };
  }

  return { config, evaluate, baseline, landscape, rollout, createTrainer };
}
