import { DEFAULT_CONFIG, validateConfig, sigmoid, transition, createExperiment } from '../engine.js';

function assert(condition, message = 'Assertion failed') {
  if (!condition) throw new Error(message);
}

function close(actual, expected, tolerance = 1e-9, label = '') {
  assert(Number.isFinite(actual), `${label}: actual is not finite (${actual})`);
  assert(Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, received ${actual}, tolerance ${tolerance}`);
}

function throws(fn) {
  let threw = false;
  try { fn(); } catch { threw = true; }
  assert(threw, 'Expected validation to reject input');
}

// Independent forward trajectory enumeration; the engine evaluates a tree
// backward. This tests probability mass, rewards and score-function gradients.
function enumerate(config, w, b) {
  const result = { return: 0, mass: 0, gradW: 0, gradB: 0, bitrate: 0, stall: 0 };
  for (const bandwidth of config.bandwidths) {
    for (let sequence = 0; sequence < 2 ** config.horizon; sequence += 1) {
      let buffer = 0;
      let probability = 1 / config.bandwidths.length;
      let reward = 0;
      let sumBitrate = 0;
      let sumStall = 0;
      let scoreW = 0;
      let scoreB = 0;
      for (let t = 0; t < config.horizon; t += 1) {
        const action = (sequence >> t) & 1;
        const raw = sigmoid(w * buffer / 60 + b);
        const p = Math.max(1e-6, Math.min(1 - 1e-6, raw));
        const score = raw >= 1e-6 && raw <= 1 - 1e-6 ? action - p : 0;
        scoreW += score * buffer / 60;
        scoreB += score;
        probability *= action ? p : 1 - p;
        const bitrate = action ? 4.3 : 2.85;
        const download = 4 * bitrate / bandwidth;
        const stall = Math.max(0, download - buffer);
        buffer = Math.min(60, Math.max(0, buffer - download) + 4);
        reward += bitrate / 4.3 - config.lambda * stall / 4;
        sumBitrate += bitrate;
        sumStall += stall;
      }
      result.return += probability * reward;
      result.gradW += probability * reward * scoreW;
      result.gradB += probability * reward * scoreB;
      result.bitrate += probability * sumBitrate / config.horizon;
      result.stall += probability * sumStall;
      result.mass += probability;
    }
  }
  return result;
}

export async function runTests() {
  const results = [];
  const test = async (name, fn) => {
    try {
      await fn();
      results.push({ name, passed: true });
    } catch (error) {
      results.push({ name, passed: false, error: error.message });
    }
  };

  await test('Environment includes startup stall, normalizes reward, and caps buffer', () => {
    const initial = transition(0, 1, 4.3);
    close(initial.download, 4);
    close(initial.stall, 4);
    close(initial.reward, -3.3);
    close(initial.buffer, 4);
    const capped = transition(60, 0, 20);
    close(capped.buffer, 60);
    close(capped.stall, 0);
    const depletion = transition(1, 0, 3, { lambda: 1.3 });
    close(depletion.stall, 2.8);
    close(depletion.buffer, 4);
    close(depletion.reward, 2.85 / 4.3 - 1.3 * 2.8 / 4);
  });

  await test('Exact tree matches independent full trajectory enumeration and unit mass', () => {
    for (const cfg of [{ horizon: 1 }, { horizon: 8 }, { horizon: 4, bandwidths: [3, 4, 6], lambda: 1.3 }]) {
      const experiment = createExperiment(cfg);
      for (const [w, b] of [[0, 0], [12, -1], [-40, 8], [1, 100]]) {
        const expected = enumerate(experiment.config, w, b);
        const actual = experiment.evaluate(w, b);
        close(expected.mass, 1, 1e-11, 'Probability mass');
        for (const key of ['return', 'gradW', 'gradB', 'bitrate', 'stall']) {
          close(actual[key], expected[key], 2e-10, key);
        }
      }
    }
  });

  await test('Analytic return gradient matches finite differences, including saturation', () => {
    const experiment = createExperiment();
    const delta = 1e-4;
    for (const [w, b] of [[0.1, -0.1], [23, -2], [-40, 8], [0, -100], [0, 100]]) {
      const analytic = experiment.evaluate(w, b);
      const numericW = (experiment.evaluate(w + delta, b).return - experiment.evaluate(w - delta, b).return) / (2 * delta);
      const numericB = (experiment.evaluate(w, b + delta).return - experiment.evaluate(w, b - delta).return) / (2 * delta);
      close(analytic.gradW, numericW, 1e-7, 'weight gradient');
      close(analytic.gradB, numericB, 1e-7, 'bias gradient');
    }
  });

  await test('Exact SGD update uses return-to-go scaling and explicit entropy coefficient', () => {
    const experiment = createExperiment({ gradient: 'exact', optimizer: 'sgd', entropy: 0.03 });
    const trainer = experiment.createTrainer();
    const before = trainer.snapshot();
    const evaluated = experiment.evaluate(before.w, before.b);
    const after = trainer.step();
    close(after.w, before.w + 0.01 * (evaluated.gradW / 8 + 0.03 * evaluated.entropyGradW));
    close(after.b, before.b + 0.01 * (evaluated.gradB / 8 + 0.03 * evaluated.entropyGradB));
    assert(after.sampledReturn === null, 'Exact update must not invent a sampled return');
  });

  await test('Sampled REINFORCE mean agrees with the exact gradient', () => {
    const config = { horizon: 4, bandwidths: [3, 6], batchSize: 256,
      optimizer: 'sgd', gradient: 'sampled', entropy: 0, learningRate: 0.01 };
    const experiment = createExperiment(config);
    const expected = experiment.evaluate(experiment.config.w, experiment.config.b);
    let dw = 0;
    let db = 0;
    const runs = 160;
    for (let seed = 0; seed < runs; seed += 1) {
      const snapshot = createExperiment({ ...config, seed }).createTrainer().step();
      dw += (snapshot.w - experiment.config.w) / config.learningRate;
      db += (snapshot.b - experiment.config.b) / config.learningRate;
    }
    close(dw / runs, expected.gradW / config.horizon, 0.0015, 'Sampled weight gradient');
    close(db / runs, expected.gradB / config.horizon, 0.006, 'Sampled bias gradient');
  });

  await test('Seeded runs replay exactly and diagnostics never consume training randomness', () => {
    const experiment = createExperiment({ iterations: 30 });
    const a = experiment.createTrainer();
    const b = experiment.createTrainer();
    for (let i = 0; i < 30; i += 1) {
      const first = a.step();
      experiment.evaluate(first.w, first.b);
      experiment.rollout(first.w, first.b, 3.5, false);
      if (i === 12) experiment.landscape({ wMin: -5, wMax: 5, bMin: -2, bMax: 2, width: 3, height: 3 });
      assert(JSON.stringify(first) === JSON.stringify(b.step()), `Seed replay differs at step ${i}`);
    }
    assert(a.step().iteration === 30, 'Trainer must respect the iteration budget');
    const c = createExperiment({ seed: 43 }).createTrainer().step();
    const d = createExperiment({ seed: 42 }).createTrainer().step();
    assert(c.w !== d.w || c.b !== d.b, 'Different seeds should change sampled training');
  });

  await test('Continuing a completed run preserves Adam and RNG state exactly', () => {
    const experiment = createExperiment({ iterations: 30, learningRate: 0.05 });
    const continued = experiment.createTrainer();
    const uninterrupted = createExperiment({ iterations: 60, learningRate: 0.05 }).createTrainer();
    for (let iteration = 1; iteration <= 60; iteration += 1) {
      if (iteration === 31) {
        const completed = JSON.stringify(continued.snapshot());
        assert(JSON.stringify(continued.step()) === completed, 'Completed run must wait for extension');
        assert(continued.extend() === 60, 'Default extension must add the original budget');
        assert(JSON.stringify(continued.snapshot()) === completed, 'Extension must not change the current policy');
      }
      assert(JSON.stringify(continued.step()) === JSON.stringify(uninterrupted.step()),
        `Continuation differs from uninterrupted training at step ${iteration}`);
    }
    assert(continued.step().iteration === 60, 'Extended run must respect its new budget');
    assert(experiment.config.iterations === 30 && Object.isFrozen(experiment.config),
      'Extension must preserve the frozen original configuration');
  });

  await test('Repeated extensions before and after completion keep one uninterrupted trajectory', () => {
    const continued = createExperiment({ iterations: 10 }).createTrainer();
    const uninterrupted = createExperiment({ iterations: 60 }).createTrainer();
    for (let iteration = 1; iteration <= 60; iteration += 1) {
      if (iteration === 11) assert(continued.extend() === 20);
      if (iteration === 16) assert(continued.extend(15) === 35);
      if (iteration === 36) assert(continued.extend() === 45);
      if (iteration === 41) assert(continued.extend(15) === 60);
      assert(JSON.stringify(continued.step()) === JSON.stringify(uninterrupted.step()),
        `Repeated continuation differs at step ${iteration}`);
    }
    assert(continued.step().iteration === 60);
  });

  await test('Invalid extensions leave both the training state and iteration budget unchanged', () => {
    const trainer = createExperiment({ iterations: 2 }).createTrainer();
    const initial = JSON.stringify(trainer.snapshot());
    for (const invalid of [0, -1, 0.5, NaN, Infinity, null, '10', 999999, 1000001]) {
      throws(() => trainer.extend(invalid));
      assert(JSON.stringify(trainer.snapshot()) === initial, 'Rejected extension mutated the current state');
    }
    trainer.step();
    trainer.step();
    assert(trainer.step().iteration === 2, 'Rejected extension mutated the iteration budget');
    assert(trainer.extend(999998) === 1000000, 'The maximum total budget must be allowed');
    throws(() => trainer.extend());
    assert(trainer.step().iteration === 3, 'Failed extension must leave the existing run usable');
  });

  await test('BBA oracle matches an independent fixture and dominates a dense threshold sweep', () => {
    const experiment = createExperiment();
    close(experiment.baseline.return, 3.9393798557477955, 1e-10, 'BBA* oracle');
    for (let threshold = -1; threshold <= 60; threshold += 0.1) {
      let total = 0;
      for (const bandwidth of experiment.config.bandwidths) {
        let buffer = 0;
        for (let t = 0; t < experiment.config.horizon; t += 1) {
          const out = transition(buffer, Number(buffer > threshold), bandwidth, experiment.config);
          total += out.reward;
          buffer = out.buffer;
        }
      }
      assert(total / experiment.config.bandwidths.length <= experiment.baseline.return + 1e-10,
        'A threshold outperformed the enumerated oracle');
    }
  });

  await test('Greedy ties choose low and monotonicity distinguishes constant versus reversed decisions', () => {
    const experiment = createExperiment();
    assert(experiment.rollout(0, 0, 4).every((step) => step.action === 0));
    const reversed = createExperiment({ w: -40, b: 8 }).createTrainer().snapshot();
    assert(!reversed.monotonic && !reversed.greedyMonotonic);
    close(reversed.threshold, 12);
    const constant = createExperiment({ w: -1, b: -1 }).createTrainer().snapshot();
    assert(!constant.monotonic && constant.greedyMonotonic);
    assert(constant.threshold === null);
    const boundary = createExperiment({ w: -1, b: 1 }).createTrainer().snapshot();
    assert(!boundary.greedyMonotonic, 'High to low at the upper endpoint is a violation');
    for (const b of [-100, 100]) {
      const clipped = createExperiment({ w: -1, b }).createTrainer().snapshot();
      assert(clipped.monotonic, 'A fully clipped constant probability is nondecreasing');
      assert(!clipped.slopeNonnegative, 'Clipping must not hide the negative parameter slope');
      assert(clipped.greedyMonotonic, 'A constant greedy policy is nondecreasing');
    }
  });

  await test('Landscape orientation and values agree with exact evaluation', () => {
    const experiment = createExperiment({ horizon: 3 });
    const landscape = experiment.landscape({ wMin: -3, wMax: 9, bMin: -2, bMax: 1, width: 4, height: 3 });
    close(landscape.values[0], experiment.evaluate(-3, -2).return);
    close(landscape.values[3], experiment.evaluate(9, -2).return);
    close(landscape.values[8], experiment.evaluate(-3, 1).return);
    close(landscape.values[11], experiment.evaluate(9, 1).return);
    close(landscape.min, Math.min(...landscape.values));
    close(landscape.max, Math.max(...landscape.values));
  });

  await test('Validation rejects invalid or unbounded workloads and preserves defaults', () => {
    for (const invalid of [null, [], { learningRate: NaN }, { learningRate: 0 }, { batchSize: 0 },
      { batchSize: 2.5 }, { horizon: 25 }, { horizon: 0 }, { iterations: Infinity },
      { iterations: 0 }, { iterations: 1.5 }, { iterations: 1000001 },
      { seed: -1 }, { w: -10001 }, { w: 10001 }, { b: -10001 }, { b: 10001 },
      { bandwidths: [] }, { bandwidths: [0] },
      { bandwidths: ['4'] }, { entropy: -1 }, { optimizer: 'unknown' }, { gradient: 'unknown' }]) {
      throws(() => validateConfig(invalid));
    }
    throws(() => transition(0, 2, 4));
    throws(() => createExperiment().evaluate(NaN, 1));
    const c = validateConfig();
    assert(c.iterations === 20000, 'Default training should run for 20,000 iterations');
    assert(validateConfig({ iterations: 1000000 }).iterations === 1000000);
    assert(validateConfig({ w: 250, b: -250 }).w === 250,
      'Expanded heatmap coordinates must be allowed as initial policy parameters');
    for (const edge of [-10000, 10000]) {
      const boundary = validateConfig({ w: edge, b: edge });
      assert(boundary.w === edge && boundary.b === edge, 'Parameter bounds must be inclusive');
    }
    c.bandwidths.push(8);
    assert(DEFAULT_CONFIG.bandwidths.length === 7, 'Config validation must copy default traces');
    close(c.entropy, 0.01);
    close(sigmoid(-1000), 0);
    close(sigmoid(1000), 1);
  });

  return { passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length, results };
}
