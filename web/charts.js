// Figure 2 views. Drawing and pointer coordinates use CSS pixels.
const FONT = '16px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const COLORS = { text: '#596779', grid: '#e9ecee', orange: '#e87645', high: '#4f79b7', low: '#e58a54' };
const MARGIN = { left: 74, right: 24, top: 80, bottom: 62 };
const rasterCache = new WeakMap();
const finite = Number.isFinite;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const number = value => String(Number(value.toFixed(1)));
const VIRIDIS = [[68, 1, 84], [70, 50, 126], [54, 92, 141], [39, 127, 142], [31, 161, 135], [74, 193, 109], [160, 218, 57], [253, 231, 37]];

function prepare(canvas) {
  const bounds = canvas.getBoundingClientRect();
  const width = canvas.clientWidth || bounds.width || 320;
  const height = canvas.clientHeight || bounds.height || 360;
  const ratio = Math.min(globalThis.devicePixelRatio || 1, 3);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  ctx.font = FONT;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.setLineDash([]);
  return { ctx, width, height };
}

function boxFor(width, height, top = MARGIN.top) {
  return { x: MARGIN.left, y: top, width: Math.max(1, width - MARGIN.left - MARGIN.right), height: Math.max(1, height - top - MARGIN.bottom) };
}

function strokeLine(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function viridis(value) {
  const position = clamp(value, 0, 1) * (VIRIDIS.length - 1);
  const index = Math.min(VIRIDIS.length - 2, Math.floor(position));
  const mix = position - index;
  return VIRIDIS[index].map((channel, offset) => Math.round(channel * (1 - mix) + VIRIDIS[index + 1][offset] * mix));
}

function ticks(min, max, count = 5) {
  const approximate = (max - min) / (count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(approximate));
  const normalized = approximate / magnitude;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 4 ? 4 : normalized <= 5 ? 5 : 10) * magnitude;
  const values = [min];
  for (let value = Math.ceil(min / step) * step; value < max; value += step) {
    if (value - min >= step * 0.6 && max - value >= step * 0.6) values.push(value);
  }
  values.push(max);
  return values;
}

function axes(ctx, box, xTicks, yTicks, x, y, xLabel, yLabel, grid = false) {
  ctx.fillStyle = COLORS.text;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const tick of xTicks) {
    if (grid) strokeLine(ctx, x(tick), box.y, x(tick), box.y + box.height);
    ctx.fillText(number(tick), x(tick), box.y + box.height + 9);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const tick of yTicks) {
    if (grid) strokeLine(ctx, box.x, y(tick), box.x + box.width, y(tick));
    ctx.fillText(number(tick), box.x - 9, y(tick));
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(xLabel, box.x + box.width / 2, box.y + box.height + 36);
  ctx.save();
  ctx.translate(13, box.y + box.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'middle';
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function validLandscape(landscape) {
  return landscape && landscape.values && landscape.width > 0 && landscape.height > 0
    && finite(landscape.wMin) && finite(landscape.wMax) && landscape.wMax > landscape.wMin
    && finite(landscape.bMin) && finite(landscape.bMax) && landscape.bMax > landscape.bMin
    && finite(landscape.min) && finite(landscape.max);
}

function raster(landscape) {
  const cached = rasterCache.get(landscape);
  if (cached && cached.values === landscape.values) return cached.canvas;
  const canvas = document.createElement('canvas');
  canvas.width = landscape.width;
  canvas.height = landscape.height;
  const ctx = canvas.getContext('2d');
  const pixels = ctx.createImageData(canvas.width, canvas.height);
  const range = Math.max(1e-12, landscape.max - landscape.min);
  const nested = Array.isArray(landscape.values[0]) || ArrayBuffer.isView(landscape.values[0]);
  for (let row = 0; row < canvas.height; row++) {
    for (let col = 0; col < canvas.width; col++) {
      const value = nested ? landscape.values[row][col] : landscape.values[row * canvas.width + col];
      const color = viridis(finite(value) ? (value - landscape.min) / range : 0);
      const offset = ((canvas.height - 1 - row) * canvas.width + col) * 4;
      pixels.data[offset] = color[0];
      pixels.data[offset + 1] = color[1];
      pixels.data[offset + 2] = color[2];
      pixels.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(pixels, 0, 0);
  rasterCache.set(landscape, { canvas, values: landscape.values });
  return canvas;
}

/** Exact expected return over (w, b), with the actual optimizer path. */
export function drawLandscape(canvas, landscape, history = [], current = null) {
  const { ctx, width, height } = prepare(canvas);
  const box = boxFor(width, height);
  if (!validLandscape(landscape)) {
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Preparing the return map…', box.x + box.width / 2, box.y + box.height / 2);
    return;
  }
  const x = w => box.x + (w - landscape.wMin) / (landscape.wMax - landscape.wMin) * box.width;
  const y = b => box.y + (landscape.bMax - b) / (landscape.bMax - landscape.bMin) * box.height;
  ctx.fillStyle = COLORS.text;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('Expected return', box.x, 0);
  for (let pixel = 0; pixel < Math.ceil(box.width); pixel++) {
    ctx.fillStyle = `rgb(${viridis(pixel / Math.max(1, box.width - 1)).join(',')})`;
    ctx.fillRect(box.x + pixel, 26, 1, 12);
  }
  ctx.fillStyle = COLORS.text;
  ctx.fillText(number(landscape.min), box.x, 45);
  ctx.textAlign = 'right';
  ctx.fillText(number(landscape.max), box.x + box.width, 45);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(raster(landscape), box.x, box.y, box.width, box.height);
  axes(ctx, box, ticks(landscape.wMin, landscape.wMax, box.width < 340 ? 5 : 8), ticks(landscape.bMin, landscape.bMax, 7), x, y, 'Weight w', 'Bias b');

  ctx.save();
  ctx.beginPath();
  ctx.rect(box.x, box.y, box.width, box.height);
  ctx.clip();
  ctx.strokeStyle = 'rgba(255,255,255,0.7)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  if (landscape.wMin <= 0 && landscape.wMax >= 0) strokeLine(ctx, x(0), box.y, x(0), box.y + box.height);
  if (landscape.bMin <= 0 && landscape.bMax >= 0) strokeLine(ctx, box.x, y(0), box.x + box.width, y(0));
  ctx.setLineDash([]);
  const points = history.filter(point => finite(point.w) && finite(point.b));
  const end = current && finite(current.w) && finite(current.b) ? current : points[points.length - 1];
  if (points.length) {
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(x(point.w), y(point.b));
      else ctx.lineTo(x(point.w), y(point.b));
    });
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.strokeStyle = COLORS.orange;
    ctx.lineWidth = 2.3;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x(points[0].w), y(points[0].b), 5, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const outside = end && (end.w < landscape.wMin || end.w > landscape.wMax || end.b < landscape.bMin || end.b > landscape.bMax);
  if (end && !outside) {
    ctx.beginPath();
    ctx.arc(x(end.w), y(end.b), 5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.orange;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
  if (outside) {
    const label = 'Policy is outside this map';
    ctx.font = FONT;
    const lines = ctx.measureText(label).width + 30 > box.width ? ['Policy is outside', 'this map'] : [label];
    const labelWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
    ctx.fillStyle = 'rgba(255,255,255,0.96)';
    ctx.fillRect(box.x + 7, box.y + 7, labelWidth + 16, lines.length * 22 + 10);
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    lines.forEach((line, index) => ctx.fillText(line, box.x + 15, box.y + 23 + index * 22));
  }
}

/** Translate a click inside the return map to policy parameters. */
export function hitLandscape(canvas, event, landscape) {
  if (!validLandscape(landscape)) return null;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return null;
  const width = canvas.clientWidth || bounds.width;
  const height = canvas.clientHeight || bounds.height;
  const box = boxFor(width, height);
  const x = (event.clientX - bounds.left) * width / bounds.width;
  const y = (event.clientY - bounds.top) * height / bounds.height;
  if (x < box.x || x > box.x + box.width || y < box.y || y > box.y + box.height) return null;
  return { w: landscape.wMin + (x - box.x) / box.width * (landscape.wMax - landscape.wMin), b: landscape.bMax - (y - box.y) / box.height * (landscape.bMax - landscape.bMin) };
}

/** Both bitrate probabilities at each normalized buffer occupancy. */
export function drawPolicy(canvas, w, b) {
  const { ctx, width, height } = prepare(canvas);
  const box = boxFor(width, height, 17);
  const x = occupancy => box.x + occupancy / 100 * box.width;
  const y = probability => box.y + (1 - probability) * box.height;
  const xTicks = box.width < 230 ? [0, 50, 100] : box.width < 340 ? [0, 25, 50, 75, 100] : [0, 20, 40, 60, 80, 100];
  axes(ctx, box, xTicks, [0, 20, 40, 60, 80, 100], x, value => y(value / 100), 'Buffer occupancy (%)', 'Probability (%)', true);
  if (!finite(w) || !finite(b)) return;
  const threshold = w === 0 ? NaN : -b / w * 100;
  if (finite(threshold) && threshold >= 0 && threshold <= 100) {
    ctx.strokeStyle = '#a6adb7';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([3, 4]);
    strokeLine(ctx, x(threshold), y(0), x(threshold), y(0.5));
    ctx.setLineDash([]);
  }
  for (const high of [true, false]) {
    ctx.beginPath();
    for (let index = 0; index <= 300; index++) {
      const occupancy = index / 3;
      const probability = clamp(1 / (1 + Math.exp(-clamp(w * occupancy / 100 + b, -50, 50))), 1e-6, 1 - 1e-6);
      const yy = y(high ? probability : 1 - probability);
      if (index === 0) ctx.moveTo(x(occupancy), yy);
      else ctx.lineTo(x(occupancy), yy);
    }
    ctx.strokeStyle = high ? COLORS.high : COLORS.low;
    ctx.lineWidth = 2.6;
    ctx.setLineDash(high ? [] : [7, 5]);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}
