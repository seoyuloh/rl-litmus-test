// Short definitions stay available to mouse, keyboard, and touch users.
const helpButtons = [...document.querySelectorAll('.info-button[data-help]')];
let activeHelp = null;
let closeTimer;

function cancelClose() {
  window.clearTimeout(closeTimer);
}

function positionHelp(item) {
  const { button, popover } = item;
  const gutter = 16;
  const gap = 10;
  const viewportWidth = document.documentElement.clientWidth;
  const viewportHeight = document.documentElement.clientHeight;
  const anchor = button.getBoundingClientRect();

  popover.style.maxWidth = `${Math.max(0, Math.min(340, viewportWidth - gutter * 2))}px`;
  popover.style.maxHeight = `${Math.max(0, viewportHeight - gutter * 2)}px`;
  const size = popover.getBoundingClientRect();
  const left = Math.max(gutter, Math.min(
    anchor.left + anchor.width / 2 - size.width / 2,
    viewportWidth - size.width - gutter,
  ));
  const below = anchor.bottom + gap;
  const above = anchor.top - size.height - gap;
  const preferredTop = below + size.height <= viewportHeight - gutter ? below : above;
  const top = Math.max(gutter, Math.min(preferredTop, viewportHeight - size.height - gutter));

  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
}

function closeHelp() {
  cancelClose();
  if (!activeHelp) return;
  activeHelp.popover.hidden = true;
  activeHelp.button.setAttribute('aria-expanded', 'false');
  activeHelp.pinned = false;
  activeHelp = null;
}

function openHelp(item, pinned = false) {
  cancelClose();
  if (activeHelp !== item) closeHelp();
  activeHelp = item;
  item.pinned = item.pinned || pinned;
  item.popover.hidden = false;
  item.button.setAttribute('aria-expanded', 'true');
  positionHelp(item);
}

function scheduleClose(item) {
  cancelClose();
  closeTimer = window.setTimeout(() => {
    if (activeHelp === item && !item.pinned && !item.hovered
      && document.activeElement !== item.button) closeHelp();
  }, 180);
}

for (const button of helpButtons) {
  const popover = document.getElementById(button.dataset.help);
  if (!popover) continue;

  const item = { button, popover, pinned: false, hovered: false };
  button.setAttribute('aria-controls', popover.id);
  button.setAttribute('aria-expanded', 'false');
  const descriptions = new Set((button.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
  descriptions.add(popover.id);
  button.setAttribute('aria-describedby', [...descriptions].join(' '));
  popover.setAttribute('role', 'tooltip');
  popover.hidden = true;
  // A body-level tooltip is not clipped by a plot or controls container.
  document.body.append(popover);

  button.addEventListener('pointerenter', (event) => {
    if (event.pointerType === 'touch') return;
    item.hovered = true;
    openHelp(item);
  });
  button.addEventListener('pointerleave', (event) => {
    if (event.pointerType === 'touch') return;
    item.hovered = false;
    scheduleClose(item);
  });
  button.addEventListener('focus', () => openHelp(item));
  button.addEventListener('blur', () => scheduleClose(item));
  button.addEventListener('click', () => {
    // Focus/hover can precede a touch-generated click; only a pinned help
    // bubble toggles closed, so the first tap always leaves its help open.
    if (activeHelp === item && item.pinned) closeHelp();
    else openHelp(item, true);
  });
  popover.addEventListener('pointerenter', (event) => {
    if (event.pointerType === 'touch') return;
    item.hovered = true;
    cancelClose();
  });
  popover.addEventListener('pointerleave', (event) => {
    if (event.pointerType === 'touch') return;
    item.hovered = false;
    scheduleClose(item);
  });
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && activeHelp) {
    closeHelp();
    // Keep focus where it is; refocusing the button would reopen its help.
    event.preventDefault();
  }
});

function closeOutsideHelp(event) {
  if (!activeHelp || activeHelp.button.contains(event.target)
    || activeHelp.popover.contains(event.target)) return;
  closeHelp();
}

document.addEventListener('pointerdown', closeOutsideHelp);
document.addEventListener('click', closeOutsideHelp);

function repositionActiveHelp() {
  if (activeHelp) positionHelp(activeHelp);
}

window.addEventListener('resize', repositionActiveHelp);
window.addEventListener('scroll', repositionActiveHelp, { capture: true, passive: true });
