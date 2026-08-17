// Shared helpers: API access, formatting, toasts, and the SSE connection.

export async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  let payload = null;
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = text; }
  }
  if (!res.ok) {
    const detail = payload && payload.detail ? payload.detail : (payload || res.statusText);
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return payload;
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: 'POST', body: JSON.stringify(body ?? {}) });
export const put = (p, body) => api(p, { method: 'PUT', body: JSON.stringify(body) });
export const patch = (p, body) => api(p, { method: 'PATCH', body: JSON.stringify(body) });
export const del = (p) => api(p, { method: 'DELETE' });

// Timestamps are stored as UTC "YYYY-MM-DD HH:MM:SS"; render them locally.
export function parseTs(ts) {
  if (!ts) return null;
  const d = new Date(ts.replace(' ', 'T') + 'Z');
  return Number.isNaN(d.getTime()) ? null : d;
}

export function fmtTime(ts) {
  const d = parseTs(ts);
  if (!d) return '-';
  return d.toLocaleTimeString('ja-JP', { hour12: false });
}

export function fmtDateTime(ts) {
  const d = parseTs(ts);
  if (!d) return '-';
  return d.toLocaleString('ja-JP', { hour12: false });
}

export function fmtNum(n) {
  if (n === null || n === undefined || n === '') return '-';
  return Number(n).toLocaleString('ja-JP');
}

export function fmtPct(n, digits = 1) {
  if (n === null || n === undefined) return '-';
  return `${(Number(n) * 100).toFixed(digits)}%`;
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function toast(message, kind = '') {
  let host = document.getElementById('toasts');
  if (!host) {
    host = el('div', { id: 'toasts' });
    document.body.append(host);
  }
  const node = el('div', { class: `toast ${kind}`, text: message });
  host.append(node);
  setTimeout(() => node.remove(), kind === 'err' ? 9000 : 4000);
}

export const toastError = (err) => toast(String(err && err.message ? err.message : err), 'err');

/** Subscribe to the server event stream. Reconnects on its own. */
export function connectStream(handlers) {
  let source;
  const open = () => {
    source = new EventSource('/api/stream');
    source.onmessage = (ev) => {
      let payload;
      try { payload = JSON.parse(ev.data); } catch { return; }
      const fn = handlers[payload.type];
      if (fn) fn(payload);
    };
    source.onerror = () => {
      source.close();
      setTimeout(open, 3000);
    };
  };
  open();
  return () => source && source.close();
}

export function renderNav(active) {
  const nav = document.querySelector('nav');
  if (!nav) return;
  for (const link of nav.querySelectorAll('a')) {
    link.classList.toggle('active', link.dataset.page === active);
  }
}
