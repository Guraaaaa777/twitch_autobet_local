// Main screen: registration, the channel table, the point-trend chart, and the log.

import {
  connectStream, del, el, fmtDateTime, fmtNum, fmtTime, get, patch, post,
  renderNav, toast, toastError,
} from '/static/common.js';

const CATEGORY_LABELS = {
  track_start: '追跡開始',
  track_stop: '追跡終了',
  prediction: '投票情報',
  inference: '推論結果',
  bet: '投票結果',
  transcript: '文字起こし',
  system: 'システム',
  error: 'エラー',
};

const state = {
  status: null,
  channels: [],
  chartChannel: null,
  activeCategories: new Set(Object.keys(CATEGORY_LABELS)),
  editing: null,
};

// -- boot -------------------------------------------------------------------

renderNav('main');
document.getElementById('form-register').addEventListener('submit', onRegister);
document.getElementById('btn-start').addEventListener('click', () => runTracking('start'));
document.getElementById('btn-stop').addEventListener('click', () => runTracking('stop'));
document.getElementById('btn-clear-log').addEventListener('click', onClearLog);
document.getElementById('chart-channel').addEventListener('change', (ev) => {
  state.chartChannel = Number(ev.target.value) || null;
  loadChart();
});
for (const dlg of document.querySelectorAll('dialog')) {
  dlg.querySelector('[data-close]').addEventListener('click', () => dlg.close());
}
document.getElementById('fixed-save').addEventListener('click', saveFixed);
document.getElementById('manual-save').addEventListener('click', saveManual);

renderLogFilters();
await refreshAll();
await loadLogs();

connectStream({
  log: (payload) => appendLog(payload.log),
  tracking: () => refreshAll(),
  points: (payload) => { if (payload.channel_id === state.chartChannel) loadChart(); },
  bet: () => refreshStatus(),
  prediction: () => refreshStatus(),
});

setInterval(refreshStatus, 5000);

// -- status and channels ----------------------------------------------------

async function refreshAll() {
  await Promise.all([refreshStatus(), loadChannels()]);
}

async function refreshStatus() {
  try {
    state.status = await get('/api/status');
  } catch (err) {
    state.status = null;
  }
  renderStatusBar();
  if (state.channels.length) renderChannels();
}

function renderStatusBar() {
  const bar = document.getElementById('statusbar');
  const s = state.status;
  bar.replaceChildren();
  if (!s) {
    bar.append(el('span', { class: 'badge err', text: 'サーバ未応答' }));
    return;
  }

  bar.append(el('span', { class: `badge ${s.running ? 'on' : 'off'}`,
    text: s.running ? '追跡中' : '停止中' }));
  bar.append(el('span', { class: `badge ${s.dry_run ? 'warn' : 'err'}`,
    text: s.dry_run ? 'ドライラン' : '実投票' }));
  if (s.running) {
    if (s.llm && s.llm.enabled === false) {
      // Not a failure: it was never started. Saying "停止" here reads like one.
      bar.append(el('span', { class: 'badge off',
        title: `LLM の使用: ${s.llm.mode}`, text: 'LLM 未使用' }));
    } else {
      bar.append(el('span', { class: `badge ${s.llama?.ready ? 'on' : 'off'}`,
        text: `llama.cpp ${s.llama?.ready ? '稼働' : '停止'}` }));
    }
    bar.append(el('span', { class: `badge ${s.pubsub?.connected ? 'on' : 'off'}`,
      text: `PubSub ${s.pubsub?.connected ? '接続' : '未接続'}` }));
    const tr = s.transcription || {};
    const liveCount = Object.values(tr.channels || {}).filter((c) => c.live).length;
    if (tr.enabled) {
      bar.append(el('span', { class: `badge ${liveCount ? 'on' : 'off'}`,
        text: `文字起こし ${liveCount}/${Object.keys(tr.channels || {}).length}` }));
    }
  }
  document.getElementById('btn-start').disabled = s.running;
  document.getElementById('btn-stop').disabled = !s.running;
}

async function runTracking(action) {
  const start = document.getElementById('btn-start');
  const stop = document.getElementById('btn-stop');
  start.disabled = stop.disabled = true;
  try {
    state.status = await post(`/api/tracking/${action}`);
    toast(action === 'start' ? '追跡を開始しました' : '追跡を停止しました', 'ok');
  } catch (err) {
    toastError(err);
  } finally {
    await refreshAll();
  }
}

async function loadChannels() {
  try {
    state.channels = await get('/api/channels');
  } catch (err) {
    toastError(err);
    return;
  }
  renderChannels();
  renderChartSelector();
}

function renderChannels() {
  const tbody = document.querySelector('#table-channels tbody');
  const empty = document.getElementById('channels-empty');
  const table = document.getElementById('table-channels');
  tbody.replaceChildren();

  if (!state.channels.length) {
    table.style.display = 'none';
    empty.style.display = '';
    return;
  }
  table.style.display = '';
  empty.style.display = 'none';

  const live = (state.status && state.status.channels) || {};

  for (const ch of state.channels) {
    const rt = live[String(ch.id)] || ch.runtime || {};
    const fixed = ch.fixed_probs && ch.fixed_probs.enabled ? ch.fixed_probs : null;

    let statusNode;
    if (rt.last_error) {
      statusNode = el('span', { class: 'badge err', title: rt.last_error, text: 'エラー' });
    } else if (rt.active_event_id) {
      statusNode = el('span', {
        class: 'badge warn',
        title: rt.next_bet_at ? `投票予定 ${fmtDateTime(rt.next_bet_at)}` : '',
        text: '投票受付中',
      });
    } else if (rt.last_poll) {
      statusNode = el('span', { class: 'badge on', title: `最終取得 ${fmtDateTime(rt.last_poll)}`,
        text: '追跡中' });
    } else {
      statusNode = el('span', { class: 'badge off', text: '-' });
    }

    tbody.append(el('tr', {},
      el('td', { class: 'mono', text: ch.login }),
      el('td', { text: ch.display_name || '-' }),
      el('td', { class: 'num', text: fmtNum(rt.balance ?? ch.last_points) }),
      el('td', {},
        el('span', { class: `badge ${ch.enabled ? 'on' : 'off'}`,
          text: ch.enabled ? '有効' : '除外' })),
      el('td', {},
        fixed
          ? el('span', { class: 'badge warn mono',
              text: fixed.probs.map((p) => Number(p).toFixed(2)).join(' / ') })
          : el('span', { class: 'muted', text: '-' })),
      el('td', {},
        ch.manual_info
          ? el('span', { class: 'badge', title: ch.manual_info,
              text: `${ch.manual_info.slice(0, 18)}${ch.manual_info.length > 18 ? '…' : ''}` })
          : el('span', { class: 'muted', text: '-' })),
      el('td', {}, statusNode),
      el('td', { class: 'actions' },
        el('button', { class: 'small', onclick: () => toggleEnabled(ch),
          text: ch.enabled ? '追跡除外' : '追跡有効' }),
        el('button', { class: 'small', onclick: () => openFixed(ch), text: '固定確率' }),
        el('button', { class: 'small', onclick: () => openManual(ch), text: '情報付加' }),
        el('button', { class: 'small danger', onclick: () => removeChannel(ch), text: '登録解除' }),
      ),
    ));
  }
}

async function onRegister(ev) {
  ev.preventDefault();
  const input = document.getElementById('input-login');
  const login = input.value.trim();
  if (!login) return;
  const button = ev.target.querySelector('button');
  button.disabled = true;
  try {
    const created = await post('/api/channels', { login });
    input.value = '';
    toast(`${created.display_name || created.login} を登録しました`, 'ok');
    await loadChannels();
  } catch (err) {
    toastError(err);
  } finally {
    button.disabled = false;
  }
}

async function toggleEnabled(ch) {
  try {
    await patch(`/api/channels/${ch.id}`, { enabled: !ch.enabled });
    await loadChannels();
  } catch (err) {
    toastError(err);
  }
}

async function removeChannel(ch) {
  const ok = confirm(
    `${ch.display_name || ch.login} の登録を解除します。\n`
    + 'このチャンネルの蓄積データ (予想履歴・投票記録・ポイント推移・文字起こし) も削除されます。\n\n'
    + 'よろしいですか?'
  );
  if (!ok) return;
  try {
    await del(`/api/channels/${ch.id}`);
    toast('登録を解除しました', 'ok');
    if (state.chartChannel === ch.id) state.chartChannel = null;
    await loadChannels();
    await loadChart();
  } catch (err) {
    toastError(err);
  }
}

// -- modals -----------------------------------------------------------------

function openFixed(ch) {
  state.editing = ch;
  const fixed = ch.fixed_probs || {};
  document.getElementById('fixed-target').textContent = `${ch.display_name || ch.login} (${ch.login})`;
  document.getElementById('fixed-enabled').checked = Boolean(fixed.enabled);
  document.getElementById('fixed-probs').value = (fixed.probs || []).join(', ');
  document.getElementById('dlg-fixed').showModal();
}

async function saveFixed() {
  const ch = state.editing;
  if (!ch) return;
  const enabled = document.getElementById('fixed-enabled').checked;
  const raw = document.getElementById('fixed-probs').value;
  const probs = raw.split(/[,\s]+/).filter(Boolean).map(Number);

  if (enabled) {
    if (probs.length < 2 || probs.some((p) => !Number.isFinite(p) || p < 0)) {
      toast('確率は 0 以上の数値を 2 つ以上入力してください', 'err');
      return;
    }
    if (probs.reduce((a, b) => a + b, 0) <= 0) {
      toast('確率の合計が 0 です', 'err');
      return;
    }
  }
  try {
    await patch(`/api/channels/${ch.id}`, { fixed_probs: { enabled, probs } });
    document.getElementById('dlg-fixed').close();
    toast('固定確率を保存しました', 'ok');
    await loadChannels();
  } catch (err) {
    toastError(err);
  }
}

function openManual(ch) {
  state.editing = ch;
  document.getElementById('manual-target').textContent = `${ch.display_name || ch.login} (${ch.login})`;
  document.getElementById('manual-text').value = ch.manual_info || '';
  document.getElementById('dlg-manual').showModal();
}

async function saveManual() {
  const ch = state.editing;
  if (!ch) return;
  try {
    await patch(`/api/channels/${ch.id}`, {
      manual_info: document.getElementById('manual-text').value,
    });
    document.getElementById('dlg-manual').close();
    toast('手動情報を保存しました', 'ok');
    await loadChannels();
  } catch (err) {
    toastError(err);
  }
}

// -- point trend chart ------------------------------------------------------

function renderChartSelector() {
  const select = document.getElementById('chart-channel');
  const previous = state.chartChannel;
  select.replaceChildren();
  for (const ch of state.channels) {
    select.append(el('option', { value: ch.id, text: ch.display_name || ch.login }));
  }
  if (!state.channels.length) {
    state.chartChannel = null;
  } else if (!previous || !state.channels.some((c) => c.id === previous)) {
    state.chartChannel = state.channels[0].id;
  }
  if (state.chartChannel) select.value = String(state.chartChannel);
  loadChart();
}

async function loadChart() {
  const summary = document.getElementById('chart-summary');
  if (!state.chartChannel) {
    drawChart([]);
    summary.textContent = '';
    return;
  }
  try {
    const data = await get(`/api/channels/${state.chartChannel}/points`);
    drawChart(data.points || []);
    const points = data.points || [];
    if (points.length >= 2) {
      const net = points[points.length - 1].points - points[0].points;
      summary.textContent = `${points.length} 点 / 期間損益 ${net >= 0 ? '+' : ''}${fmtNum(net)} pt`;
    } else {
      summary.textContent = points.length ? '1 点' : 'データなし';
    }
  } catch (err) {
    drawChart([]);
    summary.textContent = '';
  }
}

function drawChart(points) {
  const svg = document.getElementById('chart');
  const W = 1000;
  const H = 260;
  const pad = { top: 14, right: 14, bottom: 24, left: 74 };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.replaceChildren();

  const ns = 'http://www.w3.org/2000/svg';
  const make = (tag, attrs) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };
  const label = (x, y, text, anchor) => {
    const node = make('text', { x, y, class: 'axis', 'text-anchor': anchor || 'start' });
    node.textContent = text;
    return node;
  };

  if (points.length < 2) {
    svg.append(label(W / 2, H / 2, 'ポイント推移データがありません', 'middle'));
    return;
  }

  const xs = points.map((p) => new Date(p.ts.replace(' ', 'T') + 'Z').getTime());
  const ys = points.map((p) => Number(p.points));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const px = (v) => pad.left + ((v - xMin) / (xMax - xMin || 1)) * (W - pad.left - pad.right);
  const py = (v) => H - pad.bottom - ((v - yMin) / (yMax - yMin || 1)) * (H - pad.top - pad.bottom);

  for (let i = 0; i <= 4; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const y = py(value);
    svg.append(make('line', { x1: pad.left, y1: y, x2: W - pad.right, y2: y, class: 'grid' }));
    svg.append(label(pad.left - 8, y + 3, Math.round(value).toLocaleString('ja-JP'), 'end'));
  }

  const ticks = Math.min(6, points.length);
  for (let i = 0; i < ticks; i += 1) {
    const t = xMin + ((xMax - xMin) * i) / (ticks - 1 || 1);
    svg.append(label(px(t), H - 8, new Date(t).toLocaleTimeString('ja-JP', { hour12: false }),
      i === 0 ? 'start' : i === ticks - 1 ? 'end' : 'middle'));
  }

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${px(xs[i]).toFixed(1)},${py(ys[i]).toFixed(1)}`).join('');
  svg.append(make('path', {
    d: `${line}L${px(xs[xs.length - 1]).toFixed(1)},${py(yMin)}L${px(xs[0]).toFixed(1)},${py(yMin)}Z`,
    class: 'area',
  }));
  svg.append(make('path', { d: line, class: 'line' }));
}

// -- log --------------------------------------------------------------------

function renderLogFilters() {
  const host = document.getElementById('log-filters');
  host.replaceChildren();
  for (const [key, name] of Object.entries(CATEGORY_LABELS)) {
    const box = el('input', { type: 'checkbox', checked: true, onchange: (ev) => {
      if (ev.target.checked) state.activeCategories.add(key);
      else state.activeCategories.delete(key);
      applyLogFilter();
    } });
    host.append(el('label', { class: 'small' }, box, ` ${name}`));
  }
}

function applyLogFilter() {
  for (const node of document.querySelectorAll('#log .log-line')) {
    node.style.display = state.activeCategories.has(node.dataset.category) ? '' : 'none';
  }
}

async function loadLogs() {
  try {
    const data = await get('/api/logs?limit=400');
    const host = document.getElementById('log');
    host.replaceChildren();
    for (const entry of (data.logs || []).reverse()) host.append(logNode(entry));
    scrollLog(true);
  } catch (err) {
    toastError(err);
  }
}

function appendLog(entry) {
  const host = document.getElementById('log');
  host.append(logNode(entry));
  while (host.childElementCount > 1200) host.firstElementChild.remove();
  scrollLog(false);
}

function logNode(entry) {
  const node = el('div', {
    class: `log-line ${entry.level}`,
    'data-category': entry.category,
  });
  const head = el('span', { class: 'msg' },
    el('span', { class: 'ts', text: fmtTime(entry.ts) }),
    ' ',
    el('span', { class: 'cat', text: `[${CATEGORY_LABELS[entry.category] || entry.category}]` }),
    entry.channel ? ` (${entry.channel})` : '',
    ` ${entry.message}`,
  );
  node.append(head);

  if (entry.detail) {
    let pretty = entry.detail;
    try { pretty = JSON.stringify(JSON.parse(entry.detail), null, 2); } catch { /* raw */ }
    node.append(el('div', { class: 'detail', text: pretty }));
    head.append(el('span', { class: 'muted', text: ' ▾' }));
    head.addEventListener('click', () => node.classList.toggle('open'));
  }
  if (!state.activeCategories.has(entry.category)) node.style.display = 'none';
  return node;
}

function scrollLog(force) {
  const host = document.getElementById('log');
  if (!force && !document.getElementById('log-follow').checked) return;
  host.scrollTop = host.scrollHeight;
}

async function onClearLog() {
  if (!confirm('ログをすべて消去しますか?')) return;
  try {
    await del('/api/logs');
    await loadLogs();
  } catch (err) {
    toastError(err);
  }
}
