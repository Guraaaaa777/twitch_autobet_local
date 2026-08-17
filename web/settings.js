// Settings screen. The form is generated from SECTIONS so that adding a knob
// means adding one entry here, not editing HTML and JS in two places.

import { el, get, post, put, renderNav, toast, toastError } from '/static/common.js';

const CACHE_TYPES = ['f32', 'f16', 'bf16', 'q8_0', 'q5_1', 'q5_0', 'q4_1', 'q4_0'];

const SECTIONS = [
  {
    title: 'llama.cpp 設定',
    fields: [
      { path: 'llama.server_path', label: 'llama-server 実行ファイル', type: 'text',
        hint: '例: C:\\llama.cpp\\llama-server.exe' },
      { path: 'llama.model_path', label: 'モデル (GGUF)', type: 'text',
        hint: '例: C:\\models\\qwen2.5-7b-instruct-q4_k_m.gguf' },
      { path: 'llama.ctx_size', label: 'コンテキスト長', type: 'number', min: 512, step: 512,
        hint: '文字起こしをプロンプトに含めるため、余裕を持たせてください' },
      { path: 'llama.n_gpu_layers', label: '総数 (GPU オフロード層数)', type: 'number', min: 0,
        hint: '--n-gpu-layers。GPU がない環境では 0' },
      { path: 'llama.threads', label: 'スレッド数', type: 'number', min: 1 },
      { path: 'llama.batch_size', label: 'バッチサイズ', type: 'number', min: 1 },
      { path: 'llama.cache_type_k', label: '量子化タイプ (K キャッシュ)', type: 'select',
        options: CACHE_TYPES, hint: '--cache-type-k。VRAM/RAM 節約に効きます' },
      { path: 'llama.cache_type_v', label: '量子化タイプ (V キャッシュ)', type: 'select',
        options: CACHE_TYPES, hint: '--cache-type-v' },
      { path: 'llama.lora_path', label: 'LoRA', type: 'text', hint: '未使用なら空欄' },
      { path: 'llama.lora_scale', label: 'LoRA スケール', type: 'number', step: 0.05, min: 0 },
      { path: 'llama.host', label: '待ち受けホスト', type: 'text' },
      { path: 'llama.port', label: '待ち受けポート', type: 'number', min: 1, max: 65535 },
      { path: 'llama.temperature', label: 'temperature', type: 'number', step: 0.05, min: 0 },
      { path: 'llama.max_tokens', label: '最大生成トークン数', type: 'number', min: 64 },
      { path: 'llama.history_limit', label: 'プロンプトに含める過去予想数', type: 'number', min: 0,
        hint: '同じチャンネルの決着済み予想を、この件数まで推論の参考に渡します' },
      { path: 'llama.startup_timeout_sec', label: '起動タイムアウト (秒)', type: 'number', min: 10 },
      { path: 'llama.request_timeout_sec', label: '推論タイムアウト (秒)', type: 'number', min: 5 },
      { path: 'llama.extra_args', label: '追加の起動引数', type: 'text',
        hint: 'そのまま llama-server に渡されます' },
    ],
  },
  {
    title: 'ポーリングレート設定',
    fields: [
      { path: 'poll_rate_sec', label: '投票ポーリング間隔 (秒)', type: 'number', step: 0.5, min: 1,
        hint: 'チャンネルごとにこの間隔で予想状況を取得します。短すぎるとレート制限を受けます' },
      { path: 'points_poll_sec', label: 'ポイント記録間隔 (秒)', type: 'number', step: 1, min: 5,
        hint: '推移グラフ用にチャンネルポイント残高を記録する間隔' },
    ],
  },
  {
    title: '認証',
    fields: [
      { path: 'twitch.oauth_token', label: 'Twitch OAuth トークン', type: 'password',
        hint: 'twitch.tv にログインした状態の Cookie「auth-token」の値。'
            + 'このトークンはアカウントそのものへのアクセス権を持ちます' },
      { path: 'twitch.client_id', label: 'Twitch Client-Id', type: 'text',
        hint: 'Twitch Web クライアントの公開 Client-Id。通常は変更不要' },
      { path: 'twitch.device_id', label: 'Twitch Device-Id', type: 'text', hint: '任意' },
      { path: 'twitch.use_pubsub', label: 'PubSub を使用する', type: 'checkbox',
        hint: '予想の開始をリアルタイムに受信します。オフでもポーリングで動作します' },
      { path: 'twitch.request_timeout_sec', label: 'Twitch リクエストタイムアウト (秒)',
        type: 'number', min: 5 },
      { path: 'llama.api_key', label: 'LLM 用 API 認証キー', type: 'password',
        hint: 'llama-server の --api-key。空欄なら認証なしで起動します' },
    ],
  },
  {
    title: '文字起こし',
    fields: [
      { path: 'transcription.enabled', label: '文字起こしを有効にする', type: 'checkbox' },
      { path: 'transcription.retention_min', label: '保持時間 (分)', type: 'number', min: 1,
        hint: 'この時間より古い文字起こしは自動削除されます' },
      { path: 'transcription.model_size', label: 'Whisper モデル', type: 'select',
        options: ['tiny', 'base', 'small', 'medium', 'large-v3', 'large-v3-turbo'],
        hint: '初回使用時に自動ダウンロードされます。CPU では base 前後が現実的です' },
      { path: 'transcription.language', label: '言語', type: 'select',
        options: ['ja', 'en', 'auto'] },
      { path: 'transcription.device', label: 'デバイス', type: 'select',
        options: ['cpu', 'cuda', 'auto'] },
      { path: 'transcription.compute_type', label: '計算精度', type: 'select',
        options: ['int8', 'int8_float16', 'float16', 'float32'] },
      { path: 'transcription.beam_size', label: 'ビーム幅', type: 'number', min: 1, max: 10 },
      { path: 'transcription.chunk_sec', label: '処理単位 (秒)', type: 'number', min: 5, max: 60,
        hint: 'この長さごとに音声をまとめて認識します' },
      { path: 'transcription.prompt_chars', label: 'プロンプトに渡す文字数', type: 'number', min: 0,
        hint: '直近の文字起こしをこの文字数まで LLM に渡します' },
    ],
  },
  {
    title: '自動投票',
    fields: [
      { path: 'betting.dry_run', label: 'ドライラン (実際には投票しない)', type: 'checkbox',
        hint: 'オフにすると実際にチャンネルポイントを賭けます。'
            + 'ログで賭け金と根拠を確認してからオフにしてください',
        danger: true },
      { path: 'betting.kelly_fraction', label: 'ケリー係数', type: 'number', step: 0.05,
        min: 0.01, max: 1,
        hint: 'フルケリーに掛ける割合。0.25 なら 1/4 ケリー' },
      { path: 'betting.max_bet_ratio', label: '1 回の最大賭け率', type: 'number', step: 0.01,
        min: 0.01, max: 1, hint: '保有ポイントに対する上限。0.05 なら 5%' },
      { path: 'betting.max_bet_points', label: '1 回の最大賭け金 (pt)', type: 'number', min: 1,
        max: 250000, hint: 'Twitch 側の上限は 250,000 pt です' },
      { path: 'betting.min_bet_points', label: '最低賭け金 (pt)', type: 'number', min: 1,
        hint: 'これを下回る算出額なら投票を見送ります' },
      { path: 'betting.min_edge', label: '最低エッジ', type: 'number', step: 0.01, min: 0,
        hint: '1 pt あたりの期待収益。0.05 なら +5% 未満は見送り' },
      { path: 'betting.min_confidence', label: '最低推定確率', type: 'number', step: 0.05,
        min: 0, max: 1, hint: '0 なら無効' },
      { path: 'betting.bet_lead_sec', label: '締め切り何秒前に投票するか', type: 'number', min: 1 },
    ],
  },
  {
    title: 'ログ',
    fields: [
      { path: 'log_retention_days', label: 'ログ保持日数', type: 'number', min: 1 },
    ],
  },
];

let current = {};

renderNav('settings');
document.getElementById('btn-save').addEventListener('click', save);
document.getElementById('btn-reload').addEventListener('click', () => load(true));
document.getElementById('btn-test-twitch').addEventListener('click', testTwitch);
document.getElementById('btn-test-llama').addEventListener('click', testLlama);

await load(false);

// -- form -------------------------------------------------------------------

function pick(obj, path) {
  return path.split('.').reduce((acc, key) => (acc == null ? undefined : acc[key]), obj);
}

function assign(obj, path, value) {
  const keys = path.split('.');
  let node = obj;
  for (const key of keys.slice(0, -1)) {
    node[key] = node[key] ?? {};
    node = node[key];
  }
  node[keys[keys.length - 1]] = value;
}

function buildForm() {
  const host = document.getElementById('sections');
  host.replaceChildren();

  for (const section of SECTIONS) {
    const body = el('div', { class: 'body' });
    for (const field of section.fields) {
      const value = pick(current, field.path);
      let input;

      if (field.type === 'checkbox') {
        input = el('input', { type: 'checkbox', id: field.path });
        input.checked = Boolean(value);
      } else if (field.type === 'select') {
        input = el('select', { id: field.path });
        for (const option of field.options) {
          input.append(el('option', { value: option, text: option }));
        }
        input.value = String(value ?? field.options[0]);
      } else {
        input = el('input', {
          type: field.type === 'password' ? 'password' : field.type,
          id: field.path,
          step: field.step,
          min: field.min,
          max: field.max,
        });
        input.value = value ?? '';
      }
      input.dataset.path = field.path;
      input.dataset.kind = field.type;
      input.addEventListener('input', markDirty);
      input.addEventListener('change', markDirty);

      const label = el('label', { for: field.path, text: field.label });
      if (field.danger) label.style.color = 'var(--danger)';

      const row = el('div', { class: 'field' }, label, input);
      if (field.hint) row.append(el('div', { class: 'hint', text: field.hint }));
      body.append(row);
    }
    host.append(el('section', {},
      el('h2', { text: section.title }),
      body,
    ));
  }
}

function collect() {
  const patch = {};
  for (const input of document.querySelectorAll('[data-path]')) {
    const { path, kind } = input.dataset;
    let value;
    if (kind === 'checkbox') value = input.checked;
    else if (kind === 'number') {
      if (input.value === '') continue;
      value = Number(input.value);
      if (!Number.isFinite(value)) continue;
    } else value = input.value;
    assign(patch, path, value);
  }
  return patch;
}

function markDirty() {
  document.getElementById('dirty').textContent = '未保存の変更があります';
}

// -- actions ----------------------------------------------------------------

async function load(notify) {
  try {
    current = await get('/api/settings');
    buildForm();
    document.getElementById('dirty').textContent = '';
    if (notify) toast('設定を再読込しました', 'ok');
  } catch (err) {
    toastError(err);
  }
}

async function save() {
  const button = document.getElementById('btn-save');
  button.disabled = true;
  try {
    const patch = collect();
    if (patch.betting && patch.betting.dry_run === false && current.betting?.dry_run !== false) {
      const ok = confirm(
        'ドライランを解除します。\n\n'
        + '以降、追跡中に条件を満たすと実際にチャンネルポイントが賭けられます。\n'
        + 'また、Twitch の非公開 API を自動操作することは Twitch の利用規約に抵触する'
        + '可能性があり、アカウント停止のリスクがあります。\n\n続行しますか?'
      );
      if (!ok) {
        document.getElementById('betting.dry_run').checked = true;
        return;
      }
    }
    current = await put('/api/settings', patch);
    buildForm();
    document.getElementById('dirty').textContent = '';
    toast('設定を保存しました', 'ok');
  } catch (err) {
    toastError(err);
  } finally {
    button.disabled = false;
  }
}

function showResult(payload) {
  const node = document.getElementById('test-result');
  node.style.display = '';
  node.textContent = JSON.stringify(payload, null, 2);
}

async function testTwitch() {
  const login = document.getElementById('test-login').value.trim();
  if (!login) {
    toast('テスト対象のチャンネル名を入力してください', 'err');
    return;
  }
  const button = document.getElementById('btn-test-twitch');
  button.disabled = true;
  try {
    const result = await post('/api/settings/test/twitch', { login });
    showResult(result);
    toast(result.ok ? 'Twitch 接続テスト成功' : '一部のクエリが失敗しました',
      result.ok ? 'ok' : 'err');
  } catch (err) {
    toastError(err);
  } finally {
    button.disabled = false;
  }
}

async function testLlama() {
  const button = document.getElementById('btn-test-llama');
  button.disabled = true;
  try {
    const result = await post('/api/settings/test/llama');
    showResult(result);
    toast(result.ok ? 'llama.cpp 設定は問題ありません' : result.problems.join(' / '),
      result.ok ? 'ok' : 'err');
  } catch (err) {
    toastError(err);
  } finally {
    button.disabled = false;
  }
}
