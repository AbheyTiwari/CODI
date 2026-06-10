# core/__init__.py

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RAG Chat</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-dark: #060912;
    --surface: rgba(255,255,255,0.04);
    --surface-hover: rgba(255,255,255,0.07);
    --border: rgba(255,255,255,0.08);
    --border-accent: rgba(168,85,247,0.25);
    --text-primary: #e8eaf0;
    --text-muted: #6b7280;
    --text-dim: #4b5563;
    --accent-blue: #4fa3e0;
    --accent-purple: #a855f7;
    --accent-pink: #e879a0;
    --glow-purple: rgba(168,85,247,0.25);
    --glow-blue: rgba(79,163,224,0.2);
    --source-bg: rgba(79,163,224,0.07);
    --source-border: rgba(79,163,224,0.2);
  }

  html, body { height: 100%; overflow: hidden; font-family: 'Inter', system-ui, sans-serif; background: var(--bg-dark); color: var(--text-primary); }

  /* ── Hex canvas ── */
  #hex-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }

  /* ── App shell ── */
  #app { position: relative; z-index: 1; height: 100dvh; display: flex; flex-direction: column; }

  /* ── Top bar ── */
  #topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px 10px;
    border-bottom: 1px solid var(--border);
    background: rgba(6,9,18,0.6);
    backdrop-filter: blur(12px);
    flex-shrink: 0;
  }
  #topbar .brand {
    display: flex; align-items: center; gap: 9px;
  }
  #topbar .brand-ring {
    width: 28px; height: 28px; border-radius: 50%;
    border: 1.5px solid rgba(168,85,247,0.5);
    box-shadow: 0 0 12px var(--glow-purple);
    display: flex; align-items: center; justify-content: center;
  }
  #topbar .brand-name { font-size: 13px; font-weight: 500; letter-spacing: 0.04em; color: var(--text-primary); }
  #topbar .brand-name span { color: var(--accent-purple); }

  #clear-btn {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text-muted); font-family: inherit; font-size: 12px;
    padding: 5px 12px; border-radius: 8px; cursor: pointer;
    display: flex; align-items: center; gap: 5px;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
  }
  #clear-btn:hover { background: var(--surface-hover); color: var(--accent-pink); border-color: rgba(232,121,160,0.3); }
  #clear-btn svg { width: 12px; height: 12px; }

  /* ── Messages ── */
  #messages {
    flex: 1; overflow-y: auto;
    padding: 20px 16px 8px;
    display: flex; flex-direction: column; gap: 20px;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent;
  }
  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

  /* ── Empty state ── */
  #empty-state {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    pointer-events: none; user-select: none;
  }
  #empty-state .ring {
    width: 60px; height: 60px; border-radius: 50%;
    border: 1.5px solid rgba(168,85,247,0.4);
    box-shadow: 0 0 30px var(--glow-purple), inset 0 0 20px rgba(168,85,247,0.08);
    display: flex; align-items: center; justify-content: center;
  }
  #empty-state .hint { font-size: 13px; color: var(--text-muted); letter-spacing: 0.03em; }
  #empty-state .sub  { font-size: 11px; color: var(--text-dim); letter-spacing: 0.02em; }

  /* ── Message row ── */
  .msg {
    display: flex; gap: 10px;
    max-width: 760px; width: 100%; align-self: center;
    animation: fadeUp 0.22s ease both;
  }
  @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

  .msg.user { flex-direction: row-reverse; align-self: flex-end; max-width: 75%; }

  .avatar {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; margin-top: 2px;
  }
  .msg.user .avatar     { background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); color: #fff; }
  .msg.assistant .avatar { background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)); color: #fff; box-shadow: 0 0 10px var(--glow-purple); }

  .msg-body { display: flex; flex-direction: column; gap: 8px; flex: 1; min-width: 0; }

  .bubble {
    padding: 11px 15px; border-radius: 14px;
    font-size: 14px; line-height: 1.7; letter-spacing: 0.01em;
    word-break: break-word;
  }
  .msg.user .bubble {
    background: linear-gradient(135deg, rgba(79,163,224,0.18), rgba(168,85,247,0.18));
    border: 1px solid rgba(168,85,247,0.22);
    border-bottom-right-radius: 4px;
  }
  .msg.assistant .bubble {
    background: var(--surface); border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  /* ── Sources panel ── */
  .sources {
    display: flex; flex-direction: column; gap: 6px;
  }
  .sources-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--accent-blue); opacity: 0.7;
    padding-left: 2px;
  }
  .source-item {
    background: var(--source-bg);
    border: 1px solid var(--source-border);
    border-radius: 8px; padding: 8px 12px;
    display: flex; flex-direction: column; gap: 3px;
    cursor: default;
    transition: background 0.15s, border-color 0.15s;
  }
  .source-item:hover { background: rgba(79,163,224,0.12); border-color: rgba(79,163,224,0.35); }
  .source-title {
    font-size: 12px; font-weight: 500; color: var(--accent-blue);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .source-snippet {
    font-size: 11.5px; color: var(--text-muted); line-height: 1.5;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .source-score {
    font-size: 10px; color: var(--text-dim);
  }

  /* ── Typing dots ── */
  .typing-dots span {
    display: inline-block; width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent-purple); margin: 0 2px;
    animation: blink 1.1s infinite both;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%,80%,100%{opacity:.2;transform:scale(.8);} 40%{opacity:1;transform:scale(1);} }

  /* ── Input bar ── */
  #input-area { padding: 10px 16px 20px; display: flex; justify-content: center; flex-shrink: 0; }

  #input-wrap {
    width: 100%; max-width: 760px;
    display: flex; align-items: flex-end; gap: 0;
    background: rgba(10,13,26,0.8);
    backdrop-filter: blur(20px) saturate(1.4);
    -webkit-backdrop-filter: blur(20px) saturate(1.4);
    border: 1px solid var(--border-accent);
    border-radius: 22px; padding: 10px 10px 10px 18px;
    box-shadow: 0 0 0 1px rgba(79,163,224,0.06), 0 8px 32px rgba(0,0,0,0.5), 0 0 40px rgba(168,85,247,0.07);
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  #input-wrap:focus-within {
    border-color: rgba(168,85,247,0.5);
    box-shadow: 0 0 0 1px rgba(79,163,224,0.12), 0 8px 32px rgba(0,0,0,0.5), 0 0 52px rgba(168,85,247,0.16);
  }

  #prompt-input {
    flex: 1; background: transparent; border: none; outline: none; resize: none;
    color: var(--text-primary); font-family: inherit; font-size: 14px; line-height: 1.6;
    max-height: 160px; min-height: 24px; overflow-y: auto; scrollbar-width: none;
    caret-color: var(--accent-purple);
  }
  #prompt-input::placeholder { color: var(--text-muted); }
  #prompt-input::-webkit-scrollbar { display: none; }

  #send-btn {
    flex-shrink: 0; width: 36px; height: 36px; border-radius: 50%;
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    box-shadow: 0 0 14px var(--glow-purple);
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    color: #fff;
  }
  #send-btn:hover  { transform: scale(1.08); box-shadow: 0 0 22px var(--glow-purple); }
  #send-btn:active { transform: scale(0.95); }
  #send-btn:disabled { opacity: 0.3; cursor: default; transform: none; }

  /* ── Divider ── */
  .day-divider {
    display: flex; align-items: center; gap: 10px;
    font-size: 10px; color: var(--text-dim); letter-spacing: 0.06em; text-transform: uppercase;
  }
  .day-divider::before, .day-divider::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
  }
</style>
</head>
<body>

<canvas id="hex-bg"></canvas>

<div id="app">

  <!-- Top bar -->
  <div id="topbar">
    <div class="brand">
      <div class="brand-ring">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" stroke="url(#tg)" stroke-width="1.5" fill="none"/>
          <defs><linearGradient id="tg" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4fa3e0"/><stop offset="1" stop-color="#a855f7"/>
          </linearGradient></defs>
        </svg>
      </div>
      <span class="brand-name">RAG<span>Chat</span></span>
    </div>
    <button id="clear-btn">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>
      </svg>
      Clear chat
    </button>
  </div>

  <!-- Messages -->
  <div id="messages">
    <div id="empty-state">
      <div class="ring">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
          <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" stroke="url(#eg)" stroke-width="1.4" fill="none"/>
          <defs><linearGradient id="eg" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4fa3e0"/><stop offset="1" stop-color="#a855f7"/>
          </linearGradient></defs>
        </svg>
      </div>
      <p class="hint">Ask anything — your documents are ready</p>
      <p class="sub">Sources will appear alongside each answer</p>
    </div>
  </div>

  <!-- Input -->
  <div id="input-area">
    <div id="input-wrap">
      <textarea id="prompt-input" rows="1" placeholder="Ask me anything…"></textarea>
      <button id="send-btn" title="Send (Enter)">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
          <path d="M22 2L11 13" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
  </div>

</div>

<script>
/* ─────────────────────────────────────────
   HEX CANVAS BACKGROUND
───────────────────────────────────────── */
(function () {
  const canvas = document.getElementById('hex-bg');
  const ctx = canvas.getContext('2d');
  const R = 13, GAP = 1.4;
  const SX = R * Math.sqrt(3) + GAP;
  const SY = R * 1.5 + GAP * 0.866;

  function resize() { canvas.width = innerWidth; canvas.height = innerHeight; draw(); }

  function hexPath(cx, cy) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i - Math.PI / 6;
      i === 0 ? ctx.moveTo(cx + R * Math.cos(a), cy + R * Math.sin(a))
              : ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
    }
    ctx.closePath();
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function cellColor(nx, ny) {
    const dB = Math.hypot(nx - 0.33, ny - 0.76);
    const dP = Math.hypot(nx - 0.63, ny - 0.71);
    const iB = Math.max(0, 1 - dB / 0.40);
    const iP = Math.max(0, 1 - dP / 0.32);
    const tot = Math.max(iB, iP);
    if (tot < 0.015) return null;
    const t = iP / (iB + iP + 1e-4);
    return {
      r: lerp(42, 228, t) | 0,
      g: lerp(128, 78, t) | 0,
      b: lerp(218, 178, t) | 0,
      a: Math.min(1, tot * 1.3)
    };
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cols = Math.ceil(canvas.width  / SX) + 2;
    const rows = Math.ceil(canvas.height / SY) + 2;
    for (let row = -1; row < rows; row++) {
      for (let col = -1; col < cols; col++) {
        const cx = col * SX + (row % 2 === 0 ? 0 : SX / 2);
        const cy = row * SY;
        const c  = cellColor(cx / canvas.width, cy / canvas.height);
        hexPath(cx, cy);
        if (c) {
          ctx.fillStyle   = `rgba(${c.r},${c.g},${c.b},${(c.a * 0.30).toFixed(3)})`;
          ctx.fill();
          ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${c.a.toFixed(3)})`;
          ctx.lineWidth   = 0.8;
        } else {
          ctx.strokeStyle = 'rgba(255,255,255,0.035)';
          ctx.lineWidth   = 0.5;
        }
        ctx.stroke();
      }
    }
  }

  window.addEventListener('resize', resize);
  resize();
})();

/* ─────────────────────────────────────────
   CONFIG — swap these when your API is ready
───────────────────────────────────────── */
const API_BASE  = 'http://localhost:8000';   // your FastAPI base URL
const CHAT_PATH = '/chat';                   // POST endpoint

/*
  Expected request body:
  {
    "query":   "user message",
    "history": [{ "role": "user"|"assistant", "content": "..." }, ...]
  }

  Expected response body (non-streaming JSON):
  {
    "answer": "...",
    "sources": [
      { "title": "doc.pdf", "snippet": "...", "score": 0.91 },
      ...
    ]
  }

  To switch to streaming, see the streamChat() function below.
*/

/* ─────────────────────────────────────────
   STATE
───────────────────────────────────────── */
let history = [];   // [{ role, content }]
let busy    = false;

/* ─────────────────────────────────────────
   DOM REFS
───────────────────────────────────────── */
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('prompt-input');
const sendBtn    = document.getElementById('send-btn');
const clearBtn   = document.getElementById('clear-btn');

/* ─────────────────────────────────────────
   INPUT HELPERS
───────────────────────────────────────── */
function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}
inputEl.addEventListener('input', autoResize);
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

/* ─────────────────────────────────────────
   CLEAR
───────────────────────────────────────── */
clearBtn.addEventListener('click', () => {
  history = [];
  messagesEl.innerHTML = `
    <div id="empty-state">
      <div class="ring">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
          <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" stroke="url(#eg2)" stroke-width="1.4" fill="none"/>
          <defs><linearGradient id="eg2" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4fa3e0"/><stop offset="1" stop-color="#a855f7"/>
          </linearGradient></defs>
        </svg>
      </div>
      <p class="hint">Ask anything — your documents are ready</p>
      <p class="sub">Sources will appear alongside each answer</p>
    </div>`;
});

/* ─────────────────────────────────────────
   RENDER HELPERS
───────────────────────────────────────── */
function removeEmpty() {
  const e = document.getElementById('empty-state');
  if (e) e.remove();
}

function addUserMessage(text) {
  removeEmpty();
  const msg = document.createElement('div');
  msg.className = 'msg user';
  msg.innerHTML = `
    <div class="avatar">U</div>
    <div class="msg-body">
      <div class="bubble">${escHtml(text)}</div>
    </div>`;
  messagesEl.appendChild(msg);
  scrollBottom();
}

function addAssistantPlaceholder() {
  removeEmpty();
  const msg = document.createElement('div');
  msg.className = 'msg assistant';
  msg.innerHTML = `
    <div class="avatar">AI</div>
    <div class="msg-body">
      <div class="bubble" id="live-bubble">
        <span class="typing-dots"><span></span><span></span><span></span></span>
      </div>
    </div>`;
  messagesEl.appendChild(msg);
  scrollBottom();
  return msg;
}

function finalizeAssistantMsg(msgEl, answer, sources) {
  const body   = msgEl.querySelector('.msg-body');
  const bubble = msgEl.querySelector('.bubble');
  bubble.removeAttribute('id');
  bubble.textContent = answer;

  if (sources && sources.length > 0) {
    const sourcesEl = document.createElement('div');
    sourcesEl.className = 'sources';
    sourcesEl.innerHTML = `<div class="sources-label">Retrieved sources</div>` +
      sources.map(s => `
        <div class="source-item">
          <div class="source-title">${escHtml(s.title || 'Document')}</div>
          <div class="source-snippet">${escHtml(s.snippet || '')}</div>
          ${s.score != null ? `<div class="source-score">Relevance: ${(s.score * 100).toFixed(1)}%</div>` : ''}
        </div>`).join('');
    body.appendChild(sourcesEl);
  }
  scrollBottom();
}

function showError(msgEl, err) {
  const bubble = msgEl.querySelector('.bubble');
  bubble.innerHTML = `<span style="color:#e879a0">⚠ ${escHtml(String(err))}</span>`;
  scrollBottom();
}

function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ─────────────────────────────────────────
   SEND
───────────────────────────────────────── */
async function send() {
  const query = inputEl.value.trim();
  if (!query || busy) return;

  busy = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  autoResize();

  addUserMessage(query);
  history.push({ role: 'user', content: query });

  const msgEl = addAssistantPlaceholder();

  try {
    const res = await fetch(API_BASE + CHAT_PATH, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history })
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }

    const data = await res.json();
    /*
      Adjust these keys to match your actual FastAPI response shape:
        data.answer  → the LLM answer string
        data.sources → array of { title, snippet, score }
    */
    const answer  = data.answer  ?? data.response ?? data.text ?? JSON.stringify(data);
    const sources = data.sources ?? data.citations ?? [];

    history.push({ role: 'assistant', content: answer });
    finalizeAssistantMsg(msgEl, answer, sources);

  } catch (err) {
    // Remove the failed turn from history
    history.pop();
    showError(msgEl, err.message || err);
  }

  busy = false;
  sendBtn.disabled = false;
  inputEl.focus();
}
</script>
</body>
</html>
