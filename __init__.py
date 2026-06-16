# core/__init__.py


<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Cyientist AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  font-family: 'Inter', system-ui, sans-serif;
  overflow: hidden;
  background: #ffffff;
}

#bg { display: none; }


/* ── LAYOUT ── */
#shell {
  position: relative; z-index: 1;
  height: 100dvh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
}

/* ── MAIN CARD ── */
#card {
  width: 100%;
  max-width: 780px;
  height: min(820px, calc(100dvh - 32px));
  background: rgba(252, 252, 252, 0.97);
  border-radius: 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.06);
  display: flex;
  overflow: hidden;
}

/* ── SIDEBAR ── */
#sidebar {
  width: 52px;
  flex-shrink: 0;
  background: #f5f5f5;
  border-right: 1px solid #e8e8e8;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px 0;
  gap: 6px;
}

.sb-logo {
  width: 32px; height: 32px;
  background: #0a0a0a;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 10px;
}

.sb-btn {
  width: 32px; height: 32px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; border: none; background: transparent;
  color: #999; transition: background 0.15s, color 0.15s;
}
.sb-btn:hover { background: #e8e8e8; color: #333; }
.sb-btn.active { background: #e8e8e8; color: #0a0a0a; }

.sb-spacer { flex: 1; }

/* ── MAIN CONTENT ── */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}

/* ── TOP BAR ── */
#topbar {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px 12px;
  border-bottom: 1px solid #ebebeb;
}

.brand-name {
  font-size: 14px;
  font-weight: 600;
  color: #0a0a0a;
  letter-spacing: -0.01em;
}

.top-actions { display: flex; gap: 6px; }

.icon-btn {
  width: 30px; height: 30px;
  border-radius: 8px;
  border: 1px solid #e2e2e2;
  background: white;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; color: #777;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.icon-btn:hover { background: #f5f5f5; color: #333; border-color: #ccc; }

/* ── MESSAGES AREA ── */
#messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 24px 8px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  scrollbar-width: thin;
  scrollbar-color: #ddd transparent;
}
#messages::-webkit-scrollbar { width: 4px; }
#messages::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }

/* ── EMPTY STATE ── */
#empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding-bottom: 60px;
  pointer-events: none;
  user-select: none;
}

.empty-icon {
  width: 52px; height: 52px;
  background: #ffffff;
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid #e4e4e4;
}

.empty-title {
  font-size: 22px; font-weight: 600;
  color: #0a0a0a; letter-spacing: -0.02em;
  text-align: center;
}

.empty-sub {
  font-size: 13px; color: #888; font-weight: 400;
  text-align: center; max-width: 280px; line-height: 1.5;
}

.empty-chips {
  display: flex; flex-wrap: wrap;
  gap: 8px; justify-content: center;
  margin-top: 8px;
  pointer-events: auto;
  max-width: 480px;
}

.chip {
  font-size: 12.5px; font-weight: 500; color: #333;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 999px;
  padding: 6px 14px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;
}
.chip:hover { background: #f5f5f5; border-color: #ccc; }

/* ── MESSAGES ── */
.msg {
  display: flex; gap: 10px; align-items: flex-start;
  animation: fadein 0.22s ease both;
}
@keyframes fadein {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.msg.user { flex-direction: row-reverse; }

.avatar {
  flex-shrink: 0;
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 600;
  margin-top: 2px;
}
.msg.user .avatar {
  background: #0a0a0a; color: #fff;
}
.msg.assistant .avatar {
  background: #ffffff; color: #333;
  border: 1px solid #e4e4e4;
}

.msg-body {
  display: flex; flex-direction: column; gap: 8px;
  flex: 1; min-width: 0;
}
.msg.user .msg-body { align-items: flex-end; }

.bubble {
  padding: 11px 15px;
  font-size: 14px; line-height: 1.7;
  letter-spacing: 0.005em;
  word-break: break-word;
  max-width: 78%;
}
.msg.user .bubble {
  background: #0a0a0a;
  color: #f5f5f5;
  border-radius: 16px 16px 4px 16px;
  max-width: 75%;
}
.msg.assistant .bubble {
  background: #f5f5f5;
  color: #1a1a1a;
  border: 1px solid #ebebeb;
  border-radius: 16px 16px 16px 4px;
  max-width: 90%;
}

/* ── TYPING ── */
.typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 2px 0; }
.typing-dots span {
  width: 5px; height: 5px; border-radius: 50%;
  background: #999;
  animation: tdot 1.2s infinite both;
}
.typing-dots span:nth-child(2) { animation-delay: 0.18s; }
.typing-dots span:nth-child(3) { animation-delay: 0.36s; }
@keyframes tdot {
  0%,80%,100% { transform: scale(0.7); opacity: 0.3; }
  40%          { transform: scale(1); opacity: 1; }
}

/* ── SOURCES ── */
.sources { display: flex; flex-direction: column; gap: 8px; }

.sources-toggle {
  align-self: flex-start;
  display: inline-flex; align-items: center; gap: 6px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 999px;
  padding: 5px 12px 5px 10px;
  font-size: 12px; font-weight: 500; color: #555;
  cursor: pointer; user-select: none;
  transition: background 0.15s, border-color 0.15s;
}
.sources-toggle:hover { background: #f5f5f5; border-color: #ccc; color: #222; }
.chevron {
  width: 10px; height: 10px; color: #999;
  transition: transform 0.2s ease;
}
.sources-toggle.open .chevron { transform: rotate(180deg); }

.sources-row {
  display: flex; gap: 10px;
  overflow-x: auto; overflow-y: hidden;
  padding: 2px 2px 8px;
  max-height: 0; opacity: 0; pointer-events: none;
  transition: max-height 0.3s ease, opacity 0.22s ease;
  scrollbar-width: thin; scrollbar-color: #ddd transparent;
}
.sources-row.open { max-height: 260px; opacity: 1; pointer-events: auto; }
.sources-row::-webkit-scrollbar { height: 3px; }
.sources-row::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }

.source-card {
  flex: 0 0 200px;
  display: flex; flex-direction: column; gap: 7px;
  padding: 12px 13px;
  background: white;
  border: 1px solid #e8e8e8;
  border-radius: 12px;
  transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
}
.source-card:hover {
  border-color: #ccc;
  box-shadow: 0 4px 16px rgba(0,0,0,0.07);
  transform: translateY(-2px);
}

.src-card-header {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.src-num {
  width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
  background: #ffffff; border: 1px solid #e4e4e4;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; color: #555;
}
.src-open {
  font-size: 10.5px; font-weight: 600;
  color: white; background: #0a0a0a;
  border-radius: 6px; padding: 3px 9px;
  text-decoration: none; flex-shrink: 0;
  transition: opacity 0.15s, transform 0.12s;
}
.src-open:hover { opacity: 0.75; transform: scale(1.04); }

.src-title {
  font-size: 12.5px; font-weight: 600; color: #111;
  line-height: 1.38;
  display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden;
}
.src-snippet {
  font-size: 11.5px; color: #777; line-height: 1.5;
  display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden;
}
.src-footer {
  display: flex; align-items: center; gap: 7px; margin-top: auto;
}
.src-score-bar {
  flex: 1; height: 2px; border-radius: 2px; background: #eee; overflow: hidden;
}
.src-score-fill {
  height: 100%; border-radius: 2px; background: #0a0a0a;
  transition: width 0.4s ease;
}
.src-score-pct { font-size: 10px; color: #aaa; flex-shrink: 0; }

/* ── INPUT AREA ── */
#input-area {
  flex-shrink: 0;
  padding: 12px 20px 16px;
  border-top: 1px solid #ebebeb;
}

#input-box {
  display: flex; align-items: flex-end; gap: 8px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 14px;
  padding: 10px 10px 10px 16px;
  transition: border-color 0.18s, box-shadow 0.18s;
}
#input-box:focus-within {
  border-color: #aaa;
  box-shadow: 0 0 0 3px rgba(0,0,0,0.06);
}

#prompt-input {
  flex: 1; background: transparent; border: none; outline: none; resize: none;
  color: #111; font-family: inherit; font-size: 14px; line-height: 1.6;
  max-height: 140px; min-height: 24px;
  overflow-y: auto; scrollbar-width: none;
  caret-color: #333;
}
#prompt-input::placeholder { color: #bbb; }
#prompt-input::-webkit-scrollbar { display: none; }

#send-btn {
  flex-shrink: 0; width: 34px; height: 34px;
  border-radius: 10px; border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  background: #0a0a0a; color: white;
  transition: opacity 0.15s, transform 0.12s;
}
#send-btn:hover { opacity: 0.8; transform: scale(1.05); }
#send-btn:active { transform: scale(0.95); }
#send-btn:disabled { opacity: 0.25; cursor: default; transform: none; }

.char-count {
  font-size: 11px; color: #bbb; text-align: right;
  padding: 4px 2px 0;
  font-variant-numeric: tabular-nums;
}

.err-text { color: #dc2626; font-size: 13.5px; }

/* ── RESPONSIVE ── */
@media (max-width: 600px) {
  #shell { padding: 0; }
  #card {
    border-radius: 0;
    height: 100dvh;
    max-width: 100%;
  }
  #sidebar { display: none; }
  .bubble { max-width: 88%; }
  .msg.assistant .bubble { max-width: 96%; }
  #messages { padding: 16px 14px 6px; }
  #input-area { padding: 10px 14px 14px; }
  #topbar { padding: 12px 14px 10px; }
  .empty-title { font-size: 18px; }
}
</style>
</head>
<body>

<div id="bg"></div>

<div id="shell">
  <div id="card">

    <!-- Sidebar -->
    <div id="sidebar">
      <div class="sb-logo">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L22 8.5V15.5L12 22L2 15.5V8.5L12 2Z" stroke="white" stroke-width="1.8" fill="none"/>
          <circle cx="12" cy="12" r="2.5" fill="white"/>
        </svg>
      </div>
      <button class="sb-btn active" title="Chat">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>
      <button class="sb-btn" title="History">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
        </svg>
      </button>
      <div class="sb-spacer"></div>
      <button class="sb-btn" title="Settings">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
      </button>
    </div>

    <!-- Main -->
    <div id="main">

      <!-- Topbar -->
      <div id="topbar">
        <span class="brand-name">Cyientist AI</span>
        <div class="top-actions">
          <button class="icon-btn" id="clear-btn" title="Clear chat">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- Messages -->
      <div id="messages">
        <div id="empty-state">
          <div class="empty-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L22 8.5V15.5L12 22L2 15.5V8.5L12 2Z" stroke="#333" stroke-width="1.6" fill="none"/>
              <circle cx="12" cy="12" r="2.5" fill="#333"/>
            </svg>
          </div>
          <p class="empty-title">Ask Cyientist AI</p>
          <p class="empty-sub">Ask about glossary terms, definitions, or technical concepts</p>
          <div class="empty-chips">
            <span class="chip" onclick="useChip(this)">What is 2MPC?</span>
            <span class="chip" onclick="useChip(this)">Explain postal process</span>
            <span class="chip" onclick="useChip(this)">What does 3D mean?</span>
            <span class="chip" onclick="useChip(this)">Map reference data</span>
          </div>
        </div>
      </div>

      <!-- Input -->
      <div id="input-area">
        <div id="input-box">
          <textarea id="prompt-input" rows="1" placeholder="Ask AI a question or make a request…" maxlength="2000"></textarea>
          <button id="send-btn" title="Send (Enter)">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
              <path d="M22 2L11 13" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </div>
        <div class="char-count" id="char-count">0 / 2000</div>
      </div>

    </div><!-- #main -->
  </div><!-- #card -->
</div><!-- #shell -->

<script>
const API_BASE  = 'http://localhost:8000';
const CHAT_PATH = '/chat';

let history = [];
let busy    = false;

const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('prompt-input');
const sendBtn     = document.getElementById('send-btn');
const clearBtn    = document.getElementById('clear-btn');
const charCount   = document.getElementById('char-count');

/* ── INPUT ── */
function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
  charCount.textContent = inputEl.value.length + ' / 2000';
}
inputEl.addEventListener('input', autoResize);
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

/* ── CHIPS ── */
function useChip(el) {
  inputEl.value = el.textContent;
  autoResize();
  inputEl.focus();
}

/* ── CLEAR ── */
clearBtn.addEventListener('click', () => {
  history = [];
  messagesEl.innerHTML = `
    <div id="empty-state">
      <div class="empty-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L22 8.5V15.5L12 22L2 15.5V8.5L12 2Z" stroke="#333" stroke-width="1.6" fill="none"/>
          <circle cx="12" cy="12" r="2.5" fill="#333"/>
        </svg>
      </div>
      <p class="empty-title">Ask Cyientist AI</p>
      <p class="empty-sub">Ask about glossary terms, definitions, or technical concepts</p>
      <div class="empty-chips">
        <span class="chip" onclick="useChip(this)">What is 2MPC?</span>
        <span class="chip" onclick="useChip(this)">Explain postal process</span>
        <span class="chip" onclick="useChip(this)">What does 3D mean?</span>
        <span class="chip" onclick="useChip(this)">Map reference data</span>
      </div>
    </div>`;
});

/* ── HELPERS ── */
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function removeEmpty() { document.getElementById('empty-state')?.remove(); }

/* ── USER MSG ── */
function addUserMessage(text) {
  removeEmpty();
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `
    <div class="avatar">U</div>
    <div class="msg-body"><div class="bubble">${escHtml(text)}</div></div>`;
  messagesEl.appendChild(el);
  scrollBottom();
}

/* ── ASSISTANT PLACEHOLDER ── */
function addAssistantPlaceholder() {
  removeEmpty();
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `
    <div class="avatar">AI</div>
    <div class="msg-body">
      <div class="bubble" id="live-bubble">
        <span class="typing-dots"><span></span><span></span><span></span></span>
      </div>
    </div>`;
  messagesEl.appendChild(el);
  scrollBottom();
  return el;
}

/* ── FINALIZE ── */
function finalizeAssistantMsg(msgEl, answer, sources) {
  const body   = msgEl.querySelector('.msg-body');
  const bubble = msgEl.querySelector('.bubble');
  bubble.removeAttribute('id');
  bubble.textContent = answer;

  if (sources && sources.length > 0) {
    const seen = new Set();
    const unique = sources.filter(s => {
      const key = (s.link && s.link.trim()) || s.title;
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });

    if (!unique.length) { scrollBottom(); return; }

    const toggleId = 'src-' + Math.random().toString(36).slice(2,9);

    const cardsHtml = unique.map((s, i) => {
      const score    = s.score != null ? Math.round(s.score * 100) : null;
      const linkHtml = s.link
        ? `<a class="src-open" href="${escHtml(s.link)}" target="_blank" rel="noopener">Open ↗</a>`
        : '';
      const footerHtml = score != null
        ? `<div class="src-footer">
             <div class="src-score-bar"><div class="src-score-fill" style="width:${score}%"></div></div>
             <span class="src-score-pct">${score}%</span>
           </div>` : '';
      return `
        <div class="source-card">
          <div class="src-card-header">
            <div class="src-num">${i+1}</div>
            ${linkHtml}
          </div>
          <div class="src-title">${escHtml(s.title || 'Document')}</div>
          ${s.snippet ? `<div class="src-snippet">${escHtml(s.snippet)}</div>` : ''}
          ${footerHtml}
        </div>`;
    }).join('');

    const srcEl = document.createElement('div');
    srcEl.className = 'sources';
    srcEl.innerHTML = `
      <div class="sources-toggle">
        <span>📄 ${unique.length} source${unique.length !== 1 ? 's' : ''}</span>
        <svg class="chevron" viewBox="0 0 12 8" fill="none">
          <path d="M1 1.5L6 6.5L11 1.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="sources-row" id="${toggleId}">${cardsHtml}</div>`;

    body.appendChild(srcEl);

    const tog = srcEl.querySelector('.sources-toggle');
    const row = srcEl.querySelector('.sources-row');
    tog.addEventListener('click', () => {
      tog.classList.toggle('open');
      row.classList.toggle('open');
      setTimeout(scrollBottom, 340);
    });
  }
  scrollBottom();
}

/* ── ERROR ── */
function showError(msgEl, err) {
  msgEl.querySelector('.bubble').innerHTML =
    `<span class="err-text">⚠ ${escHtml(String(err))}</span>`;
  scrollBottom();
}

/* ── SEND ── */
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
      body: JSON.stringify({ query, history }),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }

    const data    = await res.json();
    const answer  = data.answer ?? data.response ?? data.text ?? JSON.stringify(data);
    const srcs    = data.sources ?? data.citations ?? [];

    history.push({ role: 'assistant', content: answer });
    finalizeAssistantMsg(msgEl, answer, srcs);

  } catch (err) {
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