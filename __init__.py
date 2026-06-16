# core/__init__.py



<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Cyientist AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── THEME TOKENS: DARK (default) ── */
:root {
  --bg:               #0a0a0a;
  --surface:          #111111;
  --surface-alt:      #0d0d0d;
  --border:           rgba(255,255,255,0.06);
  --border-mid:       rgba(255,255,255,0.09);
  --border-hi:        rgba(255,255,255,0.1);
  --border-focus:     rgba(255,255,255,0.22);
  --shadow-focus:     rgba(255,255,255,0.04);
  --shadow-card:      rgba(0,0,0,0.5);

  --text-brand:       #f0f0f0;
  --text-body:        #e0e0e0;
  --text-sub:         #ccc;
  --text-muted:       #888;
  --text-dim:         #555;
  --text-faint:       #2e2e2e;
  --text-char:        #2a2a2a;

  --title-base:       #c8c8c8;
  --title-strong:     #ffffff;

  --chip-bg:          #111;
  --chip-color:       #666;
  --chip-border:      rgba(255,255,255,0.08);
  --chip-hover-bg:    #1a1a1a;
  --chip-hover-color: #ccc;
  --chip-hover-bdr:   rgba(255,255,255,0.16);

  --sb-hover-bg:      rgba(255,255,255,0.07);
  --sb-hover-color:   #aaa;
  --sb-active-bg:     rgba(255,255,255,0.1);
  --sb-active-color:  #fff;
  --sb-color:         #444;

  --ibtn-border:      rgba(255,255,255,0.09);
  --ibtn-color:       #555;
  --ibtn-hover-bg:    rgba(255,255,255,0.06);
  --ibtn-hover-color: #ccc;
  --ibtn-hover-bdr:   rgba(255,255,255,0.15);

  --logo-bg:          #ffffff;
  --logo-stroke:      #000000;

  --send-bg:          #ffffff;
  --send-color:       #000000;

  --bub-user-bg:      #1a1a1a;
  --bub-user-color:   #e8e8e8;
  --bub-user-bdr:     rgba(255,255,255,0.08);
  --bub-ai-bg:        #111;
  --bub-ai-color:     #ccc;
  --bub-ai-bdr:       rgba(255,255,255,0.07);

  --av-user-bg:       #ffffff;
  --av-user-color:    #000000;
  --av-ai-bg:         #151515;
  --av-ai-color:      #888;
  --av-ai-bdr:        rgba(255,255,255,0.1);

  --dot-color:        #3a3a3a;

  --src-bg:           #111;
  --src-border:       rgba(255,255,255,0.08);
  --src-hover-bdr:    rgba(255,255,255,0.16);
  --src-num-bg:       #1a1a1a;
  --src-num-bdr:      rgba(255,255,255,0.1);
  --src-num-color:    #666;
  --src-open-bg:      #ffffff;
  --src-open-color:   #000000;
  --src-title:        #ccc;
  --src-snippet:      #555;
  --src-bar-bg:       #1e1e1e;
  --src-bar-fill:     #ffffff;
  --src-pct:          #333;

  --stog-bg:          #111;
  --stog-bdr:         rgba(255,255,255,0.1);
  --stog-color:       #555;
  --stog-hover-bg:    #1a1a1a;
  --stog-hover-color: #aaa;

  --input-bg:         #111;
  --input-border:     rgba(255,255,255,0.1);
  --input-color:      #e0e0e0;
  --input-ph:         #2e2e2e;
  --scroll-thumb:     #1e1e1e;
}

/* ── THEME TOKENS: LIGHT ── */
html.light {
  --bg:               #ffffff;
  --surface:          #f5f5f5;
  --surface-alt:      #f5f5f5;
  --border:           rgba(0,0,0,0.07);
  --border-mid:       rgba(0,0,0,0.1);
  --border-hi:        rgba(0,0,0,0.1);
  --border-focus:     rgba(0,0,0,0.28);
  --shadow-focus:     rgba(0,0,0,0.06);
  --shadow-card:      rgba(0,0,0,0.12);

  --text-brand:       #0a0a0a;
  --text-body:        #111111;
  --text-sub:         #333;
  --text-muted:       #666;
  --text-dim:         #888;
  --text-faint:       #bbb;
  --text-char:        #bbb;

  --title-base:       #333;
  --title-strong:     #0a0a0a;

  --chip-bg:          #ffffff;
  --chip-color:       #555;
  --chip-border:      rgba(0,0,0,0.1);
  --chip-hover-bg:    #f0f0f0;
  --chip-hover-color: #111;
  --chip-hover-bdr:   rgba(0,0,0,0.18);

  --sb-hover-bg:      rgba(0,0,0,0.05);
  --sb-hover-color:   #333;
  --sb-active-bg:     rgba(0,0,0,0.08);
  --sb-active-color:  #0a0a0a;
  --sb-color:         #999;

  --ibtn-border:      rgba(0,0,0,0.1);
  --ibtn-color:       #888;
  --ibtn-hover-bg:    rgba(0,0,0,0.04);
  --ibtn-hover-color: #333;
  --ibtn-hover-bdr:   rgba(0,0,0,0.18);

  --logo-bg:          #0a0a0a;
  --logo-stroke:      #ffffff;

  --send-bg:          #0a0a0a;
  --send-color:       #ffffff;

  --bub-user-bg:      #0a0a0a;
  --bub-user-color:   #f5f5f5;
  --bub-user-bdr:     transparent;
  --bub-ai-bg:        #f5f5f5;
  --bub-ai-color:     #1a1a1a;
  --bub-ai-bdr:       rgba(0,0,0,0.07);

  --av-user-bg:       #0a0a0a;
  --av-user-color:    #ffffff;
  --av-ai-bg:         #ebebeb;
  --av-ai-color:      #555;
  --av-ai-bdr:        rgba(0,0,0,0.1);

  --dot-color:        #ccc;

  --src-bg:           #ffffff;
  --src-border:       rgba(0,0,0,0.08);
  --src-hover-bdr:    rgba(0,0,0,0.18);
  --src-num-bg:       #f5f5f5;
  --src-num-bdr:      rgba(0,0,0,0.1);
  --src-num-color:    #777;
  --src-open-bg:      #0a0a0a;
  --src-open-color:   #ffffff;
  --src-title:        #1a1a1a;
  --src-snippet:      #777;
  --src-bar-bg:       #e0e0e0;
  --src-bar-fill:     #0a0a0a;
  --src-pct:          #aaa;

  --stog-bg:          #ffffff;
  --stog-bdr:         rgba(0,0,0,0.1);
  --stog-color:       #777;
  --stog-hover-bg:    #f5f5f5;
  --stog-hover-color: #333;

  --input-bg:         #ffffff;
  --input-border:     rgba(0,0,0,0.12);
  --input-color:      #111;
  --input-ph:         #bbb;
  --scroll-thumb:     #ddd;
}

/* ── BASE ── */
html, body {
  height: 100%;
  font-family: 'Inter', system-ui, sans-serif;
  overflow: hidden;
  background: var(--bg);
  color: var(--text-body);
  transition: background 0.25s ease, color 0.25s ease;
}

/* ── LAYOUT ── */
#shell { position: relative; z-index: 1; height: 100dvh; display: flex; }

#card {
  width: 100%; height: 100dvh;
  background: var(--bg);
  display: flex; overflow: hidden;
  transition: background 0.25s ease;
}

/* ── SIDEBAR ── */
#sidebar {
  width: 52px; flex-shrink: 0;
  background: var(--surface-alt);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  align-items: center; padding: 16px 0; gap: 6px;
  transition: background 0.25s ease, border-color 0.25s ease;
}

.sb-logo {
  width: 32px; height: 32px;
  background: var(--logo-bg);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 10px;
  transition: background 0.25s ease;
}

.sb-btn {
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; border: none; background: transparent;
  color: var(--sb-color);
  transition: background 0.15s, color 0.15s;
}
.sb-btn:hover  { background: var(--sb-hover-bg);   color: var(--sb-hover-color); }
.sb-btn.active { background: var(--sb-active-bg);  color: var(--sb-active-color); }
.sb-spacer { flex: 1; }

/* ── MAIN ── */
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

/* ── TOPBAR ── */
#topbar {
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px 12px;
  border-bottom: 1px solid var(--border);
  transition: border-color 0.25s ease;
}

.brand-name {
  font-size: 14px; font-weight: 600;
  color: var(--text-brand); letter-spacing: -0.01em;
  transition: color 0.25s ease;
}

.top-actions { display: flex; gap: 6px; }

.icon-btn {
  width: 30px; height: 30px; border-radius: 8px;
  border: 1px solid var(--ibtn-border);
  background: transparent;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; color: var(--ibtn-color);
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.icon-btn:hover {
  background: var(--ibtn-hover-bg);
  color: var(--ibtn-hover-color);
  border-color: var(--ibtn-hover-bdr);
}

/* ── MESSAGES ── */
#messages {
  flex: 1; overflow-y: auto;
  padding: 32px 0 8px;
  display: flex; flex-direction: column; gap: 20px;
  scrollbar-width: thin;
  scrollbar-color: var(--scroll-thumb) transparent;
}
#messages::-webkit-scrollbar { width: 4px; }
#messages::-webkit-scrollbar-thumb { background: var(--scroll-thumb); border-radius: 2px; }

#messages > *, #empty-state {
  width: 100%; max-width: 720px;
  margin-left: auto; margin-right: auto;
  padding-left: 24px; padding-right: 24px;
}

/* ── EMPTY STATE ── */
#empty-state {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 18px; padding-bottom: 40px;
  pointer-events: none; user-select: none; text-align: center;
}

.empty-title {
  font-size: 38px; font-weight: 300;
  color: var(--title-base);
  letter-spacing: -0.04em; line-height: 1.12; max-width: 540px;
  transition: color 0.25s ease;
}
.empty-title strong { font-weight: 700; color: var(--title-strong); transition: color 0.25s ease; }

/* ── CHIPS ── */
.empty-chips {
  display: flex; flex-wrap: wrap; gap: 8px;
  justify-content: center; pointer-events: auto; max-width: 520px;
}

.chip {
  font-size: 12.5px; font-weight: 400;
  color: var(--chip-color);
  background: var(--chip-bg);
  border: 1px solid var(--chip-border);
  border-radius: 999px; padding: 6px 15px;
  cursor: pointer; white-space: nowrap;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.chip:hover {
  background: var(--chip-hover-bg);
  border-color: var(--chip-hover-bdr);
  color: var(--chip-hover-color);
}

/* ── MESSAGE BUBBLES ── */
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
  flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 600; margin-top: 2px;
  transition: background 0.25s ease, color 0.25s ease;
}
.msg.user      .avatar { background: var(--av-user-bg); color: var(--av-user-color); }
.msg.assistant .avatar { background: var(--av-ai-bg);   color: var(--av-ai-color); border: 1px solid var(--av-ai-bdr); }

.msg-body { display: flex; flex-direction: column; gap: 8px; flex: 1; min-width: 0; }
.msg.user .msg-body { align-items: flex-end; }

.bubble {
  padding: 11px 15px; font-size: 14px; line-height: 1.7;
  letter-spacing: 0.005em; word-break: break-word; max-width: 78%;
  transition: background 0.25s ease, color 0.25s ease, border-color 0.25s ease;
}
.msg.user .bubble {
  background: var(--bub-user-bg); color: var(--bub-user-color);
  border: 1px solid var(--bub-user-bdr);
  border-radius: 16px 16px 4px 16px; max-width: 75%;
}
.msg.assistant .bubble {
  background: var(--bub-ai-bg); color: var(--bub-ai-color);
  border: 1px solid var(--bub-ai-bdr);
  border-radius: 16px 16px 16px 4px; max-width: 90%;
}

/* ── TYPING ── */
.typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 2px 0; }
.typing-dots span {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--dot-color);
  animation: tdot 1.2s infinite both;
}
.typing-dots span:nth-child(2) { animation-delay: 0.18s; }
.typing-dots span:nth-child(3) { animation-delay: 0.36s; }
@keyframes tdot {
  0%,80%,100% { transform: scale(0.7); opacity: 0.3; }
  40%          { transform: scale(1);   opacity: 1; }
}

/* ── SOURCES ── */
.sources { display: flex; flex-direction: column; gap: 8px; }

.sources-toggle {
  align-self: flex-start;
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--stog-bg);
  border: 1px solid var(--stog-bdr);
  border-radius: 999px; padding: 5px 12px 5px 10px;
  font-size: 12px; font-weight: 500; color: var(--stog-color);
  cursor: pointer; user-select: none;
  transition: background 0.15s, color 0.15s;
}
.sources-toggle:hover { background: var(--stog-hover-bg); color: var(--stog-hover-color); }

.chevron { width: 10px; height: 10px; color: var(--text-dim); transition: transform 0.2s ease; }
.sources-toggle.open .chevron { transform: rotate(180deg); }

.sources-row {
  display: flex; gap: 10px;
  overflow-x: auto; overflow-y: hidden;
  padding: 2px 2px 8px;
  max-height: 0; opacity: 0; pointer-events: none;
  transition: max-height 0.3s ease, opacity 0.22s ease;
  scrollbar-width: thin; scrollbar-color: var(--scroll-thumb) transparent;
}
.sources-row.open { max-height: 260px; opacity: 1; pointer-events: auto; }

.source-card {
  flex: 0 0 200px; display: flex; flex-direction: column; gap: 7px;
  padding: 12px 13px;
  background: var(--src-bg); border: 1px solid var(--src-border);
  border-radius: 12px;
  transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
}
.source-card:hover {
  border-color: var(--src-hover-bdr);
  box-shadow: 0 4px 20px var(--shadow-card);
  transform: translateY(-2px);
}

.src-card-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.src-num {
  width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
  background: var(--src-num-bg); border: 1px solid var(--src-num-bdr);
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; color: var(--src-num-color);
}
.src-open {
  font-size: 10.5px; font-weight: 600;
  color: var(--src-open-color); background: var(--src-open-bg);
  border-radius: 6px; padding: 3px 9px;
  text-decoration: none; flex-shrink: 0;
  transition: opacity 0.15s;
}
.src-open:hover { opacity: 0.75; }
.src-title   { font-size: 12.5px; font-weight: 600; color: var(--src-title); line-height: 1.38; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.src-snippet { font-size: 11.5px; color: var(--src-snippet); line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.src-footer  { display: flex; align-items: center; gap: 7px; margin-top: auto; }
.src-score-bar  { flex: 1; height: 2px; border-radius: 2px; background: var(--src-bar-bg); overflow: hidden; }
.src-score-fill { height: 100%; border-radius: 2px; background: var(--src-bar-fill); }
.src-score-pct  { font-size: 10px; color: var(--src-pct); flex-shrink: 0; }

/* ── INPUT ── */
#input-area-wrap {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: var(--bg);
  transition: background 0.25s ease, border-color 0.25s ease;
}
#input-area { padding: 12px 24px 20px; max-width: 720px; width: 100%; margin: 0 auto; }

#input-box {
  display: flex; align-items: flex-end; gap: 8px;
  background: var(--input-bg);
  border: 1px solid var(--input-border);
  border-radius: 14px; padding: 10px 10px 10px 16px;
  transition: border-color 0.18s, box-shadow 0.18s, background 0.25s ease;
}
#input-box:focus-within {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px var(--shadow-focus);
}

#prompt-input {
  flex: 1; background: transparent; border: none; outline: none; resize: none;
  color: var(--input-color);
  font-family: inherit; font-size: 14px; line-height: 1.6;
  max-height: 140px; min-height: 24px;
  overflow-y: auto; scrollbar-width: none; caret-color: var(--text-muted);
}
#prompt-input::placeholder { color: var(--input-ph); }
#prompt-input::-webkit-scrollbar { display: none; }

#send-btn {
  flex-shrink: 0; width: 34px; height: 34px;
  border-radius: 10px; border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  background: var(--send-bg); color: var(--send-color);
  transition: opacity 0.15s, transform 0.12s, background 0.25s ease;
}
#send-btn:hover   { opacity: 0.82; transform: scale(1.05); }
#send-btn:active  { transform: scale(0.95); }
#send-btn:disabled { opacity: 0.18; cursor: default; transform: none; }

.char-count {
  font-size: 11px; color: var(--text-char); text-align: right;
  padding: 4px 2px 0; font-variant-numeric: tabular-nums;
  transition: color 0.25s ease;
}

.err-text { color: #ef4444; font-size: 13.5px; }

/* ── THEME TOGGLE ICON ── */
#theme-btn .sun  { display: none; }
#theme-btn .moon { display: block; }
html.light #theme-btn .sun  { display: block; }
html.light #theme-btn .moon { display: none; }

/* ── RESPONSIVE ── */
@media (max-width: 600px) {
  #sidebar { display: none; }
  .bubble { max-width: 88%; }
  .msg.assistant .bubble { max-width: 96%; }
  #messages { padding: 16px 14px 6px; }
  #input-area { padding: 10px 14px 14px; }
  #topbar { padding: 12px 14px 10px; }
  .empty-title { font-size: 26px; }
}
</style>
</head>
<body>

<div id="shell">
  <div id="card">

    <!-- Sidebar -->
    <div id="sidebar">
      <div class="sb-logo">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L22 8.5V15.5L12 22L2 15.5V8.5L12 2Z" stroke="var(--logo-stroke)" stroke-width="1.8" fill="none"/>
          <circle cx="12" cy="12" r="2.5" fill="var(--logo-stroke)"/>
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
          <circle cx="12" cy="12" r="3"/>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
      </button>
    </div>

    <!-- Main -->
    <div id="main">

      <!-- Topbar -->
      <div id="topbar">
        <span class="brand-name">Cyientist AI</span>
        <div class="top-actions">

          <!-- Theme toggle -->
          <button class="icon-btn" id="theme-btn" title="Toggle theme">
            <!-- Moon (shown in dark mode) -->
            <svg class="moon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
            <!-- Sun (shown in light mode) -->
            <svg class="sun" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1"  x2="12" y2="3"/>
              <line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22"  x2="5.64" y2="5.64"/>
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3"  y2="12"/>
              <line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          </button>

          <!-- Clear -->
          <button class="icon-btn" id="clear-btn" title="Clear chat">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6l-1 14H6L5 6"/>
              <path d="M10 11v6M14 11v6"/>
              <path d="M9 6V4h6v2"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- Messages -->
      <div id="messages">
        <div id="empty-state">
          <h1 class="empty-title">
            <strong>AI-Powered</strong> Chat Assistant<br>
            for <strong>Cyient</strong> Teams
          </h1>
          <div class="empty-chips">
            <span class="chip" onclick="useChip(this)">What is 2MPC?</span>
            <span class="chip" onclick="useChip(this)">Explain postal process</span>
            <span class="chip" onclick="useChip(this)">What does 3D mean?</span>
            <span class="chip" onclick="useChip(this)">Map reference data</span>
          </div>
        </div>
      </div>

      <!-- Input -->
      <div id="input-area-wrap">
        <div id="input-area">
          <div id="input-box">
            <textarea id="prompt-input" rows="1" placeholder="Ask AI a question or make a request…" maxlength="2000"></textarea>
            <button id="send-btn" title="Send (Enter)">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                <path d="M22 2L11 13" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
          <div class="char-count" id="char-count">0 / 2000</div>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
const API_BASE  = 'http://localhost:8000';
const CHAT_PATH = '/chat';

let history = [];
let busy    = false;

const messagesEl  = document.getElementById('messages');
const inputEl     = document.getElementById('prompt-input');
const sendBtn     = document.getElementById('send-btn');
const clearBtn    = document.getElementById('clear-btn');
const themeBtn    = document.getElementById('theme-btn');
const charCount   = document.getElementById('char-count');

/* ── THEME ── */
(function initTheme() {
  const saved = localStorage.getItem('cyientist-theme');
  if (saved === 'light') document.documentElement.classList.add('light');
})();

themeBtn.addEventListener('click', () => {
  const isLight = document.documentElement.classList.toggle('light');
  localStorage.setItem('cyientist-theme', isLight ? 'light' : 'dark');
});

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

/* ── EMPTY STATE HTML ── */
function emptyStateHTML() {
  return `
    <div id="empty-state">
      <h1 class="empty-title"><strong>AI-Powered</strong> Chat Assistant<br>for <strong>Cyient</strong> Teams</h1>
      <div class="empty-chips">
        <span class="chip" onclick="useChip(this)">What is 2MPC?</span>
        <span class="chip" onclick="useChip(this)">Explain postal process</span>
        <span class="chip" onclick="useChip(this)">What does 3D mean?</span>
        <span class="chip" onclick="useChip(this)">Map reference data</span>
      </div>
    </div>`;
}

/* ── CLEAR ── */
clearBtn.addEventListener('click', () => {
  history = [];
  messagesEl.innerHTML = emptyStateHTML();
});

/* ── HELPERS ── */
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function removeEmpty()  { document.getElementById('empty-state')?.remove(); }

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

    const toggleId  = 'src-' + Math.random().toString(36).slice(2, 9);
    const cardsHtml = unique.map((s, i) => {
      const score      = s.score != null ? Math.round(s.score * 100) : null;
      const linkHtml   = s.link
        ? `<a class="src-open" href="${escHtml(s.link)}" target="_blank" rel="noopener">Open ↗</a>`
        : '';
      const footerHtml = score != null
        ? `<div class="src-footer">
             <div class="src-score-bar"><div class="src-score-fill" style="width:${score}%"></div></div>
             <span class="src-score-pct">${score}%</span>
           </div>` : '';
      return `
        <div class="source-card">
          <div class="src-card-header"><div class="src-num">${i + 1}</div>${linkHtml}</div>
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

    const data   = await res.json();
    const answer = data.answer ?? data.response ?? data.text ?? JSON.stringify(data);
    const srcs   = data.sources ?? data.citations ?? [];

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
