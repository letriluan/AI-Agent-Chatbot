'use strict';
const $ = id => document.getElementById(id);
const input = $('input'), chat = $('chat'), sendButton = $('send');
let busy = false, messageId = 0, cooldownUntil = 0, cooldownTimer;
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function updateInput() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 170) + 'px';
  $('counter').textContent = input.value.length.toLocaleString() + ' / 2,000';
  sendButton.disabled = busy || Date.now() < cooldownUntil || !input.value.trim();
}
function scrollToEnd() { $('scroll-area').scrollTop = $('scroll-area').scrollHeight; }
function setBusy(value) {
  busy = value;
  input.disabled = value;
  $('new-chat').disabled = value;
  document.querySelectorAll('.suggestion').forEach(button => { button.disabled = value; });
  sendButton.textContent = value ? 'Running…' : 'Send ↑';
  chat.setAttribute('aria-busy', String(value));
  updateInput();
}
function showCooldown(seconds) {
  clearInterval(cooldownTimer);
  cooldownUntil = Date.now() + Math.max(1, Math.min(seconds, 3600)) * 1000;
  const tick = () => {
    const remaining = Math.ceil((cooldownUntil - Date.now()) / 1000);
    $('rate-notice').hidden = remaining <= 0;
    $('rate-notice').textContent = remaining > 0 ? `A short pause. You can try again in ${remaining}s.` : '';
    if (remaining <= 0) clearInterval(cooldownTimer);
    updateInput();
  };
  tick(); cooldownTimer = setInterval(tick, 1000);
}
function addMessage(role, text) {
  const article = node('article', 'message ' + role);
  article.id = 'message-' + (++messageId);
  const header = node('div', 'message-header');
  header.append(node('span', 'message-mark', role === 'user' ? 'Y' : 'SQL'), node('span', '', role === 'user' ? 'You' : 'Assistant'));
  const content = node('div', 'message-content');
  if (text) content.append(node('p', '', text));
  article.append(header, content); chat.append(article); scrollToEnd();
  return {article, content};
}
// Only DOM text nodes are used for model output; never execute model HTML.
function appendInline(parent, text) {
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach(part => {
    if (part.startsWith('**') && part.endsWith('**')) parent.append(node('strong', '', part.slice(2, -2)));
    else if (part.startsWith('`') && part.endsWith('`')) parent.append(node('code', '', part.slice(1, -1)));
    else parent.append(document.createTextNode(part));
  });
}
function renderAnswer(parent, text) {
  const lines = text.split('\n');
  const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    if (lines[i].trim().startsWith('```')) {
      const code = [];
      while (++i < lines.length && !lines[i].trim().startsWith('```')) code.push(lines[i]);
      parent.append(node('pre', '', code.join('\n')));
    } else if (lines[i].includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      const table = node('table', 'answer-table'), head = node('thead'), row = node('tr');
      cells(lines[i]).forEach(value => { const th = node('th'); th.scope = 'col'; appendInline(th, value); row.append(th); });
      head.append(row); table.append(head); i += 2;
      const body = node('tbody');
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        const tr = node('tr');
        cells(lines[i]).forEach(value => { const td = node('td'); appendInline(td, value); tr.append(td); });
        body.append(tr); i++;
      }
      table.append(body); const wrap = node('div', 'table-wrap'); wrap.append(table); parent.append(wrap); i--;
    } else if (/^\s*(?:[-*]\s+|\d+\.\s+)/.test(lines[i])) {
      const ordered = /^\s*\d+\./.test(lines[i]);
      const list = node(ordered ? 'ol' : 'ul');
      const pattern = ordered ? /^\s*\d+\.\s+/ : /^\s*[-*]\s+/;
      while (i < lines.length && pattern.test(lines[i])) {
        const item = node('li'); appendInline(item, lines[i].replace(pattern, '')); list.append(item); i++;
      }
      parent.append(list); i--;
    } else {
      const paragraph = node('p'); appendInline(paragraph, lines[i].replace(/^#{1,6}\s+/, '')); parent.append(paragraph);
    }
  }
}
function copyButton(label, text) {
  const button = node('button', 'copy-button', label); button.type = 'button';
  button.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); button.textContent = 'Copied ✓'; }
    catch { button.textContent = 'Copy unavailable'; }
    setTimeout(() => { button.textContent = label; }, 1800);
  });
  return button;
}
async function ask(question) {
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/ask', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({question}), signal: controller.signal});
    if (!response.ok) {
      let payload; try { payload = await response.json(); } catch { /* Safe fallback. */ }
      const error = new Error(typeof payload?.detail?.message === 'string' ? payload.detail.message : 'We could not complete that request. Please try again.');
      if (response.status === 429) error.retryAfter = Math.max(1, Number(response.headers.get('Retry-After')) || 15);
      throw error;
    }
    return await response.json();
  } finally { clearTimeout(timeout); }
}
async function submitQuestion(event) {
  event?.preventDefault(); const question = input.value.trim();
  if (busy || Date.now() < cooldownUntil || !question || question.length > 2000) return;
  $('welcome').hidden = true;
  addMessage('user', question);
  input.value = ''; setBusy(true);
  const waiting = addMessage('bot'), loading = node('div', 'loading');
  loading.append(node('span'), node('span'), node('span'), node('em', '', 'Running query…')); waiting.content.append(loading);
  try {
    const data = await ask(question); waiting.content.replaceChildren();
    const answer = (data.answer || 'No answer was returned. Try adding a specific metric or filter.').replace(/^⚡\s*\*This answer was served from cache\.\*\s*/u, '');
    if (data.ok === false) {
      waiting.article.classList.add('error');
      const notice = node('div', 'result-notice'); renderAnswer(notice, answer); waiting.content.append(notice);
    } else renderAnswer(waiting.content, answer);
    if (data.image_b64) {
      const source = 'data:image/png;base64,' + data.image_b64, image = node('img', 'chart');
      image.src = source; image.alt = 'Chart for: ' + question; image.addEventListener('load', scrollToEnd);
      const download = node('a', 'download', '↓ Download chart'); download.href = source; download.download = 'ai-agent-chatbot-chart.png';
      waiting.content.append(image, download);
    }
    const footer = node('div', 'result-footer');
    if (data.ok !== false) {
      footer.append(node('span', '', data.cache_hit ? '↺ Served from cache' : '✓ Query complete'));
      const latency = data.metrics?.latency_ms;
      if (Number.isFinite(latency)) footer.append(node('span', '', latency < 1000 ? Math.round(latency) + ' ms' : (latency / 1000).toFixed(1) + ' s'));
    }
    footer.append(copyButton('Copy answer', answer)); waiting.content.append(footer);
    if (data.sql) {
      const details = node('details', 'sql-details'); details.append(node('summary', '', 'View SQL'), node('pre', '', data.sql), copyButton('Copy SQL', data.sql)); waiting.content.append(details);
    }
  } catch (error) {
    waiting.content.replaceChildren(); waiting.article.classList.add('error');
    waiting.content.append(node('p', 'result-notice', error.name === 'AbortError' ? 'This question is taking a little too long. Try a shorter date range or a simpler question.' : error.message || 'We could not connect. Please try again.'));
    if (error.retryAfter) showCooldown(error.retryAfter);
    const retry = node('button', 'copy-button', 'Edit this question ↗'); retry.type = 'button';
    retry.addEventListener('click', () => { if (!busy) { input.value = question; updateInput(); input.focus(); } }); waiting.content.append(retry);
  } finally { setBusy(false); input.focus(); scrollToEnd(); }
}
$('question-form').addEventListener('submit', submitQuestion);
input.addEventListener('input', updateInput);
input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitQuestion(); } });
function setMobileView(view) {
  document.body.dataset.view = view;
  $('show-project').setAttribute('aria-pressed', String(view === 'project'));
  $('show-demo').setAttribute('aria-pressed', String(view === 'demo'));
}
$('show-project').addEventListener('click', () => setMobileView('project'));
$('show-demo').addEventListener('click', () => setMobileView('demo'));
document.querySelector('.skip-link').addEventListener('click', () => setMobileView('demo'));
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  input.value = button.dataset.question;
  if (matchMedia('(max-width: 760px)').matches) setMobileView('demo');
  updateInput(); input.focus();
}));
function resetConversation() {
  if (busy) return;
  chat.replaceChildren();
  $('welcome').hidden = false; input.value = ''; updateInput(); input.focus(); $('scroll-area').scrollTop = 0;
}
$('new-chat').addEventListener('click', resetConversation);
async function checkConnection() {
  try { const response = await fetch('/ready', {signal: AbortSignal.timeout(5000)}); if (!response.ok) throw new Error('Not ready'); $('connection').replaceChildren(node('span', 'status-dot'), document.createTextNode('Dataset ready')); }
  catch { $('connection').classList.add('offline'); $('connection').replaceChildren(node('span', 'status-dot'), document.createTextNode('Dataset unavailable')); }
}
checkConnection(); updateInput();
