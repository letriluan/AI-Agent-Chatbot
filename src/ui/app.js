'use strict';
const $ = id => document.getElementById(id);
const input = $('input'), chat = $('chat'), sendButton = $('send');
let busy = false, messageId = 0;
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
  sendButton.disabled = busy || !input.value.trim();
}
function scrollToEnd() { $('scroll-area').scrollTop = $('scroll-area').scrollHeight; }
function setBusy(value) {
  busy = value;
  input.disabled = value;
  $('new-chat').disabled = value;
  $('new-chat-mobile').disabled = value;
  document.querySelectorAll('.suggestion').forEach(button => { button.disabled = value; });
  sendButton.textContent = value ? '·' : '↑';
  chat.setAttribute('aria-busy', String(value));
  updateInput();
}
function addMessage(role, text) {
  const article = node('article', 'message ' + role);
  article.id = 'message-' + (++messageId);
  const header = node('div', 'message-header');
  header.append(node('span', 'message-mark', role === 'user' ? 'TL' : '◈'), node('span', '', role === 'user' ? 'You' : 'Data assistant'));
  const content = node('div', 'message-content');
  if (text) content.append(node('p', '', text));
  article.append(header, content);
  chat.append(article);
  scrollToEnd();
  return {article, content};
}
// Render a small Markdown subset using DOM nodes only. Model text is never HTML.
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
    if (lines[i].includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      const table = node('table', 'answer-table');
      const head = node('thead'), row = node('tr');
      cells(lines[i]).forEach(value => { const th = node('th'); appendInline(th, value); row.append(th); });
      head.append(row); table.append(head); i += 2;
      const body = node('tbody');
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        const tr = node('tr');
        cells(lines[i]).forEach(value => { const td = node('td'); appendInline(td, value); tr.append(td); });
        body.append(tr); i++;
      }
      table.append(body); parent.append(table); i--;
    } else {
      const paragraph = node('p');
      appendInline(paragraph, lines[i].replace(/^#{1,6}\s+/, ''));
      parent.append(paragraph);
    }
  }
}
function addRecent(question, article) {
  const recent = $('recent');
  if (recent.querySelector('p')) recent.replaceChildren();
  const button = node('button', '', question);
  button.type = 'button'; button.title = question;
  button.addEventListener('click', () => article.scrollIntoView({behavior: 'smooth', block: 'start'}));
  recent.prepend(button);
}
async function ask(question) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}), signal: controller.signal});
    if (!response.ok) {
      let payload; try { payload = await response.json(); } catch { /* Use safe fallback. */ }
      const message = payload?.detail?.message;
      throw new Error(typeof message === 'string' ? message : 'The request could not be completed. Please try again.');
    }
    return await response.json();
  } finally { clearTimeout(timeout); }
}
async function submitQuestion(event) {
  event?.preventDefault();
  const question = input.value.trim();
  if (busy || !question || question.length > 2000) return;
  $('welcome').hidden = true;
  const user = addMessage('user', question);
  addRecent(question, user.article);
  input.value = ''; setBusy(true);
  const waiting = addMessage('bot');
  const loading = node('div', 'loading');
  loading.append(node('span'), node('span'), node('span'), node('em', '', 'Exploring your data…'));
  waiting.content.append(loading);
  try {
    const data = await ask(question);
    waiting.content.replaceChildren();
    if (data.ok === false) waiting.article.classList.add('error');
    renderAnswer(waiting.content, data.answer || 'No answer was returned. Please try rephrasing your question.');
    if (data.image_b64) {
      const source = 'data:image/png;base64,' + data.image_b64;
      const image = node('img', 'chart');
      image.src = source; image.alt = 'Chart for: ' + question;
      image.addEventListener('load', scrollToEnd);
      const download = node('a', 'download', '↓ Download chart');
      download.href = source; download.download = 'olist-chart.png';
      waiting.content.append(image, download);
    }
    const footer = node('div', 'result-footer');
    footer.append(node('span', '', data.cache_hit ? 'Served from cache' : 'Query complete'));
    const latency = data.metrics?.latency_ms;
    if (Number.isFinite(latency)) footer.append(node('span', '', latency < 1000 ? Math.round(latency) + ' ms' : (latency / 1000).toFixed(1) + ' s'));
    if (data.ok !== false) waiting.content.append(footer);
    if (data.sql) {
      const details = node('details', 'sql-details');
      details.append(node('summary', '', 'View SQL'), node('pre', '', data.sql));
      waiting.content.append(details);
    }
  } catch (error) {
    waiting.content.replaceChildren(); waiting.article.classList.add('error');
    waiting.content.append(node('p', '', error.name === 'AbortError' ? 'This request took too long. Please try a simpler question.' : error.message || 'Unable to connect. Please try again.'));
  } finally { setBusy(false); input.focus(); scrollToEnd(); }
}
$('question-form').addEventListener('submit', submitQuestion);
input.addEventListener('input', updateInput);
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitQuestion(); }
});
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  input.value = button.dataset.question; updateInput(); input.focus();
}));
function resetConversation() {
  if (busy) return;
  chat.replaceChildren(); $('recent').replaceChildren(node('p', '', 'Your questions will appear here.'));
  $('welcome').hidden = false; input.value = ''; updateInput(); input.focus(); $('scroll-area').scrollTop = 0;
}
$('new-chat').addEventListener('click', resetConversation);
$('new-chat-mobile').addEventListener('click', resetConversation);
async function checkConnection() {
  try {
    const response = await fetch('/ready', {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error('Not ready');
    $('connection').replaceChildren(node('span', 'status-dot'), document.createTextNode('Database ready'));
  } catch {
    $('connection').classList.add('offline');
    $('connection').replaceChildren(node('span', 'status-dot'), document.createTextNode('Database unavailable'));
  }
}
checkConnection();
updateInput();
