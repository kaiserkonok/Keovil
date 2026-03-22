/**
 * Initialization & Identity
 */
const md = window.markdownit({
    html: true,
    highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try { return '<pre class="hljs"><code>' + hljs.highlight(str, { language: lang }).value + '</code></pre>'; } catch (__) {}
        }
        return '<pre class="hljs"><code>' + md.utils.escapeHtml(str) + '</code></pre>';
    }
});

const messages = document.getElementById('messages');
const inputBox = document.getElementById('inputBox');
const sendBtn = document.getElementById('sendBtn');
const historyList = document.getElementById('historyList');
const sessionLabel = document.getElementById('sessionLabel');
const deleteModal = document.getElementById('deleteModal');
const confirmDelBtn = document.getElementById('confirmDelBtn');

let currentSessionId = null;

let userId = localStorage.getItem('keo_user_id');
if (!userId) {
    userId = 'user_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
    localStorage.setItem('keo_user_id', userId);
}

async function apiFetch(url, options = {}) {
    const defaultHeaders = {
        'X-User-ID': userId,
        'Content-Type': 'application/json'
    };
    options.headers = { ...defaultHeaders, ...options.headers };
    return fetch(url, options);
}

/**
 * Modal Controls
 */
function closeModal() {
    deleteModal.style.display = 'none';
}

/**
 * Session Management
 */
async function loadSessions() {
    try {
        const res = await apiFetch('/api/chat/sessions');
        const sessions = await res.json();

        historyList.innerHTML = '';
        sessions.forEach(s => {
            const container = document.createElement('div');
            container.className = `session-item ${currentSessionId == s.id ? 'active' : ''}`;

            const item = document.createElement('div');
            item.className = 'session-title';
            item.textContent = s.title;
            item.onclick = () => switchSession(s.id);

            const delBtn = document.createElement('button');
            delBtn.className = 'delete-btn';
            delBtn.innerHTML = '&times;';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                deleteModal.style.display = 'flex';
                confirmDelBtn.onclick = () => executeDelete(s.id);
            };

            container.appendChild(item);
            container.appendChild(delBtn);
            historyList.appendChild(container);
        });
    } catch (err) {
        console.error("Failed to load sessions:", err);
    }
}

async function executeDelete(id) {
    closeModal();
    try {
        await apiFetch('/api/chat/sessions/delete', {
            method: 'POST',
            body: JSON.stringify({ session_id: id })
        });
        if (currentSessionId == id) {
            startNewChat();
        } else {
            loadSessions();
        }
    } catch (err) {
        console.error("Delete failed:", err);
    }
}

// Kept for backward compat if called elsewhere, but we use executeDelete now
async function deleteSession(id) {
    executeDelete(id);
}

async function switchSession(id) {
    currentSessionId = id;
    sessionLabel.textContent = `Session: #${id}`;
    messages.innerHTML = '';

    try {
        const res = await apiFetch(`/api/chat/history/${id}`);
        const history = await res.json();
        history.forEach(m => {
            if(m.role === 'user') appendUser(m.content);
            else appendAssistant(m.content);
        });
        loadSessions();
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

function startNewChat() {
    currentSessionId = null;
    messages.innerHTML = '';
    sessionLabel.textContent = "Direct Connection";
    inputBox.focus();
    loadSessions();
}

/**
 * UI Rendering
 */
function appendUser(text) {
    const el = document.createElement('div');
    el.className = 'bubble user';
    el.textContent = text;
    messages.appendChild(el);
    scrollToBottom();
}

function appendAssistant(text) {
    const el = document.createElement('div');
    el.className = 'bubble ai';
    el.innerHTML = md.render(text);
    messages.appendChild(el);
    el.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
    scrollToBottom();
}

function appendPlaceholder() {
    const el = document.createElement('div');
    el.className = 'bubble ai';
    el.innerHTML = `<span style="color:var(--accent); font-family:'Fira Code'; font-size:12px;">Analysing...</span>`;
    messages.appendChild(el);
    scrollToBottom();
    return el;
}

function scrollToBottom() {
    messages.parentElement.scrollTop = messages.parentElement.scrollHeight;
}

let isKeoBusy = false; // The "Pro" Lock

async function handleQuery(query) {
    if (isKeoBusy) return;

    isKeoBusy = true;
    $('#ingest-status-card').stop(true, true).fadeOut(200);
    // UI Feedback: Immediately lock the buttons
    inputBox.disabled = true;
    sendBtn.disabled = true;

    appendUser(query);
    const placeholder = appendPlaceholder();

    try {
        const res = await apiFetch('/api/chat', {
            method: 'POST',
            body: JSON.stringify({ query, session_id: currentSessionId })
        });

        const data = await res.json();
        if (data.error) {
            placeholder.textContent = "Error: " + data.error;
        } else {
            const isFirstInSession = !currentSessionId;
            currentSessionId = data.session_id;
            placeholder.remove();
            appendAssistant(data.response);
            if (isFirstInSession) {
                sessionLabel.textContent = `Session: #${currentSessionId}`;
                setTimeout(() => { loadSessions(); }, 250);
            }
        }
    } catch (err) {
        placeholder.textContent = "Connection lost. Please try again.";
    } finally {
        isKeoBusy = false;
        // ALWAYS re-enable here so you never get stuck
        inputBox.disabled = false;
        sendBtn.disabled = false;
        inputBox.focus();
    }
}

// Update the click handler to respect the lock
sendBtn.onclick = () => {
    if (isKeoBusy) return; // Don't even try if busy
    const q = inputBox.value.trim();
    if (!q) return;
    inputBox.value = '';
    handleQuery(q);
};

inputBox.onkeydown = e => {
    if (e.key === 'Enter') sendBtn.click();
};

loadSessions();


// Initialize Socket Connection
const socket = io();

socket.on('system_status', function(data) {
    const card = $('#ingest-status-card');
    const dot = document.getElementById('status-dot');
    const bar = document.getElementById('ingest-progress-bar');
    const title = document.getElementById('status-title');
    const detailText = document.getElementById('current-status-detail');

    if (!dot || !detailText) return;

    // 1. DETERMINE IF SYSTEM IS BUSY (Vectorizing or Answering)
    const isVectorizing = (data.reason === 'processing' || data.reason === 'waiting');

    // 2. TOGGLE INPUT LOCK
    // We disable the input if the AI is answering OR if the engine is vectorizing
    const shouldLockInput = isKeoBusy || isVectorizing;
    if (isVectorizing && !isKeoBusy) {
        inputBox.placeholder = "System updating... please wait.";
    } else if (!isKeoBusy) {
        inputBox.placeholder = "Ask Keo anything...";
    }

    if (isKeoBusy) {
        card.stop(true, true).fadeOut(200);
        return;
    }

    if (isVectorizing) {
        card.stop(true, true).fadeIn(200);

        if (data.reason === 'waiting') {
            dot.className = 'status-indicator pulse-amber';
            title.innerText = "QUEUING BATCH...";
            dot.style.background = "#e3b341";
        } else {
            dot.className = 'status-indicator pulse-green';
            if (data.rag.current_file && data.rag.current_file.includes("Purging")) {
                title.innerText = "PURGING DATA...";
                dot.style.background = "#ff7b72";
            } else {
                title.innerText = "VECTORIZING (GPU)...";
                dot.style.background = "#3fb950";
            }
        }

        detailText.innerText = data.rag.current_file || data.rag.message || "Working...";
        bar.style.width = (data.rag.progress || 10) + "%";
    }
    else {
        card.stop(true, true).fadeOut(800);
    }
});
