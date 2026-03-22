    const userInput = document.getElementById('userInput');
    let currentSessionId = null;
    const userId = localStorage.getItem('kevil_user_id') || 'anonymous_user';
    let cellCount = 1;

    let modalResolve;

    const socket = io();
    let isEngineBusy = false;

    socket.on('system_status', function(data) {
        const card = document.getElementById('ingest-status-card');
        const dot = document.getElementById('status-dot');
        const bar = document.getElementById('ingest-progress-bar');
        const title = document.getElementById('status-title');
        const detailText = document.getElementById('current-status-detail');

        // 1. DETERMINE THE STATE
        // Check if SQL is syncing OR if RAG is processing/waiting
        const isSqlSyncing = data.sql_syncing === true;
        const ragState = data.rag ? data.rag.state : 'idle';
        const isVectorizing = (ragState === 'processing' || ragState === 'waiting' || data.reason === 'processing');

        const isSystemBusy = isSqlSyncing || isVectorizing;

        // 2. LOCK INPUT
        // Lock if system is busy OR if the AI is currently generating an answer
        userInput.disabled = isSystemBusy || isEngineBusy;

        if (isSystemBusy) {
            card.style.display = 'block';
            userInput.placeholder = "System updating... please wait.";

            // 3. UI PERSONALIZATION (Color & Text)
            if (isSqlSyncing) {
                dot.className = 'status-indicator pulse-amber';
                title.innerText = "SQL ENGINE ACTIVE";
                detailText.innerText = data.rag?.current_file || "Updating Database Views...";
                bar.style.width = (data.rag?.progress || 100) + "%";
            }
            else if (isVectorizing) {
                const isPurging = data.rag?.current_file?.includes("Purging");

                if (ragState === 'waiting') {
                    dot.className = 'status-indicator pulse-amber';
                    title.innerText = "QUEUING BATCH...";
                    detailText.innerText = "Waiting for quiet period...";
                } else if (isPurging) {
                    dot.className = 'status-indicator'; // Red-ish via style
                    dot.style.background = "#ff7b72";
                    title.innerText = "PURGING DATA...";
                    detailText.innerText = data.rag.current_file;
                } else {
                    dot.className = 'status-indicator pulse-green';
                    title.innerText = "VECTORIZING (GPU)...";
                    detailText.innerText = data.rag?.current_file || "Processing Documents...";
                }
                bar.style.width = (data.rag?.progress || 0) + "%";
            }
        } else {
            if (!isEngineBusy) {
                card.style.display = 'none';
                userInput.placeholder = "Ask your data...";
            }
        }
    });

    function showConfirm(message) {
        document.querySelector('.modal-msg').innerText = message;
        document.getElementById('deleteModal').style.display = 'flex';
        return new Promise((resolve) => {
            modalResolve = resolve;
        });
    }

    function closeModal(result) {
        document.getElementById('deleteModal').style.display = 'none';
        if (modalResolve) modalResolve(result);
    }

    const md = window.markdownit({
        html: true,
        highlight: function (str, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return '<pre class="hljs"><code>' + hljs.highlight(str, { language: lang }).value + '</code></pre>';
            }
            return '';
        }
    });

    async function handleKey(e) {
        // Enter to submit
        if (e.key === 'Enter' && !e.shiftKey && userInput.value.trim() !== '' && !userInput.disabled) {
            e.preventDefault();
            const query = userInput.value;
            userInput.value = '';
            userInput.style.height = '56px';
            await addCell(query);
        }
    }
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 300) + 'px';
    });

    async function addCell(query) {
        const container = document.getElementById('notebook');
        const mainContent = document.querySelector('.main-content'); // Target the scrollable wrapper
        const currentId = cellCount++;
        isEngineBusy = true;
        userInput.disabled = true;

        const cellHtml = `
            <div class="cell" id="cell-${currentId}">
                <div class="cell-label">In [${currentId}]:</div>
                <div class="cell-content">
                    <div class="input-area">${query}</div>
                    <div class="output-area" id="out-${currentId}">
                        <span style="color:var(--accent); font-family:'Fira Code'; font-size:13px;">Executing Engine...</span>
                    </div>
                </div>
            </div>`;

        container.insertAdjacentHTML('beforeend', cellHtml);

        // Smooth scroll to the bottom of the main-content
        mainContent.scrollTo({ top: mainContent.scrollHeight, behavior: 'smooth' });

        try {
            const response = await fetch('/api/sql_query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-User-ID': userId },
                body: JSON.stringify({ query: query, session_id: currentSessionId })
            });
            const data = await response.json();

            if (data.session_id) currentSessionId = data.session_id;

            const outDiv = document.getElementById(`out-${currentId}`);
            outDiv.innerHTML = `
                <div style="color:var(--input-num); font-family:'Fira Code'; font-size:12px; margin-bottom:10px;">Out [${currentId}]:</div>
                <div class="markdown-body">${md.render(data.output)}</div>
            `;
            outDiv.querySelectorAll('pre code').forEach((el) => hljs.highlightElement(el));

            // Scroll again after content is rendered (tables can be tall!)
            setTimeout(() => {
                mainContent.scrollTo({ top: mainContent.scrollHeight, behavior: 'smooth' });
            }, 100);

            loadSessions();
        } catch (err) {
            document.getElementById(`out-${currentId}`).innerText = "Error: " + err.message;
        } finally {
            isEngineBusy = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    async function loadSessions() {
        const list = document.getElementById('session-list');
        try {
            const response = await fetch(`/api/chat/sessions?type=sql`, { headers: { 'X-User-ID': userId } });
            const sessions = await response.json();
            list.innerHTML = '';

            if (sessions.length === 0) {
                list.innerHTML = `
                    <div class="empty-history">
                        <div>No analysis yet</div>
                        <div style="font-size:11px; margin-top:5px; opacity:0.7;">Start by asking your data</div>
                    </div>
                `;
                return;
            }

            sessions.forEach((s, index) => {
                const div = document.createElement('div');
                div.className = `session-item ${currentSessionId == s.id ? 'active' : ''}`;
                div.onclick = () => loadHistory(s.id);

                div.innerHTML = `
                    <span class="session-title">${s.title}</span>
                    <span class="delete-btn" onclick="deleteSession('${s.id}', event)">✕</span>
                `;
                list.appendChild(div);
            });
        } catch (e) { console.error("Could not refresh sessions:", e); }
    }

    async function loadHistory(id) {
        currentSessionId = id;
        document.getElementById('session-status').innerText = `Session ID: ${id}`;
        const container = document.getElementById('notebook');
        const mainContent = document.querySelector('.main-content');

        const header = container.querySelector('.header-area').outerHTML;
        container.innerHTML = header;
        cellCount = 1;

        try {
            const response = await fetch(`/api/chat/history/${id}`);
            const messages = await response.json();
            let lastOutId = null;

            messages.forEach(msg => {
                if(msg.role === 'user') {
                    const idx = cellCount++;
                    lastOutId = `out-hist-${idx}`;
                    container.insertAdjacentHTML('beforeend', `
                        <div class="cell">
                            <div class="cell-label">In [${idx}]:</div>
                            <div class="cell-content">
                                <div class="input-area">${msg.content}</div>
                                <div class="output-area" id="${lastOutId}"></div>
                            </div>
                        </div>`);
                } else if(msg.role === 'assistant' && lastOutId) {
                    const out = document.getElementById(lastOutId);
                    out.innerHTML = `
                        <div style="color:var(--input-num); font-family:'Fira Code'; font-size:12px; margin-bottom:10px;">Out:</div>
                        <div class="markdown-body">${md.render(msg.content)}</div>
                    `;
                    out.querySelectorAll('pre code').forEach((el) => hljs.highlightElement(el));
                }
            });
            loadSessions();

            // Jump to bottom after history is populated
            setTimeout(() => {
                mainContent.scrollTo({ top: mainContent.scrollHeight, behavior: 'auto' });
            }, 50);

        } catch (e) { console.error(e); }
    }

    function startNewChat() {
        currentSessionId = null;
        cellCount = 1;
        const container = document.getElementById('notebook');
        const header = container.querySelector('.header-area').outerHTML;
        container.innerHTML = header;
        document.getElementById('session-status').innerText = "Stateless";
        loadSessions();
        userInput.focus();
    }

    async function deleteSession(id, event) {
        if (event) event.stopPropagation();

        // Use our sexy new modal instead of the poor window.confirm
        const confirmed = await showConfirm("Are you sure you want to permanently delete this analysis session?");
        if (!confirmed) return;

        try {
            const response = await fetch('/api/chat/sessions/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-ID': userId
                },
                body: JSON.stringify({ session_id: id })
            });

            const data = await response.json();
            if (data.ok) {
                if (currentSessionId === id) {
                    startNewChat();
                }
                loadSessions();
            }
        } catch (e) {
            console.error("Error during deletion:", e);
        }
    }

    // Optional: Function for a "Clear All" button in the header if you want one
    async function deleteCurrentSession() {
        if (currentSessionId) {
            await deleteSession(currentSessionId);
        } else {
            // Just a UI reset if it's a fresh, unsaved session
            startNewChat();
        }
    }

    window.onload = loadSessions;
