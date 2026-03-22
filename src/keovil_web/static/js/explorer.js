        let selectedPath = null;
        let currentFolder = '';
        let contextMenuTarget = null;
        let currentPdf = null;
        let currentPage = 1;
        let pdfScale = 1;

        const textExtensions = ['txt', 'js', 'html', 'css', 'json', 'xml', 'py', 'java', 'c', 'cpp', 'md', 'yml', 'yaml'];
        const pdfExtensions = ['pdf'];
        const docExtensions = ['docx', 'doc', 'ppt', 'pptx'];
        const spreadsheetExtensions = ['xlsx', 'xls', 'csv'];
        const dbExtensions = ['db', 'sqlite', 'sqlite3'];

        function getFileExtension(filename) { return filename.split('.').pop().toLowerCase(); }
        function isTextFile(filename) { return textExtensions.includes(getFileExtension(filename)); }
        function isPdfFile(filename) { return pdfExtensions.includes(getFileExtension(filename)); }
        function isDocFile(filename) { return docExtensions.includes(getFileExtension(filename)); }
        function isSpreadsheetFile(filename) { return spreadsheetExtensions.includes(getFileExtension(filename)); }
        function isDatabaseFile(filename) { return dbExtensions.includes(getFileExtension(filename)); }

        $(document).ready(function() {
            $('#file-tree').jstree({
                'core': {
                    'data': function(node, cb){
                        const path = node.id === '#' ? '' : node.id;
                        loadTreeData(path, cb);
                    },
                    'themes': {'name':'default','dots':true,'icons':true}
                },
                'plugins': ['wholerow']
            });

            $('#file-tree').on('select_node.jstree', function(e,data){
                const node = data.node;
                selectedPath = node.id;
                if (node.id === '#') {
                    currentFolder = ''; showPlaceholder(); updateFileInfo('Explorer', 'Root'); hideFileActions(); return;
                }
                const isDirectory = (node.original && node.original.is_dir) || node.text.endsWith('/');
                if(!isDirectory){
                    currentFolder = getParentPath(node.id); loadFileContent(node.id); showFileActions();
                    updateFileInfo(node.text.replace(/\/$/,''), node.id);
                } else {
                    currentFolder = node.id; showPlaceholder();
                    updateFileInfo(node.text.replace(/\/$/,''), node.id); hideFileActions();
                }
            });

            $('#file-tree').on('contextmenu.jstree', function(e){
                e.preventDefault();
                const node = $(e.target).closest('a');
                if(node.length){
                    const tree = $('#file-tree').jstree(true);
                    const nodeData = tree.get_node(node);
                    if(nodeData){ showContextMenu(e.pageX, e.pageY, nodeData); }
                }
            });

            $('#pdf-prev').click(() => renderPdfPage(currentPage - 1));
            $('#pdf-next').click(() => renderPdfPage(currentPage + 1));
            $('#pdf-zoom').change((e) => { pdfScale = parseFloat(e.target.value); renderPdfPage(currentPage); });
            $('#pdf-download').click(() => { if (selectedPath) downloadFile(selectedPath); });
            $('#sheet-download').click(() => { if (selectedPath) downloadFile(selectedPath); });
        });

        async function loadFileContent(path) {
            const filename = path.split('/').pop();
            hideAllViewers(); showPlaceholder();
            document.getElementById('fileName').textContent = filename;
            document.getElementById('filePath').textContent = path;
            document.getElementById('editorPlaceholder').innerHTML = `<div class="loading"><i class="fas fa-spinner fa-spin"></i> RUNNING_IO...</div>`;

            try {
                if (isTextFile(filename)) await loadTextFile(path);
                else if (isPdfFile(filename)) await loadPdfFile(path);
                else if (isDocFile(filename)) await loadDocFile(path);
                else if (isSpreadsheetFile(filename)) await loadSpreadsheetFile(path);
                else if (isDatabaseFile(filename)) await loadDatabaseFile(path);
                else showUnsupportedFile();
            } catch (error) { showUnsupportedFile(); }
        }

        async function loadTextFile(path) {
            try {
                const resp = await fetch(`/api/explorer/files/view?path=${encodeURIComponent(path)}`);
                const data = await resp.json();
                document.getElementById('editor').value = data.content || '';
                document.getElementById('editor').style.display = 'block';
                document.getElementById('editorPlaceholder').style.display = 'none';
                updateFileStats(data);
                document.getElementById('saveBtn').style.display = 'inline-flex';
            } catch (err) { showUnsupportedFile('Error loading text content'); }
        }

        async function loadPdfFile(path) {
            try {
                const pdfUrl = `/api/explorer/files/download?path=${encodeURIComponent(path)}`;
                const loadingTask = pdfjsLib.getDocument(pdfUrl);
                currentPdf = await loadingTask.promise;
                document.getElementById('pdf-viewer').style.display = 'flex';
                document.getElementById('editorPlaceholder').style.display = 'none';
                document.getElementById('pdf-total-pages').textContent = currentPdf.numPages;
                currentPage = 1; await renderPdfPage(1); hideSaveButton();
            } catch (err) { showUnsupportedFile('Error loading PDF'); }
        }

        async function renderPdfPage(pageNum) {
            if (!currentPdf || pageNum < 1 || pageNum > currentPdf.numPages) return;
            currentPage = pageNum; document.getElementById('pdf-current-page').textContent = currentPage;
            const page = await currentPdf.getPage(currentPage);
            const viewport = page.getViewport({ scale: pdfScale });
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height; canvas.width = viewport.width;
            await page.render({canvasContext: context, viewport: viewport}).promise;
            const container = document.getElementById('pdf-canvas-container'); container.innerHTML = '';
            container.appendChild(canvas);
        }

        // --- UPDATED SPREADSHEET (CSV/EXCEL) LOGIC ---
        async function loadSpreadsheetFile(path) {
            const filename = path.split('/').pop();
            const container = document.getElementById('sheet-table-container');
            const selector = document.getElementById('sheet-selector');

            try {
                // 1. Call the PREVIEW API instead of the DOWNLOAD API
                const resp = await fetch(`/api/explorer/files/preview?path=${encodeURIComponent(path)}`);
                const result = await resp.json();

                if (result.error) throw new Error(result.error);

                // 2. Handle Sheet Selection (for Excel)
                if (result.sheets && result.sheets.length > 0) {
                    selector.style.display = 'block';
                    selector.innerHTML = '';
                    result.sheets.forEach(name => {
                        const opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        selector.appendChild(opt);
                    });

                    // Re-fetch preview when the user changes the sheet
                    selector.onchange = async () => {
                        const sheetName = selector.value;
                        const sResp = await fetch(`/api/explorer/files/preview?path=${encodeURIComponent(path)}&sheet=${encodeURIComponent(sheetName)}`);
                        const sData = await sResp.json();
                        renderRawDataToTable(sData.data, 'sheet-table-container');
                    };
                } else {
                    selector.style.display = 'none';
                }

                // 3. Render the 100-row chunk returned by the server
                renderRawDataToTable(result.data, 'sheet-table-container');

                document.getElementById('spreadsheet-viewer').style.display = 'flex';
                document.getElementById('editorPlaceholder').style.display = 'none';
                hideSaveButton();
            } catch (err) {
                console.error(err);
                showUnsupportedFile('Error loading spreadsheet preview');
            }
        }

        // Unified rendering for Data Grids with a safety limit
        function renderRawDataToTable(data, containerId) {
            const container = document.getElementById(containerId);

            // We don't need to slice(0, 100) here anymore because the
            // SERVER already did that. Data IS the 100-row chunk.
            if (!data || data.length === 0) {
                container.innerHTML = '<div style="padding:20px; color:var(--muted);">No data found.</div>';
                return;
            }

            const totalRows = data.length;
            let html = '';

            // 1. Updated Table HTML with the class that matches our new CSS
            html += `<table class="data-grid-table">
                        <thead>
                            <tr>`;

            // Headers (Row 0)
            data[0].forEach(col => {
                html += `<th>${col || ''}</th>`;
            });
            html += `</tr></thead><tbody>`;

            // Body (Starting from Row 1)
            for (let i = 1; i < data.length; i++) {
                html += `<tr>`;
                data[i].forEach(cell => {
                    // Fix: Show a dash for empty strings from Pandas fillna("")
                    const val = (cell === "" || cell === null || cell === undefined) ? '-' : cell;
                    html += `<td>${val}</td>`;
                });
                html += `</tr>`;
            }

            html += `</tbody></table>`;

            // 2. Inject the HTML
            container.innerHTML = html;

            // 3. Reset Scroll (CRITICAL: This fixes the ugly alignment/view)
            container.scrollLeft = 0;
            container.scrollTop = 0;

            // 4. Update the stats label
            const statsEl = document.getElementById('sheet-stats');
            if (statsEl) {
                // Since the server sends 100 rows, we label it as a preview
                statsEl.textContent = `PREVIEW: ${totalRows} ROWS LOADED`;
            }
        }

        async function loadDocFile(path) {
            try {
                const response = await fetch(`/api/explorer/files/download?path=${encodeURIComponent(path)}`);
                const arrayBuffer = await response.arrayBuffer();
                const result = await mammoth.convertToHtml({ arrayBuffer: arrayBuffer });
                document.getElementById('doc-viewer').innerHTML = `<div class="doc-content" style="padding:40px; margin:20px auto; max-width:800px; line-height:1.5; background:#fff; color:#000;">${result.value}</div>`;
                document.getElementById('doc-viewer').style.display = 'block';
                document.getElementById('editorPlaceholder').style.display = 'none';
                hideSaveButton();
            } catch (err) { showUnsupportedFile(); }
        }

        // DATABASE LOADING LOGIC
        async function loadDatabaseFile(path) {
            try {
                const resp = await fetch(`/api/explorer/db/tables?path=${encodeURIComponent(path)}`);
                const data = await resp.json();
                if(data.error) throw new Error(data.error);

                const selector = document.getElementById('db-table-selector');
                selector.innerHTML = '';
                data.tables.forEach(table => {
                    const opt = document.createElement('option'); opt.value = table; opt.textContent = table; selector.appendChild(opt);
                });

                if (data.tables.length > 0) renderDatabaseTable(path, data.tables[0]);
                selector.onchange = () => renderDatabaseTable(path, selector.value);

                document.getElementById('database-viewer').style.display = 'flex';
                document.getElementById('editorPlaceholder').style.display = 'none';
                hideSaveButton();
            } catch (err) { showUnsupportedFile('Error reading database schema'); }
        }

        async function renderDatabaseTable(path, tableName) {
            const container = document.getElementById('db-grid-container');
            container.innerHTML = '<div style="padding:20px; color:var(--muted);"><i class="fas fa-spinner fa-spin"></i> Fetching records...</div>';
            try {
                const resp = await fetch(`/api/explorer/db/data?path=${encodeURIComponent(path)}&table=${encodeURIComponent(tableName)}`);
                const data = await resp.json();

                let html = `<table class="data-grid-table"><thead><tr>`;
                data.columns.forEach(col => html += `<th>${col}</th>`);
                html += `</tr></thead><tbody>`;
                data.rows.forEach(row => {
                    html += `<tr>`;
                    data.columns.forEach(col => {
                        const val = row[col] === null ? '<span style="color:#6e7681">NULL</span>' : row[col];
                        html += `<td>${val}</td>`;
                    });
                    html += `</tr>`;
                });
                html += `</tbody></table>`;
                container.innerHTML = html;
                document.getElementById('db-stats').textContent = `${data.rows.length} ROWS LOADED`;
            } catch (err) { container.innerHTML = `<div style="padding:20px; color:#ff7b72;">Failed to load table data.</div>`; }
        }

        function showUnsupportedFile(msg = 'Preview not available') {
            document.getElementById('editorPlaceholder').innerHTML = `<div style="text-align:center;"><i class="fas fa-eye-slash" style="font-size:40px; color:var(--muted);"></i><p style="margin-top:15px;">${msg}</p></div>`;
            hideSaveButton();
        }

        function hideAllViewers() {
            ['editor','pdf-viewer','doc-viewer','spreadsheet-viewer','database-viewer','editorPlaceholder']
            .forEach(id => {
                const el = document.getElementById(id);
                if(el) el.style.display = 'none';
            });
        }

        function hideSaveButton() { document.getElementById('saveBtn').style.display = 'none'; }
        function showFileActions() {
            document.getElementById('renameBtn').style.display = 'inline-flex';
            document.getElementById('deleteBtn').style.display = 'inline-flex';
        }

        function hideFileActions() {
            document.getElementById('renameBtn').style.display = 'none';
            document.getElementById('deleteBtn').style.display = 'none';
            document.getElementById('saveBtn').style.display = 'none';
        }

        function getParentPath(path){ if(!path) return ''; const parts=path.split('/'); parts.pop(); return parts.join('/'); }

        function showContextMenu(x,y,node){
            const menu=document.getElementById('contextMenu'); contextMenuTarget=node;
            menu.style.left=x+'px'; menu.style.top=y+'px'; menu.style.display='block';
            const isDir = (node.original && node.original.is_dir) || node.text.endsWith('/');
            contextMenuTarget.contextPath = isDir ? node.id : getParentPath(node.id);
            setTimeout(() => {
                const handler = (e) => { if(!menu.contains(e.target)){ menu.style.display='none'; document.removeEventListener('click', handler); }};
                document.addEventListener('click', handler);
            }, 100);
        }

        document.getElementById('contextMenu').addEventListener('click', function(e){
            const action = e.target.closest('[data-action]')?.getAttribute('data-action');
            if(!action || !contextMenuTarget) return;
            this.style.display='none';
            switch(action){
                case 'open': $('#file-tree').jstree('select_node', contextMenuTarget); break;
                case 'download': downloadFile(contextMenuTarget.id); break;
                case 'rename': renameItem(contextMenuTarget.id, contextMenuTarget.text.replace(/\/$/,'')); break;
                case 'delete': deleteItem(contextMenuTarget.id); break;
                case 'newFile': createNewFile(contextMenuTarget.contextPath); break;
                case 'newFolder': createNewFolder(contextMenuTarget.contextPath); break;
            }
        });

        async function loadTreeData(path, callback){
            try{
                const resp=await fetch(`/api/explorer/files${path ? `?path=${encodeURIComponent(path)}` : ''}`);
                const data=await resp.json();
                callback(data.files.map(item=>({id:item.path, text:item.name+(item.is_dir?'/':''), icon:item.is_dir?'jstree-folder':'jstree-file', children:item.is_dir, original:item})));
            }catch(err){ callback([]); }
        }

        async function createNewFolder(path=currentFolder){
            const name=prompt('Folder Name:'); if(!name) return;
            const resp=await fetch('/api/explorer/files/mkdir',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path, name})});
            if((await resp.json()).ok) refreshTree();
        }

        async function createNewFile(path=currentFolder){
            const name=prompt('File Name:'); if(!name) return;
            const fullPath = path ? `${path}/${name}` : name;
            const resp=await fetch('/api/explorer/files/save',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path: fullPath, content:''})});
            if((await resp.json()).ok) refreshTree();
        }

        async function saveFile(){
            if(!selectedPath) return;
            const content=document.getElementById('editor').value;
            const resp=await fetch('/api/explorer/files/save',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path:selectedPath, content})});
            if((await resp.json()).ok) {
                document.getElementById('fileStats').textContent='SAVED @ ' + new Date().toLocaleTimeString();
            }
        }

        async function deleteItem(path){
            if(!confirm(`Delete "${path}"?`)) return;
            const resp=await fetch('/api/explorer/files/delete',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path})});
            if((await resp.json()).ok) { 
                // Clear selection state after delete
                selectedPath = null;
                
                // If deleted folder was current folder, reset to root
                if (currentFolder && (path === currentFolder || path.startsWith(currentFolder + '/'))) {
                    currentFolder = '';
                }
                
                refreshTree(); 
                showPlaceholder(); 
                updateFileInfo('Explorer', 'Root');
                hideFileActions();
            }
        }

        async function renameItem(oldPath, oldName){
            const name=prompt('Rename to:', oldName); if(!name || name===oldName) return;
            const parent=getParentPath(oldPath);
            const resp=await fetch('/api/explorer/files/rename',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({old:oldPath, new: parent?`${parent}/${name}`:name})});
            if((await resp.json()).ok) refreshTree();
        }

        async function uploadBatch(files, path=currentFolder){
            if (files.length === 0) return;
            const fd = new FormData();
            fd.append('path', path);
            for (let i = 0; i < files.length; i++) {
                fd.append('file', files[i]);
                const relPath = files[i].webkitRelativePath || files[i].name;
                fd.append('full_paths', relPath);
            }

            try {
                const resp = await fetch('/api/explorer/files/upload', {method:'POST', body:fd});
                const result = await resp.json();
                if(result.ok) {
                    refreshTree();
                }
            } catch (e) {
                console.error("Upload failed", e);
            }
        }

        function downloadFile(path){ window.location.href=`/api/explorer/files/download?path=${encodeURIComponent(path)}`; }
        function updateFileInfo(name,path){ document.getElementById('fileName').textContent=name; document.getElementById('filePath').textContent=path; }
        function updateFileStats(data){
            const size = data.size ? (data.size/1024).toFixed(2)+' KB' : '0 KB';
            document.getElementById('fileStats').textContent=`${size} • Last Modified: ${new Date().toLocaleTimeString()}`;
            document.getElementById('charCount').textContent=`CHARS: ${data.content?.length || 0}`;
        }
        function showPlaceholder(){ hideAllViewers(); document.getElementById('editorPlaceholder').style.display='flex'; }
        function refreshTree(){ $('#file-tree').jstree(true).refresh(); }

        document.getElementById('refreshBtn').onclick = () => refreshTree();
        document.getElementById('newFolderBtn').onclick = () => createNewFolder();
        document.getElementById('newFileBtn').onclick = () => createNewFile();
        document.getElementById('saveBtn').onclick = () => saveFile();

        document.getElementById('uploadMainBtn').onclick = (e) => {
            e.stopPropagation();
            const menu = document.getElementById('uploadOptions');
            menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
        };

        window.onclick = () => { document.getElementById('uploadOptions').style.display = 'none'; };

        document.getElementById('optUploadFiles').onclick = () => document.getElementById('fileInput').click();
        document.getElementById('optUploadFolder').onclick = () => document.getElementById('folderInput').click();

        document.getElementById('fileInput').onchange = function(){ uploadBatch(this.files); this.value=''; };
        document.getElementById('folderInput').onchange = function(){ uploadBatch(this.files); this.value=''; };

        document.getElementById('renameBtn').onclick = () => selectedPath && renameItem(selectedPath, selectedPath.split('/').pop());
        document.getElementById('deleteBtn').onclick = () => selectedPath && deleteItem(selectedPath);
        document.addEventListener('keydown', e => { if(e.ctrlKey && e.key==='s'){ e.preventDefault(); saveFile(); }});

        // --- REAL-TIME INGESTION MONITORING ---
        // --- ELITE REAL-TIME MONITORING (SOCKET.IO) ---
// --- ELITE REAL-TIME MONITORING (SOCKET.IO) ---
// Adding a small delay to ensure the DOM is fully ready and the server is responsive
const socket = io({
    transports: ['websocket', 'polling'], // Fallback to polling if websocket fails
    reconnectionAttempts: 5
});

socket.on('connect', () => {
    console.log("✅ Connected to RTX 5060 Ti Monitoring System");
});

socket.on('system_status', function(data) {
    console.log("DEBUG Status Received:", data); // Check F12 Console for this!

    const card = $('#ingest-status-card');
    const dot = document.getElementById('status-dot');
    const bar = document.getElementById('ingest-progress-bar');
    const title = document.getElementById('status-title');
    const fileText = document.getElementById('current-ingest-file');

    if (data.rag && data.rag.state !== 'idle') {
        // Stop current animations and show the card
        card.stop(true, true).show();
        card.css('opacity', '1');

        // Handle Different States
        if (data.rag.state === 'waiting') {
            dot.className = 'status-indicator pulse-amber';
            title.innerText = "QUEUING BATCH...";
            fileText.innerText = data.rag.message || "Waiting for quiet period...";
            bar.style.width = "15%";
        }
        else if (data.rag.state === 'processing') {
            dot.className = 'status-indicator pulse-green';

            // UI Hint: If 'Purging' is in the text, change title to RED
            if (data.rag.current_file && data.rag.current_file.includes("Purging")) {
                title.innerText = "PURGING DATA...";
                dot.style.background = "#ff7b72"; // Red for deletion
            } else {
                title.innerText = "VECTORIZING (GPU)...";
                dot.style.background = "#3fb950"; // Green for processing
            }

            fileText.innerText = data.rag.current_file || "Processing documents...";
            bar.style.width = (data.rag.progress || 10) + "%";
        }
    } else {
        // Idle: Smooth fade out
        card.fadeOut(1000);
    }
});
