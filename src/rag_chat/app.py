import eventlet
eventlet.monkey_patch()
import os
import sys
import shutil
import json
import sqlite3
from pathlib import Path
from typing import List, Dict
from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, stream_with_context
)
from flask_socketio import SocketIO, emit, join_room # <--- ADD THIS
from colorama import Fore, Style, init

init(autoreset=True)

# ---------------------------------------------------------
# Path Configurations (FIXED FOR ABSOLUTE 'src' IMPORTS)
# ---------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve() # K_RAG/src/rag_chat/app.py
SRC_DIR = CURRENT_FILE.parent.parent    # K_RAG/src/
PROJECT_ROOT = SRC_DIR.parent           # K_RAG/

# We add PROJECT_ROOT (K_RAG) so that 'from src.utils...' works
# We add SRC_DIR (src) so that 'from agents...' or 'import rag_engine' works
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(1, str(SRC_DIR))

print(f"DEBUG: Project Root is {PROJECT_ROOT}")

HOME_STORAGE = Path.home() / ".k_rag_storage"
DATA_DIR = HOME_STORAGE / "data"
DB_DIR = HOME_STORAGE / "database"
SETTINGS_FILE = HOME_STORAGE / "settings.json"
CHAT_DB = DB_DIR / "chat_history.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Global variable for all explorer routes
FILES_DIR = DATA_DIR

# ---------------------------------------------------------
# Chat History Database Initialization
# ---------------------------------------------------------
def init_chat_db():
    conn = sqlite3.connect(CHAT_DB)
    curr = conn.cursor()
    curr.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id TEXT,
            title TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    curr.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            session_id INTEGER, 
            role TEXT, 
            content TEXT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    ''')
    conn.commit()
    conn.close()

init_chat_db()

# ---------------------------------------------------------
# Settings Management
# ---------------------------------------------------------
def load_settings():
    defaults = {"llm_model": "qwen2.5:7b-instruct", "temperature": 0.8}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return {**defaults, **json.load(f)}
        except Exception:
            return defaults
    return defaults

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)

current_cfg = load_settings()

# ---------------------------------------------------------
# Engine Imports (Corrected for your tree structure)
# ---------------------------------------------------------
try:
    # rag_engine.py is inside /src/
    import rag_engine
    from rag_engine import CollegeRAG
    print("✅ Found rag_engine")
except ImportError as e:
    print(f"❌ Failed to import rag_engine: {e}")
    CollegeRAG = None

try:
    # sql_agent.py is inside /src/agents/
    from agents.sql_agent import StructuredDataAgent
    print("✅ Found StructuredDataAgent")
except ImportError as e:
    print(f"❌ Failed to import StructuredDataAgent: {e}")
    StructuredDataAgent = None

# ---------------------------------------------------------
# Flask Initialization
# ---------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # Allow up to 100GB
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
# Globals for engines
rag = None
sql_system = None

# Create a simple flag to prevent double-init within the same process
_ENGINES_INITIALIZED = False


def initialize_engines():
    global rag, sql_system, _ENGINES_INITIALIZED

    # If we already did this in THIS process, stop.
    if _ENGINES_INITIALIZED:
        return

    # Check if we are the 'Worker' child process
    # Flask-SocketIO sometimes uses 'true' (lowercase) or 'True'
    is_worker = os.environ.get("WERKZEUG_RUN_MAIN") in ['true', 'True']

    # If debug is OFF, we always init. If debug is ON, only the worker inits.
    if not app.debug or is_worker:
        print(f"{Fore.GREEN}🚀 [Worker] INITIALIZING ENGINES...{Style.RESET_ALL}")

        # 1. Init RAG
        if CollegeRAG and rag is None:
            try:
                rag = CollegeRAG(str(DATA_DIR), socketio=socketio)
                print("✅ RAG Registered")
            except Exception as e:
                print(f"❌ RAG Error: {e}")

        # 2. Init SQL
        if StructuredDataAgent and sql_system is None:
            try:
                sql_system = StructuredDataAgent(socketio=socketio)
                sql_system.start_monitoring()
                print("✅ SQL Agent Registered")
            except Exception as e:
                print(f"❌ SQL Error: {e}")

        _ENGINES_INITIALIZED = True
    else:
        print(f"{Fore.YELLOW}🛡️ [Watcher] Waiting for Worker process...{Style.RESET_ALL}")


# Instead of calling it at the top level, we call it INSIDE the first request
# or right before the app starts to ensure the environment is ready.
@app.before_request
def startup_init():
    initialize_engines()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def safe_rel_path(p: str | None) -> str:
    return (p or "").strip("/\\")

# ---------------------------------------------------------
# UI ROUTES
# ---------------------------------------------------------


@socketio.on('connect')
def handle_connect():
    # Automatically put every tab in its own private room based on its ID
    join_room(request.sid)
    print(f"Tab connected and private room created: {request.sid}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/cms")
def cms():
    return render_template("explorer.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/data-lab")
def data_lab():
    return render_template("data_chat.html")

# ---------------------------------------------------------
# CHAT SESSIONS & HISTORY API
# ---------------------------------------------------------
@app.route("/api/chat/sessions", methods=["GET"])
def manage_sessions():
    uid = request.headers.get("X-User-ID")
    if not uid:
        return jsonify([])
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.execute(
        "SELECT id, title FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (uid,)
    )
    sessions = [{"id": r[0], "title": r[1]} for r in cur.fetchall()]
    conn.close()
    return jsonify(sessions)

@app.route("/api/chat/history/<int:session_id>")
def get_chat_history_db(session_id):
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    msgs = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]
    conn.close()
    return jsonify(msgs)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    global rag
    data = request.json or {}
    q = (data.get("query") or "").strip()
    session_id = data.get("session_id")
    uid = request.headers.get("X-User-ID")

    if not q or not uid:
        return jsonify({"error": "Missing query or user ID"}), 400

    conn = sqlite3.connect(CHAT_DB)

    rag_history = []
    if session_id:
        cur = conn.execute(
            """SELECT role, content FROM (
                SELECT role, content, timestamp FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT 10
            ) ORDER BY timestamp ASC""",
            (session_id,)
        )
        history_rows = cur.fetchall()
        rag_history = [("You" if r[0] == 'user' else "AI", r[1]) for r in history_rows]

    socketio.emit('system_status', {"is_busy": True, "rag": {"state": "processing"}})

    ans = ""
    if rag:
        try:
            ans = rag.ask(q, chat_history=rag_history, stream=False)
        except Exception as e:
            conn.close()
            return jsonify({"error": str(e)}), 500
    else:
        socketio.emit('system_status', {"is_busy": False, "rag": {"state": "idle"}})
        ans = "(RAG not initialized)"

    if not session_id:
        title = q[:35] + "..." if len(q) > 35 else q
        cur = conn.execute(
            "INSERT INTO sessions (user_id, title) VALUES (?, ?)",
            (uid, title)
        )
        session_id = cur.lastrowid

    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
        (session_id, q)
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
        (session_id, ans)
    )
    conn.commit()
    conn.close()
    socketio.emit('system_status', {"is_busy": False, "rag": {"state": "idle"}})
    return jsonify({"response": ans, "session_id": session_id})

@app.route("/api/chat/sessions/delete", methods=["POST"])
def delete_session():
    data = request.json or {}
    session_id = data.get("session_id")
    uid = request.headers.get("X-User-ID")
    if not session_id or not uid:
        return jsonify({"error": "Missing data"}), 400
    conn = sqlite3.connect(CHAT_DB)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------
# SETTINGS & FILE EXPLORER API
# ---------------------------------------------------------
@app.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    global rag
    if request.method == "GET":
        return jsonify(load_settings())
    data = request.json
    save_settings(data)
    if rag:
        from langchain_ollama import OllamaLLM
        rag.llm = OllamaLLM(
            model=data["llm_model"],
            temperature=float(data["temperature"])
        )
    return jsonify({"ok": True})

@app.route("/api/explorer/files")
def list_files():
    rel_path = safe_rel_path(request.args.get("path"))
    target_dir = (FILES_DIR / rel_path).resolve()
    if not str(target_dir).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Not found"}), 404
    items = []
    sorted_items = sorted(target_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    for p in sorted_items:
        try:
            stat = p.stat()
            items.append({
                "name": p.name,
                "is_dir": p.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "path": str(p.relative_to(FILES_DIR))
            })
        except Exception: continue
    return jsonify({"files": items})

@app.route("/api/explorer/files/view")
def view_file():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())) or not path.exists():
        return jsonify({"error": "Not found"}), 404
    if path.is_dir():
        return jsonify({"content": "(Directory)"}), 200
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        text = "(Unable to read file)"
    return jsonify({"content": text, "name": path.name})

@app.route("/api/explorer/files/save", methods=["POST"])
def save_file():
    data = request.json or {}
    path = (FILES_DIR / safe_rel_path(data.get("path"))).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    with open(path, "w", encoding="utf-8") as f:
        f.write(data.get("content", ""))
    return jsonify({"ok": True})

@app.route("/api/explorer/files/delete", methods=["POST"])
def delete_file():
    path = (FILES_DIR / safe_rel_path(request.json.get("path"))).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())) or not path.exists():
        return jsonify({"error": "Not found"}), 404
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return jsonify({"ok": True})

@app.route("/api/explorer/files/rename", methods=["POST"])
def rename_file():
    data = request.json or {}
    old = (FILES_DIR / safe_rel_path(data.get("old"))).resolve()
    new = (FILES_DIR / safe_rel_path(data.get("new"))).resolve()
    if not str(old).startswith(str(FILES_DIR.resolve())) or \
       not str(new).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    return jsonify({"ok": True})

@app.route("/api/explorer/files/mkdir", methods=["POST"])
def mkdir():
    target = safe_rel_path(request.json.get("path"))
    name = request.json.get("name")
    path = (FILES_DIR / target / name).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    path.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/upload", methods=["POST"])
def upload_file():
    parent_dir = (FILES_DIR / safe_rel_path(request.form.get("path", ""))).resolve()
    files = request.files.getlist("file")
    full_paths = request.form.getlist("full_paths")

    if not str(parent_dir).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400

    print(f"{Fore.CYAN}📥 Upload started for {len(files)} items...{Style.RESET_ALL}")

    for i, file in enumerate(files):
        if not file.filename: continue

        rel_path = full_paths[i] if (full_paths and i < len(full_paths)) else file.filename
        dest = (parent_dir / rel_path).resolve()

        if not str(dest).startswith(str(FILES_DIR.resolve())): continue
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Log for GB files so you know it's working
        print(f"{Fore.YELLOW}💾 Saving: {file.filename} to {dest}...{Style.RESET_ALL}")

        # Using a buffer to save to avoid memory spikes
        file.save(dest)

        print(f"{Fore.GREEN}✅ Saved: {file.filename}{Style.RESET_ALL}")

    return jsonify({"ok": True})

@app.route("/api/explorer/files/download")
def download():
    path = (FILES_DIR / safe_rel_path(request.args.get("path"))).resolve()
    if not path.exists() or not str(path).startswith(str(FILES_DIR.resolve())):
        return "Not found", 404
    return send_file(path, as_attachment=True)

@app.route("/api/explorer/files/tree")
def list_files_tree():
    def get_tree_items(p: Path):
        res = []
        try:
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            for item in items:
                entry = {
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "path": str(item.relative_to(FILES_DIR))
                }
                if item.is_dir():
                    entry["children"] = get_tree_items(item)
                res.append(entry)
        except Exception: pass
        return res
    return jsonify({"tree": get_tree_items(FILES_DIR)})


@app.route("/api/sql_query", methods=["POST"])
def api_sql_query():
    # We don't need the sid here anymore for status!
    data = request.json or {}
    query = data.get("query", "")

    if not query:
        return jsonify({"output": "Please enter a query."}), 400

    try:
        # Just crunch the numbers and return them
        response = sql_system.query(query) if sql_system else "System Offline"
    except Exception as e:
        response = f"Error: {str(e)}"

    # NO SOCKET EMIT HERE. Keep the "Megaphone" silent.
    return jsonify({"output": response})


# ---------------------------------------------------------
# DATABASE EXPLORER API
# ---------------------------------------------------------
@app.route("/api/explorer/db/tables")
def get_db_tables():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()

    if not str(path).startswith(str(FILES_DIR.resolve())) or not path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return jsonify({"tables": tables})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/explorer/db/data")
def get_db_data():
    rel = safe_rel_path(request.args.get("path"))
    table_name = request.args.get("table")
    path = (FILES_DIR / rel).resolve()

    if not table_name:
        return jsonify({"error": "No table specified"}), 400

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row  # This allows us to get dictionary-like rows
        cursor = conn.cursor()

        # Use parameterized query for table name is tricky,
        # but since we get table names from the DB itself, it's safer.
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 100")
        rows = cursor.fetchall()

        if not rows:
            # Handle empty table but still return columns
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = [info[1] for info in cursor.fetchall()]
            data = []
        else:
            columns = list(rows[0].keys())
            data = [dict(row) for row in rows]

        conn.close()
        return jsonify({"columns": columns, "rows": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import pandas as pd


@app.route("/api/explorer/files/preview")
def preview_data_file():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()
    ext = path.suffix.lower()

    if not str(path).startswith(str(FILES_DIR.resolve())) or not path.exists():
        return jsonify({"error": "Not found"}), 404

    try:
        if ext == '.csv':
            df = pd.read_csv(path, nrows=100, on_bad_lines='skip', encoding_errors='replace')
            # Add this line to handle NaN
            df = df.fillna("")
            data = [df.columns.tolist()] + df.values.tolist()
            return jsonify({"data": data, "total_rows": "Large File"})

        elif ext in ['.xlsx', '.xls']:
            xl = pd.ExcelFile(path)
            sheet_name = request.args.get("sheet") or xl.sheet_names[0]
            df = pd.read_excel(path, sheet_name=sheet_name, nrows=100)

            # --- CRITICAL FIX HERE ---
            # Replace NaN/Infinity with empty strings so JSON remains valid
            df = df.fillna("")

            data = [df.columns.tolist()] + df.values.tolist()
            return jsonify({"data": data, "sheets": xl.sheet_names, "total_rows": "Large File"})

        return jsonify({"error": "Unsupported preview format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/status")
def get_ingest_status():
    """Returns the current status and triggers a broadcast."""
    rag_info = rag.get_status() if rag else {"state": "idle"}
    sql_is_syncing = sql_system.is_syncing if sql_system else False

    return jsonify({
        "is_busy": (rag_info.get("state") != "idle") or sql_is_syncing,
        "rag": rag_info,
        "sql_syncing": sql_is_syncing
    })

if __name__ == "__main__":
    # threaded=True is required for watchdog and web requests to run in parallel
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True, log_output=True)