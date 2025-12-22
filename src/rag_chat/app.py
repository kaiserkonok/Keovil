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

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))

HOME_STORAGE = Path.home() / ".k_rag_storage"
DATA_DIR = HOME_STORAGE / "data"
DB_DIR = HOME_STORAGE / "database"
SETTINGS_FILE = HOME_STORAGE / "settings.json"
CHAT_DB = DB_DIR / "chat_history.db"  # Dedicated DB for Chat History

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Chat History Database Initialization
# ---------------------------------------------------------
def init_chat_db():
    conn = sqlite3.connect(CHAT_DB)
    curr = conn.cursor()

    # Sessions table (for the sidebar)
    curr.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            title TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Messages table (the actual content)
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
        except:
            return defaults
    return defaults


def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)


current_cfg = load_settings()

# ---------------------------------------------------------
# Engine Imports
# ---------------------------------------------------------
try:
    from rag_engine import CollegeRAG
except Exception as e:
    print(f"Import error for CollegeRAG: {e}")
    CollegeRAG = None

try:
    from agents.sql_agent import StructuredDataAgent
except Exception as e:
    print(f"Import error for StructuredDataAgent: {e}")
    StructuredDataAgent = None

# ---------------------------------------------------------
# Flask Init
# ---------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
FILES_DIR = DATA_DIR

# Init RAG engine
rag = None
if CollegeRAG:
    try:
        rag = CollegeRAG(
            str(DATA_DIR),
            llm_model=current_cfg["llm_model"],
            temperature=current_cfg["temperature"],
        )
        print(f"✅ RAG initialized")
    except Exception as e:
        print("❌ RAG init failed:", e)

# Init SQL Agent
sql_system = None
if StructuredDataAgent:
    try:
        sql_system = StructuredDataAgent()
        sql_system.start_monitoring()
        print(f"✅ SQL Agent initialized")
    except Exception as e:
        print("❌ SQL Agent init failed:", e)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def safe_rel_path(p: str | None) -> str:
    return (p or "").strip("/\\")


# ---------------------------------------------------------
# UI ROUTES
# ---------------------------------------------------------
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
@app.route("/api/chat/sessions", methods=["GET", "POST"])
def manage_sessions():
    conn = sqlite3.connect(CHAT_DB)
    if request.method == "POST":
        title = request.json.get("title", "New Chat")
        cur = conn.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        conn.commit()
        s_id = cur.lastrowid
        conn.close()
        return jsonify({"id": s_id, "title": title})

    cur = conn.execute("SELECT id, title FROM sessions ORDER BY created_at DESC")
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

    if not q:
        return jsonify({"error": "Empty query"}), 400

    conn = sqlite3.connect(CHAT_DB)

    # 1. Fetch history chronologically (Oldest First)
    rag_history = []
    if session_id:
        # Get last 10 messages in correct chronological order
        cur = conn.execute(
            "SELECT role, content FROM (SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10) ORDER BY timestamp ASC",
            (session_id,)
        )
        history_rows = cur.fetchall()
        # Ensure mapping: user -> "You", assistant -> "AI"
        rag_history = [("You" if r[0] == 'user' else "AI", r[1]) for r in history_rows]

    # 2. Generate Response (History is currently Turn 1 to N-1)
    ans = ""
    if rag:
        try:
            ans = rag.ask(q, chat_history=rag_history, stream=False)
        except Exception as e:
            conn.close()
            return jsonify({"error": f"RAG error: {e}"}), 500
    else:
        ans = "(RAG not initialized)"

    # 3. Create session if it doesn't exist
    if not session_id:
        title = q[:35] + "..." if len(q) > 35 else q
        cur = conn.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
        session_id = cur.lastrowid

    # 4. Save the current Turn (N) after AI responds
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

    return jsonify({"response": ans, "session_id": session_id})


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
        rag.llm = OllamaLLM(model=data["llm_model"], temperature=float(data["temperature"]))
    return jsonify({"ok": True})


@app.route("/api/explorer/files")
def list_files():
    rel_path = safe_rel_path(request.args.get("path"))
    target_dir = (FILES_DIR / rel_path).resolve()

    if not str(target_dir).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Directory not found"}), 404

    items = []
    for p in sorted(target_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        try:
            stat = p.stat()
            items.append({
                "name": p.name,
                "is_dir": p.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "path": str(p.relative_to(FILES_DIR))
            })
        except:
            continue
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
    except:
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

    if not str(old).startswith(str(FILES_DIR.resolve())) or not str(new).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400

    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/mkdir", methods=["POST"])
def mkdir():
    path = (FILES_DIR / safe_rel_path(request.json.get("path")) / request.json.get("name")).resolve()
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

    for i, file in enumerate(files):
        if not file.filename:
            continue
        rel_path = full_paths[i] if (full_paths and i < len(full_paths)) else file.filename
        dest = (parent_dir / rel_path).resolve()

        if not str(dest).startswith(str(FILES_DIR.resolve())):
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        file.save(dest)
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
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                entry = {
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "path": str(item.relative_to(FILES_DIR))
                }
                if item.is_dir():
                    entry["children"] = get_tree_items(item)
                res.append(entry)
        except:
            pass
        return res

    return jsonify({"tree": get_tree_items(FILES_DIR)})


@app.route("/api/sql_query", methods=["POST"])
def api_sql_query():
    data = request.json or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"output": "Please enter a query."}), 400

    response = sql_system.query(query) if sql_system else "SQL system not initialized."
    return jsonify({"output": response})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)