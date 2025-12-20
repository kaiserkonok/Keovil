from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, stream_with_context
)
import sys
from pathlib import Path
import os
import shutil
from typing import List, Dict

# ---------------------------------------------------------
# Fix Python path so rag_engine can be imported
# ---------------------------------------------------------
ROOT = Path(__file__).parent.parent  # /src
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))  # project root

try:
    from rag_engine import CollegeRAG
except Exception:
    CollegeRAG = None

from agents.sql_agent import StructuredDataAgent

# ---------------------------------------------------------
# Flask Init
# ---------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "test_data"
FILES_DIR = DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Init RAG engine
# ---------------------------------------------------------
rag = None
if CollegeRAG:
    try:
        rag = CollegeRAG(str(DATA_DIR))
        print("RAG initialized OK")
    except Exception as e:
        print("RAG init failed:", e)

# Initialize the SQL Agent
DB_FILE = str(PROJECT_ROOT / "database" / "main.db")
SQL_DATA_DIR = str(PROJECT_ROOT / "test_data")
sql_system = StructuredDataAgent(DB_FILE, SQL_DATA_DIR)
sql_system.start_monitoring()

def safe_rel_path(p: str | None) -> str:
    return (p or "").strip("/\\")

def convert_client_history_to_rag_history(client_history: List[Dict[str, str]]):
    rag_history = []
    for item in client_history or []:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            rag_history.append(("You", content))
        elif role == "assistant":
            rag_history.append(("AI", content))
    return rag_history

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html")

@app.route("/cms")
def cms():
    return render_template("explorer.html")

# ---------------------------------------------------------
# FILE EXPLORER API
# ---------------------------------------------------------

@app.route("/api/explorer/files")
def list_files():
    rel_path = safe_rel_path(request.args.get("path"))
    target_dir = (FILES_DIR / rel_path).resolve()
    if not str(target_dir).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Directory not found"}), 404

    items = []
    for p in sorted(target_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        stat = p.stat()
        items.append({
            "name": p.name,
            "is_dir": p.is_dir(),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "path": str(p.relative_to(FILES_DIR))
        })
    return jsonify({"files": items})

@app.route("/api/explorer/files/view")
def view_file():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR)) or not path.exists():
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
    rel = data.get("path")
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    open(path, "w", encoding="utf-8").write(data.get("content", ""))
    return jsonify({"ok": True})

@app.route("/api/explorer/files/delete", methods=["POST"])
def delete_file():
    rel = safe_rel_path(request.json.get("path"))
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR)) or not path.exists():
        return jsonify({"error": "Not found"}), 404
    if path.is_dir(): shutil.rmtree(path)
    else: path.unlink()
    return jsonify({"ok": True})

@app.route("/api/explorer/files/rename", methods=["POST"])
def rename_file():
    data = request.json or {}
    old = (FILES_DIR / safe_rel_path(data.get("old"))).resolve()
    new = (FILES_DIR / safe_rel_path(data.get("new"))).resolve()
    if not str(old).startswith(str(FILES_DIR)) or not str(new).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    return jsonify({"ok": True})

@app.route("/api/explorer/files/mkdir", methods=["POST"])
def mkdir():
    rel = request.json.get("path")
    name = request.json.get("name")
    path = (FILES_DIR / rel / name if rel else FILES_DIR / name).resolve()
    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    path.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})

@app.route("/api/explorer/files/upload", methods=["POST"])
def upload_file():
    folder_path = request.form.get("path", "")
    upload = request.files.get("file")
    parent = (FILES_DIR / folder_path).resolve()
    if not str(parent).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    upload.save(parent / upload.filename)
    return jsonify({"ok": True})

@app.route("/api/explorer/files/download")
def download():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()
    if not path.exists() or not str(path).startswith(str(FILES_DIR)):
        return "File not found", 404
    return send_file(path, as_attachment=True)

@app.route("/api/explorer/files/tree")
def list_files_tree():
    def get_tree_items(path: Path):
        items = []
        try:
            for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                item = {
                    "name": p.name, "is_dir": p.is_dir(),
                    "path": str(p.relative_to(FILES_DIR))
                }
                if p.is_dir(): item["children"] = get_tree_items(p)
                items.append(item)
        except: pass
        return items
    return jsonify({"tree": get_tree_items(FILES_DIR)})

# ---------------------------------------------------------
# RAG CHAT API (NON-STREAMING)
# ---------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    q = (data.get("query") or "").strip()
    client_history = data.get("history", [])

    if not q:
        return jsonify({"error": "Empty query"}), 400

    rag_history = convert_client_history_to_rag_history(client_history)

    if rag:
        try:
            # Always non-streaming
            ans = rag.ask(q, chat_history=rag_history, stream=False)
        except Exception as e:
            ans = f"RAG error: {e}"
    else:
        ans = "(RAG not initialized)"

    return jsonify({"response": ans})

# ---------------------------------------------------------
# DATA LAB / SQL AGENT
# ---------------------------------------------------------

@app.route("/data-lab")
def data_lab():
    return render_template("data_chat.html")

@app.route("/api/sql_query", methods=["POST"])
def api_sql_query():
    data = request.json or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"output": "Please enter a query."}), 400
    response = sql_system.query(query)
    return jsonify({"output": response})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)