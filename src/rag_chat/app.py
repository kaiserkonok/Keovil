import os
import sys
import shutil
import json  # Added for settings
from pathlib import Path
from typing import List, Dict
from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, stream_with_context
)

# ---------------------------------------------------------
# Fix Python path so rag_engine and agents can be imported
# ---------------------------------------------------------
ROOT = Path(__file__).parent.parent  # /src
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))  # project root

# Define the central storage directory (matching RAG and SQL agents)
HOME_STORAGE = Path.home() / ".k_rag_storage"
DATA_DIR = HOME_STORAGE / "data"
DB_DIR = HOME_STORAGE / "database"
SETTINGS_FILE = HOME_STORAGE / "settings.json"  # Added settings path

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# --- Settings Management Functions ---
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

# Use the standardized data directory for the file explorer
FILES_DIR = DATA_DIR

# ---------------------------------------------------------
# Init RAG engine
# ---------------------------------------------------------
rag = None
if CollegeRAG:
    try:
        # Initialized with saved settings
        rag = CollegeRAG(
            str(DATA_DIR),
            llm_model=current_cfg["llm_model"],
            temperature=current_cfg["temperature"],
        )
        print(f"✅ RAG initialized at {DATA_DIR}")
    except Exception as e:
        print("❌ RAG init failed:", e)

# ---------------------------------------------------------
# Init SQL Agent
# ---------------------------------------------------------
sql_system = None
if StructuredDataAgent:
    try:
        DB_FILE = str(DB_DIR / "main.db")
        sql_system = StructuredDataAgent()
        # Ensure SQL agent uses the saved model
        sql_system.start_monitoring()
        print(f"✅ SQL Agent initialized with DB at {DB_FILE}")
    except Exception as e:
        print("❌ SQL Agent init failed:", e)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
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


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/cms")
def cms():
    return render_template("explorer.html")


@app.route("/settings")  # Added settings route
def settings():
    return render_template("settings.html")


# ---------------------------------------------------------
# SETTINGS API (New)
# ---------------------------------------------------------
@app.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    global rag
    if request.method == "GET":
        return jsonify(load_settings())

    data = request.json
    save_settings(data)

    # Update engines live without restart
    if rag:
        from langchain_ollama import OllamaLLM
        rag.llm = OllamaLLM(model=data["llm_model"], temperature=float(data["temperature"]))

    return jsonify({"ok": True})


# ---------------------------------------------------------
# FILE EXPLORER API
# ---------------------------------------------------------

@app.route("/api/explorer/files")
def list_files():
    rel_path = safe_rel_path(request.args.get("path"))
    target_dir = (FILES_DIR / rel_path).resolve()

    # Security check: Ensure we stay inside FILES_DIR
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
        except Exception:
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
    except Exception:
        text = "(Unable to read file)"
    return jsonify({"content": text, "name": path.name})


@app.route("/api/explorer/files/save", methods=["POST"])
def save_file():
    data = request.json or {}
    rel = data.get("path")
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    open(path, "w", encoding="utf-8").write(data.get("content", ""))
    return jsonify({"ok": True})


@app.route("/api/explorer/files/delete", methods=["POST"])
def delete_file():
    rel = safe_rel_path(request.json.get("path"))
    path = (FILES_DIR / rel).resolve()
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
    rel = request.json.get("path", "")
    name = request.json.get("name")
    path = (FILES_DIR / rel / name).resolve()
    if not str(path).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    path.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/upload", methods=["POST"])
def upload_file():
    # Modified to support multiple files and folder structures
    target_folder_path = request.form.get("path", "")
    files = request.files.getlist("file")
    # This list holds the relative paths for folder uploads
    full_paths = request.form.getlist("full_paths")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    parent_dir = (FILES_DIR / target_folder_path).resolve()

    # Check if target is inside FILES_DIR
    if not str(parent_dir).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid target directory"}), 400

    for i, file in enumerate(files):
        if file.filename == '':
            continue

        # Use relative path if provided (for folder upload), else use filename
        rel_path = full_paths[i] if (full_paths and i < len(full_paths)) else file.filename
        dest_path = (parent_dir / rel_path).resolve()

        # Security check for each file
        if not str(dest_path).startswith(str(FILES_DIR.resolve())):
            continue

        # Create necessary subdirectories
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(dest_path)

    return jsonify({"ok": True})


@app.route("/api/explorer/files/download")
def download():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()
    if not path.exists() or not str(path).startswith(str(FILES_DIR.resolve())):
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
        except:
            pass
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

    if sql_system:
        response = sql_system.query(query)
    else:
        response = "SQL system not initialized."

    return jsonify({"output": response})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)