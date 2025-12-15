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

print("FILES_DIR →", FILES_DIR)

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

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def safe_rel_path(p: str | None) -> str:
    """
    Normalize client paths.
    FIX: prevents '', None, /path, \\path causing silent fallback
    """
    return (p or "").strip("/\\")


def convert_client_history_to_rag_history(client_history: List[Dict[str, str]]):
    """
    Convert client-side history to RAG engine format
    """
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
    return "<h1>RAG Chat + CMS</h1><p><a href='/chat'>Chat UI</a> | <a href='/cms'>CMS Explorer</a></p>"


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

    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400

    if not path.exists():
        return jsonify({"error": "File not found"}), 404

    if path.is_dir():
        return jsonify({"content": "(Directory)"}), 200

    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        text = "(Unable to read file)"

    return jsonify({
        "content": text,
        "name": path.name,
        "size": path.stat().st_size,
        "modified": path.stat().st_mtime
    })



@app.route("/api/explorer/files/save", methods=["POST"])
def save_file():
    data = request.json or {}
    rel = data.get("path")
    content = data.get("content", "")

    path = (FILES_DIR / rel).resolve()

    print("=== Save Debug ===")
    print("FILES_DIR:", FILES_DIR)
    print("Relative path:", rel)
    print("Full path:", path)
    print("=================")

    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400

    open(path, "w", encoding="utf-8").write(content)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/delete", methods=["POST"])
def delete_file():
    rel = safe_rel_path(request.json.get("path"))
    path = (FILES_DIR / rel).resolve()

    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400

    if not path.exists():
        return jsonify({"error": "Not found"}), 404

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    return jsonify({"ok": True})


@app.route("/api/explorer/files/rename", methods=["POST"])
def rename_file():
    data = request.json or {}

    old_rel = safe_rel_path(data.get("old"))
    new_rel = safe_rel_path(data.get("new"))

    old = (FILES_DIR / old_rel).resolve()
    new = (FILES_DIR / new_rel).resolve()

    if not str(old).startswith(str(FILES_DIR)) or not str(new).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400

    if not old.exists():
        return jsonify({"error": "Source not found"}), 404

    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)

    return jsonify({"ok": True})


@app.route("/api/explorer/files/mkdir", methods=["POST"])
def mkdir():
    rel = request.json.get("path")  # parent folder
    name = request.json.get("name")  # new folder name

    path = (FILES_DIR / rel / name if rel else FILES_DIR / name).resolve()

    print("=== Mkdir Debug ===")
    print("FILES_DIR:", FILES_DIR)
    print("Parent folder (rel):", rel)
    print("New folder path:", path)
    print("===================")

    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400

    path.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/upload", methods=["POST"])
def upload_file():
    folder_path = request.form.get("path", "")  # folder from frontend
    upload = request.files.get("file")

    parent = (FILES_DIR / folder_path).resolve()
    dest = parent / upload.filename

    print("=== Upload Debug ===")
    print("FILES_DIR:", FILES_DIR)
    print("Folder path from frontend:", folder_path)
    print("Resolved parent folder:", parent)
    print("Destination file path:", dest)
    print("===================")

    if not str(parent).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400

    os.makedirs(parent, exist_ok=True)
    upload.save(dest)

    return jsonify({"ok": True, "filename": upload.filename})


@app.route("/api/explorer/files/download")
def download():
    rel = safe_rel_path(request.args.get("path"))
    path = (FILES_DIR / rel).resolve()

    if not path.exists() or not str(path).startswith(str(FILES_DIR)):
        return "File not found", 404

    return send_file(path, as_attachment=True)


# ---------------------------------------------------------
# FILE TREE (RESTORED)
# ---------------------------------------------------------

@app.route("/api/explorer/files/tree")
def list_files_tree():
    """Recursively list all files and folders for tree view"""

    def get_tree_items(path: Path):
        items = []
        try:
            for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                rel_path = str(p.relative_to(FILES_DIR))
                stat = p.stat()
                item = {
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "size": stat.st_size if not p.is_dir() else 0,
                    "modified": stat.st_mtime,
                    "path": rel_path
                }
                if p.is_dir():
                    item["children"] = get_tree_items(p)
                items.append(item)
        except (PermissionError, OSError):
            pass
        return items

    tree = get_tree_items(FILES_DIR)
    return jsonify({"tree": tree})


# ---------------------------------------------------------
# RAG CHAT API - STREAMING
# ---------------------------------------------------------

@app.route("/api/stream_chat", methods=["POST"])
def api_stream_chat():
    data = request.json or {}
    q = (data.get("query") or "").strip()
    client_history = data.get("history", [])

    if not q:
        return jsonify({"error": "Empty query"}), 400

    rag_history = convert_client_history_to_rag_history(client_history)

    def generate():
        yield "event: typing\ndata: {}\n\n"
        if rag:
            try:
                for chunk in rag.ask(q, chat_history=rag_history, stream=True):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                yield f"data: RAG error: {e}\n\n"
        else:
            yield "data: (RAG not initialized)\n\n"
        yield "event: done\ndata: {}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ---------------------------------------------------------
# RAG CHAT API - NON-STREAMING (RESTORED)
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
# Run App
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
