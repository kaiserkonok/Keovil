# src/rag_chat/app.py
from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, stream_with_context
)
import sys
from pathlib import Path
import os
import shutil
import time
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
def chunks(text, size=120):
    """Split text into chunks for streaming"""
    for i in range(0, len(text), size):
        yield text[i:i + size]


def convert_client_history_to_rag_history(client_history: List[Dict[str, str]]):
    """
    Convert client-side history [{role: 'user'|'assistant', content: str}, ...]
    to RAG engine format: [("You", "..."), ("AI", "..."), ...]
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
    # Stateless: no server-side session; simply serve the UI
    return render_template("chat.html")


@app.route("/cms")
def cms():
    return render_template("explorer.html")


# ---------------------------------------------------------
# FILE EXPLORER API (unchanged behavior)
# ---------------------------------------------------------
@app.route("/api/explorer/files")
def list_files():
    items = []
    for p in sorted(FILES_DIR.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
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
    rel = request.args.get("path", "")
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    if path.is_dir():
        return jsonify({"content": "(Directory)"}), 200
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except:
        text = "(Unable to read file)"
    return jsonify({"content": text, "name": path.name, "size": path.stat().st_size})


@app.route("/api/explorer/files/save", methods=["POST"])
def save_file():
    data = request.json or {}
    rel = data.get("path")
    content = data.get("content", "")
    if not rel:
        return jsonify({"error": "Missing path"}), 400
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    open(path, "w", encoding="utf-8").write(content)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/delete", methods=["POST"])
def delete_file():
    rel = request.json.get("path", "")
    path = (FILES_DIR / rel).resolve()
    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return jsonify({"ok": True})


@app.route("/api/explorer/files/rename", methods=["POST"])
def rename_file():
    data = request.json
    old = (FILES_DIR / data["old"]).resolve()
    new = (FILES_DIR / data["new"]).resolve()
    if not str(old).startswith(str(FILES_DIR)) or not str(new).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    old.rename(new)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/mkdir", methods=["POST"])
def mkdir():
    rel = request.json.get("path")
    name = request.json.get("name")
    if not name:
        return jsonify({"error": "Missing folder name"}), 400
    path = (FILES_DIR / rel / name if rel else FILES_DIR / name).resolve()
    if not str(path).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    path.mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})


@app.route("/api/explorer/files/upload", methods=["POST"])
def upload_file():
    folder = request.form.get("path", "")
    upload = request.files.get("file")
    if not upload:
        return jsonify({"error": "No file uploaded"}), 400
    parent = (FILES_DIR / folder).resolve()
    if not str(parent).startswith(str(FILES_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    dest = parent / upload.filename
    upload.save(dest)
    return jsonify({"ok": True, "filename": upload.filename})


@app.route("/api/explorer/files/download")
def download():
    rel = request.args.get("path")
    path = (FILES_DIR / rel).resolve()
    if not path.exists() or not str(path).startswith(str(FILES_DIR)):
        return "File not found", 404
    return send_file(path, as_attachment=True)


# ---------------------------------------------------------
# RAG CHAT API - STREAMING (stateless)
# ---------------------------------------------------------
@app.route("/api/stream_chat", methods=["POST"])
def api_stream_chat():
    """Streaming chat endpoint; stateless and client-driven history."""
    data = request.json or {}
    q = (data.get("query") or "").strip()
    client_history = data.get("history", [])

    if not q:
        return jsonify({"error": "Empty query"}), 400

    rag_history = convert_client_history_to_rag_history(client_history)

    def generate():
        # typing event
        yield "event: typing\ndata: {}\n\n"

        if rag:
            try:
                ans = rag.ask(q, chat_history=rag_history, stream=False)
            except Exception as e:
                ans = f"RAG error: {e}"
        else:
            ans = "(RAG not initialized)"

        # Stream response in chunks to client
        for c in chunks(ans):
            yield f"data: {c}\n\n"
            time.sleep(0.02)

        # done event
        yield "event: done\ndata: {}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ---------------------------------------------------------
# RAG CHAT API - Non-streaming (stateless)
# ---------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Non-streaming chat endpoint; stateless and client-driven history."""
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

    # Return response; client will update its own history
    return jsonify({"response": ans})


# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure this is set to False in a production environment
    app.run(debug=True, threaded=True)
