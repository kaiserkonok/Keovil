from flask import Flask, render_template, request, jsonify, send_file, session
import sys
from pathlib import Path
import os
import shutil
import uuid

# ----------------------
# Fix Python path to access rag_engine
# ----------------------
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import CollegeRAG (your RAG engine)
try:
    from rag_engine import CollegeRAG
except Exception:
    CollegeRAG = None

# ----------------------
# Initialize Flask
# ----------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "super-secret-key"


# ----------------------
# Project directories
# ----------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "test_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILES_DIR = DATA_DIR

print("Using FILES_DIR:", str(FILES_DIR))


# ----------------------
# Initialize RAG
# ----------------------
rag = None
if CollegeRAG:
    try:
        rag = CollegeRAG(str(DATA_DIR))
        print("CollegeRAG initialized successfully.")
    except Exception as e:
        print("RAG initialization failed:", e)
        rag = None


# ----------------------
# Helper functions
# ----------------------
def safe_path_join(base: Path, rel_path: str) -> Path:
    """
    Prevent path escape
    """
    candidate = (base / rel_path).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise ValueError("Unsafe path")
    return candidate


def list_dir_tree(base: Path):
    items = []
    for entry in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        stat = entry.stat()
        items.append({
            "name": entry.name,
            "is_dir": entry.is_dir(),
            "size": stat.st_size,
            "modified": stat.st_mtime
        })
    return items


# ----------------------
# BASIC ROUTES
# ----------------------
@app.route("/")
def home():
    return "<h1>RAG Chat / Explorer</h1><p>Go to <a href='/chat'>/chat</a> or <a href='/cms'>/cms</a></p>"


@app.route("/chat")
def chat():
    """
    Chat UI Page
    """
    return render_template("chat.html")


@app.route("/cms")
def cms():
    """
    File Explorer UI Page
    """
    return render_template("explorer.html")


# ----------------------
# CHAT API  (THIS WAS MISSING)
# ----------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Handles chat messages and returns RAG response.
    """
    data = request.json or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Ensure session has a history
    if "history" not in session:
        session["history"] = []

    session["history"].append({"role": "user", "content": query})

    # If RAG engine is missing, return dummy text
    if rag is None:
        answer = "(RAG engine not initialized)"
    else:
        try:
            answer = rag.ask(query)
        except Exception as e:
            answer = f"RAG error: {e}"

    session["history"].append({"role": "assistant", "content": answer})

    session.modified = True

    return jsonify({"response": answer, "history": session["history"]})


# ----------------------
# FILE EXPLORER API (YOU ALREADY HAD THIS — KEEPING SAME)
# ----------------------
@app.route("/api/explorer/list", methods=["GET"])
def api_list():
    rel = request.args.get("path", "").strip()
    try:
        target = safe_path_join(FILES_DIR, rel) if rel else FILES_DIR
        if not target.exists():
            return jsonify({"error": "Path not found", "files": []}), 404
        items = list_dir_tree(target)
        return jsonify({"files": items, "path": str(target.relative_to(FILES_DIR))})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/api/explorer/view", methods=["GET"])
def api_view():
    rel = request.args.get("path", "").strip()
    max_bytes = int(request.args.get("max_bytes", "200000"))
    if not rel:
        return jsonify({"content": "", "error": "No path provided"}), 400
    try:
        path = safe_path_join(FILES_DIR, rel)
        if not path.exists() or path.is_dir():
            return jsonify({"content": "", "error": "Not found or directory"}), 404
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_bytes)
        return jsonify({"content": text, "name": path.name, "size": path.stat().st_size})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/api/explorer/save", methods=["POST"])
def api_save():
    data = request.json or {}
    rel = data.get("path", "").strip()
    content = data.get("content", "")
    if not rel:
        return jsonify({"error": "No path"}), 400
    try:
        path = safe_path_join(FILES_DIR, rel)
        if path.is_dir():
            return jsonify({"error": "Is directory"}), 400
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return jsonify({"ok": True, "message": "Saved"})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/api/explorer/delete", methods=["POST"])
def api_delete():
    data = request.json or {}
    rel = data.get("path", "").strip()
    if not rel:
        return jsonify({"error": "No path"}), 400
    try:
        path = safe_path_join(FILES_DIR, rel)
        if path.is_dir():
            try:
                path.rmdir()
                return jsonify({"ok": True, "msg": "Dir removed"})
            except OSError:
                return jsonify({"error": "Dir not empty"}), 400
        else:
            path.unlink()
            return jsonify({"ok": True, "msg": "File removed"})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/api/explorer/rename", methods=["POST"])
def api_rename():
    data = request.json or {}
    rel = data.get("path", "")
    new_name = data.get("new_name", "")
    if not rel or not new_name:
        return jsonify({"error": "Missing params"}), 400
    try:
        path = safe_path_join(FILES_DIR, rel)
        new_path = path.parent / new_name
        if new_path.exists():
            return jsonify({"error": "Name exists"}), 400
        path.rename(new_path)
        return jsonify({"ok": True})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/api/explorer/mkdir", methods=["POST"])
def api_mkdir():
    data = request.json or {}
    rel = data.get("path", "")
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Missing name"}), 400
    try:
        parent = safe_path_join(FILES_DIR, rel) if rel else FILES_DIR
        new_dir = parent / name
        if new_dir.exists():
            return jsonify({"error": "Exists"}), 400
        new_dir.mkdir()
        return jsonify({"ok": True})
    except ValueError:
        return jsonify({"error": "Invalid path"})


@app.route("/api/explorer/upload", methods=["POST"])
def api_upload():
    rel = request.form.get("path", "")
    upload = request.files.get("file")
    if not upload:
        return jsonify({"error": "No file"}), 400
    try:
        parent = safe_path_join(FILES_DIR, rel) if rel else FILES_DIR
        dest = parent / upload.filename
        if dest.exists():
            stem, suf = dest.stem, dest.suffix
            dest = parent / f"{stem}_{uuid.uuid4().hex[:6]}{suf}"
        upload.save(str(dest))
        return jsonify({"ok": True, "filename": dest.name})
    except ValueError:
        return jsonify({"error": "Invalid path"})


@app.route("/api/explorer/download")
def api_download():
    rel = request.args.get("path", "")
    if not rel:
        return jsonify({"error": "No path"}), 400
    try:
        path = safe_path_join(FILES_DIR, rel)
        if path.is_dir():
            return jsonify({"error": "Is directory"}), 400
        return send_file(str(path), as_attachment=True)
    except ValueError:
        return jsonify({"error": "Invalid path"})


@app.route("/api/explorer/search")
def api_search():
    q = request.args.get("q", "").lower().strip()
    rel = request.args.get("path", "").strip()
    try:
        base = safe_path_join(FILES_DIR, rel) if rel else FILES_DIR
        results = []
        for entry in base.rglob("*"):
            if entry.is_file() and q in entry.name.lower():
                results.append(str(entry.relative_to(FILES_DIR)))
        return jsonify({"results": results})
    except ValueError:
        return jsonify({"error": "Invalid path"})


# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
