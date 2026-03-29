import os
import sys

import shutil
import json
import sqlite3
from pathlib import Path
from typing import List, Dict
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    Response,
    stream_with_context,
)
import traceback
from threading import Thread
from flask_socketio import SocketIO, emit, join_room  # <--- ADD THIS
from colorama import Fore, Style, init
import hashlib
import platform
import subprocess
import requests

init(autoreset=True)

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------
APP_MODE = "production"

HOME_STORAGE = Path.home() / ".keovil_storage"
STORAGE_STR = os.getenv("STORAGE_BASE", str(HOME_STORAGE))
HOME_STORAGE = Path(STORAGE_STR).absolute()


# Ensure storage is writable, fallback to local if not
def ensure_storage_writable(path):
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
        return path
    except:
        # Fallback to local storage in project directory
        fallback = Path(__file__).parent.parent.parent / "keovil_data"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"{Fore.YELLOW}⚠️ Cannot write to {path}, using fallback: {fallback}{Style.RESET_ALL}"
        )
        return fallback


HOME_STORAGE = ensure_storage_writable(HOME_STORAGE)

DATA_DIR = HOME_STORAGE / "data"
DB_DIR = HOME_STORAGE / "database"

# Check for old database file or create new one
old_chat_db = DB_DIR / "chat_history_production.db"
new_chat_db = DB_DIR / "chat_history.db"
CHAT_DB = old_chat_db if old_chat_db.exists() else new_chat_db
print(f"Chat Database: {CHAT_DB}")

# Global for explorer
FILES_DIR = DATA_DIR

print(f"{Fore.CYAN}🚀 SILO ACTIVE: {APP_MODE.upper()}")
print(f"📍 STORAGE PATH: {HOME_STORAGE}{Style.RESET_ALL}")

# ---------------------------------------------------------
# BOUNCER CONFIGURATION (Registry Handshake)
# ---------------------------------------------------------
FOUNDRY_LOCAL = "http://localhost:8000"
FOUNDRY_PROD = "https://kevil.io"

# Toggle this based on where your Django is running
REGISTRY_URL = FOUNDRY_PROD
AUTH_FILE = HOME_STORAGE / ".kevil_auth"

print(f"Auth file path: {AUTH_FILE}")


def get_chubby_hwid():
    import subprocess
    import hashlib
    import platform
    import os

    system = platform.system()
    components = []

    # --- STEP 1: THE PRIMARY ANCHOR (Motherboard / System UUID) ---
    try:
        if system == "Windows":
            # Stable BIOS UUID
            cmd = "wmic csproduct get uuid"
            m_uuid = (
                subprocess.check_output(cmd, shell=True).decode().split("\n")[1].strip()
            )
            components.append(m_uuid)
        elif system == "Linux":
            # In Docker, /sys/class/dmi is the gold standard if mapped, otherwise machine-id
            if os.path.exists("/sys/class/dmi/id/product_uuid"):
                with open("/sys/class/dmi/id/product_uuid", "r") as f:
                    components.append(f.read().strip())
            elif os.path.exists("/etc/machine-id"):
                with open("/etc/machine-id", "r") as f:
                    components.append(f.read().strip())
            else:
                components.append(platform.node())  # Last resort hostname
        elif system == "Darwin":
            cmd = "ioreg -d2 -c IOPlatformExpertDevice | awk -F\\\" '/IOPlatformUUID/{print $(NF-1)}'"
            components.append(subprocess.check_output(cmd, shell=True).decode().strip())
    except:
        components.append("BASE-NODE")

    # --- STEP 2: THE SECONDARY ANCHOR (The 5060 Ti / Silicon Fingerprint) ---
    try:
        # GPU UUID is the best secondary anchor for an AI app
        gpu_uuid = (
            subprocess.check_output(
                "nvidia-smi --query-gpu=uuid --format=csv,noheader",
                shell=True,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        components.append(gpu_uuid)
    except:
        # Fallback to CPU ID if GPU is not visible (e.g., during container setup)
        try:
            if system == "Windows":
                cpu = (
                    subprocess.check_output("wmic cpu get processorid", shell=True)
                    .decode()
                    .split("\n")[1]
                    .strip()
                )
                components.append(cpu)
            else:
                # Get CPU Model Name from /proc/cpuinfo
                cpu = (
                    subprocess.check_output(
                        "grep -m 1 'model name' /proc/cpuinfo", shell=True
                    )
                    .decode()
                    .strip()
                )
                components.append(cpu)
        except:
            components.append(platform.processor())

    # --- STEP 3: HASHING ---
    # We join with a unique separator and add your secret salt
    raw_id = "|".join(components) + "KEV-SALT-99-PROD"
    return hashlib.sha256(raw_id.encode()).hexdigest()[:32]


# Final Hardware Identity
HWID = get_chubby_hwid()

print(f"Generated hardware id: {HWID}")

# In-memory cache to prevent constant API pinging (Speed optimization)
is_verified_session = False


def is_node_authorized():
    """Checks if the local auth file exists and is valid on the Registry."""
    global is_verified_session

    # If already verified this session, skip the network call
    if is_verified_session:
        return True

    if not AUTH_FILE.exists():
        return False

    try:
        with open(AUTH_FILE, "r") as f:
            saved_key = f.read().strip()

        # Verify with Foundry (Django)
        r = requests.post(
            f"{REGISTRY_URL}/api/verify/",
            json={"master_key": saved_key, "hwid": HWID, "product_slug": "keovil"},
            timeout=5,
        )

        if r.json().get("status") == "authorized":
            is_verified_session = True  # Success! Lock it in for this session
            return True
        return False
    except:
        return False


# ---------------------------------------------------------
# Chat History Database Initialization
# ---------------------------------------------------------
def init_chat_db():
    conn = sqlite3.connect(str(CHAT_DB))
    curr = conn.cursor()

    # 1. Create the table with the new column (for brand new setups)
    curr.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id TEXT,
            title TEXT, 
            session_type TEXT DEFAULT 'rag', 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. MIGRATION: Check if session_type column exists in an existing DB
    curr.execute("PRAGMA table_info(sessions)")
    columns = [column[1] for column in curr.fetchall()]

    if "session_type" not in columns:
        print(f"{Fore.YELLOW}🛠️ Migrating database: Adding 'session_type' column...")
        curr.execute("ALTER TABLE sessions ADD COLUMN session_type TEXT DEFAULT 'rag'")
        # Ensure any NULLs from the alter table are set to 'rag'
        curr.execute(
            "UPDATE sessions SET session_type = 'rag' WHERE session_type IS NULL"
        )
        conn.commit()
        print(f"{Fore.GREEN}✅ Migration complete.")

    # 3. Create messages table as usual
    curr.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            session_id INTEGER, 
            role TEXT, 
            content TEXT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


init_chat_db()

# ---------------------------------------------------------
DEFAULT_MODEL = "qwen2.5-coder:7b-instruct"

# ---------------------------------------------------------
# Engine Imports - Clean & Direct
# ---------------------------------------------------------
# server.py is in src/keovil_web/server.py. We need to add 'src' to the path.
src_root = Path(__file__).resolve().parent.parent

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from college_rag import CollegeRAG
    from agents.db_agent import StructuredDataAgent

    print(f"{Fore.GREEN}✅ Engines linked from: {src_root}{Style.RESET_ALL}")
except ImportError as e:
    print(f"{Fore.RED}❌ Link Error: {e}{Style.RESET_ALL}")
    CollegeRAG = None
    StructuredDataAgent = None

# ---------------------------------------------------------
# Flask Initialization
# ---------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.jinja_env.auto_reload = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 * 1024  # Allow up to 100GB

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

import threading

# Globals for engines
rag = None
sql_system = None
# FIX: Change Lock() to RLock() to allow nested calls (Recursion)
ENGINE_INIT_LOCK = threading.RLock()


def initialize_engines():
    """
    Initializes heavy GPU engines with full interactive feedback.
    Optimized for RTX 5060 Ti performance.
    """

    # Use OLLAMA_HOST env var, default to localhost:11434 for network mode
    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    OLLAMA_BASE_URL = ollama_host

    global rag, sql_system

    # 1. Double-check lock to prevent race conditions during boot
    with ENGINE_INIT_LOCK:
        if rag is not None and sql_system is not None:
            return

        # --- 1. Ollama Hardware Handshake ---
        try:
            import requests

            model_name = DEFAULT_MODEL
            print(f"{Fore.CYAN}Verifying {model_name}...{Style.RESET_ALL}")

            # Check Ollama connection
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/show", json={"name": model_name}, timeout=5
            )
            if r.status_code != 200:
                print(
                    f"{Fore.RED}Model '{model_name}' not found on Ollama.{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}Please run: ollama pull {model_name}{Style.RESET_ALL}"
                )
                import sys

                sys.exit(1)
            else:
                print(f"{Fore.GREEN}Model verified.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Ollama Link Error: {e}{Style.RESET_ALL}")
            import sys

            sys.exit(1)

        # --- 2. RAG & VRAM Allocation ---
        if CollegeRAG and rag is None:
            try:
                print(f"{Fore.CYAN}Loading RAG engine...{Style.RESET_ALL}")

                # We initialize into a local variable first to ensure
                # we don't set the global 'rag' to a half-broken object
                temp_rag = CollegeRAG(data_dir=str(DATA_DIR), socketio=socketio)
                rag = temp_rag
                print(f"{Fore.GREEN}RAG engine ready.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}💥 RAG INIT FATAL ERROR:{Style.RESET_ALL}")
                traceback.print_exc()
                rag = None  # Ensure it stays None so api_chat knows it's broken

        # --- 3. SQL Agent Boot ---
        if StructuredDataAgent and sql_system is None:
            try:
                print(f"{Fore.CYAN}Loading SQL agent...{Style.RESET_ALL}")
                temp_sql = StructuredDataAgent(socketio=socketio)
                if hasattr(temp_sql, "agent") and hasattr(temp_sql.agent, "llm"):
                    temp_sql.agent.llm.temperature = 0.0
                temp_sql.start_monitoring()
                sql_system = temp_sql
                print(f"{Fore.GREEN}SQL agent ready.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}💥 SQL INIT FATAL ERROR:{Style.RESET_ALL}")
                traceback.print_exc()
                sql_system = None

        # --- 4. Final ---
        if rag and sql_system:
            print(f"{Fore.GREEN}All engines ready.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Partial initialization.{Style.RESET_ALL}")


# Trigger initialization
# We keep this as a daemon thread so the Flask server starts instantly
Thread(target=initialize_engines, daemon=True).start()


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def safe_rel_path(p: str | None) -> str:
    return (p or "").strip("/\\")


# ---------------------------------------------------------
# UI ROUTES
# ---------------------------------------------------------


@app.before_request
def gatekeeper():
    """The High-Level Bouncer."""
    # 1. Exempt routes (Don't lock the door if we're trying to put the key in)
    exempt_paths = ["/static", "/activate", "/api/bootstrap", "/", "/index.html"]
    if any(request.path.startswith(path) for path in exempt_paths):
        return None

    # 2. Check Identity
    if not is_node_authorized():
        # Passing HWID to template so user can see their 'Node Signature'
        return render_template("activate.html", hwid=HWID)


@app.route("/activate")
def activate_ui():
    """Renders the chunky activation screen."""
    return render_template("activate.html", hwid=HWID)


@app.route("/api/bootstrap", methods=["POST"])
def bootstrap():
    """Endpoint for the First-Run Handshake."""
    key = request.json.get("master_key")
    if not key:
        return jsonify({"status": "error", "msg": "Master Key Required"}), 400

    try:
        print(f"{Fore.MAGENTA}Registry URL: {REGISTRY_URL}")
        # Attempt to bond the Node to the Registry
        r = requests.post(
            f"{REGISTRY_URL}/api/verify/",
            json={"master_key": key, "hwid": HWID, "product_slug": "keovil"},
        )

        res_data = r.json()

        if res_data.get("status") == "authorized":
            # Save the key locally in your isolated storage
            with open(AUTH_FILE, "w") as f:
                f.write(key)
            return jsonify({"status": "success"})
        else:
            # Return specific error (e.g., 'Hardware lock active' or 'Invalid Key')
            return jsonify(
                {
                    "status": "error",
                    "msg": res_data.get("msg", "Registry denied handshake."),
                }
            ), 401

    except requests.exceptions.ConnectionError as e:
        print(e)
        return jsonify(
            {"status": "error", "msg": "FOUNDRY_OFFLINE: Could not reach Kevil.io"}
        ), 500
    except Exception as e:
        return jsonify({"status": "error", "msg": f"SYSTEM_ERROR: {str(e)}"}), 500


@app.route("/api/logout", methods=["POST"])
def logout():
    """Wipes the local auth file and resets the session."""
    global is_verified_session
    try:
        # 1. Kill the local file
        if AUTH_FILE.exists():
            AUTH_FILE.unlink(missing_ok=True)

        # 2. Reset the in-memory cache
        is_verified_session = False

        print(
            f"{Fore.RED}🔴 NODE DEACTIVATED: Local auth file purged.{Style.RESET_ALL}"
        )
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@socketio.on("connect")
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
    # Get the type from query params (e.g., /api/chat/sessions?type=sql)
    stype = request.args.get("type", "rag")

    if not uid:
        return jsonify([])

    conn = sqlite3.connect(CHAT_DB)
    # Added WHERE session_type = ?
    cur = conn.execute(
        "SELECT id, title FROM sessions WHERE user_id = ? AND session_type = ? ORDER BY created_at DESC",
        (uid, stype),
    )
    sessions = [{"id": r[0], "title": r[1]} for r in cur.fetchall()]
    conn.close()
    return jsonify(sessions)


@app.route("/api/chat/history/<int:session_id>")
def get_chat_history_db(session_id):
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,),
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

    # 1. Check if we need to initialize (With RLock, this is now safe)
    if rag is None:
        print(
            f"{Fore.YELLOW}⏳ GPU engine missing. Attempting to boot...{Style.RESET_ALL}"
        )
        with ENGINE_INIT_LOCK:
            if rag is None:
                try:
                    # Capture the return or state change
                    initialize_engines()

                    if rag is None:
                        print(
                            f"{Fore.RED}⚠️ initialize_engines() finished but 'rag' global variable is still None! Check if you declared 'global rag' inside initialize_engines().{Style.RESET_ALL}"
                        )
                    else:
                        print(
                            f"{Fore.GREEN}✅ Engine recovered successfully.{Style.RESET_ALL}"
                        )

                except Exception as e:
                    print(
                        f"{Fore.RED}💥 CRITICAL: initialize_engines() crashed during request-time boot!{Style.RESET_ALL}"
                    )
                    # THIS IS THE KEY: It will show you the Errno -3 or whatever is killing it
                    traceback.print_exc()

    # 2. Final check: If it's STILL None after initialization attempt
    if rag is None:
        print(f"{Fore.RED}❌ RAG failed to initialize after request.{Style.RESET_ALL}")
        return jsonify(
            {
                "response": "I'm sorry, the AI engine is currently offline. Please check the server console for errors."
            }
        ), 503

    # --- Proceed with normal Chat logic ---
    conn = sqlite3.connect(CHAT_DB)
    rag_history = []
    if session_id:
        cur = conn.execute(
            """SELECT role, content
               FROM (SELECT role, content, timestamp
                     FROM messages
                     WHERE session_id = ?
                     ORDER BY timestamp DESC LIMIT 10)
               ORDER BY timestamp ASC""",
            (session_id,),
        )
        history_rows = cur.fetchall()
        rag_history = [("You" if r[0] == "user" else "AI", r[1]) for r in history_rows]

    socketio.emit("system_status", {"is_busy": True, "rag": {"state": "processing"}})

    try:
        ans = rag.ask(q, chat_history=rag_history)

        if not session_id:
            title = q[:35] + "..." if len(q) > 35 else q
            # UPDATE THE INSERT BELOW:
            cur = conn.execute(
                "INSERT INTO sessions (user_id, title, session_type) VALUES (?, ?, 'rag')",
                (uid, title),
            )
            session_id = cur.lastrowid

        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
            (session_id, q),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
            (session_id, ans),
        )
        conn.commit()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        socketio.emit("system_status", {"is_busy": False, "rag": {"state": "idle"}})

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
@app.route("/api/explorer/files")
def list_files():
    rel_path = safe_rel_path(request.args.get("path"))
    target_dir = (FILES_DIR / rel_path).resolve()
    if not str(target_dir).startswith(str(FILES_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({"error": "Not found"}), 404
    items = []
    sorted_items = sorted(
        target_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
    )
    for p in sorted_items:
        try:
            stat = p.stat()
            items.append(
                {
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "path": str(p.relative_to(FILES_DIR)),
                }
            )
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
    if not str(old).startswith(str(FILES_DIR.resolve())) or not str(new).startswith(
        str(FILES_DIR.resolve())
    ):
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
        if not file.filename:
            continue

        rel_path = (
            full_paths[i] if (full_paths and i < len(full_paths)) else file.filename
        )
        dest = (parent_dir / rel_path).resolve()

        if not str(dest).startswith(str(FILES_DIR.resolve())):
            continue
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
                    "path": str(item.relative_to(FILES_DIR)),
                }
                if item.is_dir():
                    entry["children"] = get_tree_items(item)
                res.append(entry)
        except Exception:
            pass
        return res

    return jsonify({"tree": get_tree_items(FILES_DIR)})


@app.route("/api/sql_query", methods=["POST"])
def api_sql_query():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    session_id = data.get("session_id")
    uid = request.headers.get("X-User-ID")

    if not query or not uid:
        return jsonify({"output": "Missing query or user ID."}), 400

    # 1. Fetch history from the shared database (Filtering for SQL context only)
    sql_history = []
    conn = sqlite3.connect(CHAT_DB)
    if session_id:
        try:
            # We fetch the last 10 messages to keep the context window efficient
            cur = conn.execute(
                """SELECT role, content
                   FROM (SELECT role, content, timestamp
                         FROM messages
                         WHERE session_id = ?
                         ORDER BY timestamp DESC LIMIT 10)
                   ORDER BY timestamp ASC""",
                (session_id,),
            )
            rows = cur.fetchall()
            # Format as a list of dicts for the LangChain-based SQL agent
            sql_history = [{"role": r[0], "content": r[1]} for r in rows]
        except Exception as e:
            print(f"{Fore.RED}⚠️ SQL History Fetch Error: {e}{Style.RESET_ALL}")

    # 2. Execute the Query through the Agent
    try:
        if sql_system:
            # Pass both query and history. The agent uses this to create a 'standalone_query'
            response = sql_system.query(query, chat_history=sql_history)
        else:
            response = "SQL System Offline"
    except Exception as e:
        traceback.print_exc()
        response = f"I ran into an issue: {str(e)}"

    # 3. Handle Session Persistence
    try:
        # Create a new session if one doesn't exist
        if not session_id:
            title = query[:35] + "..." if len(query) > 35 else query
            # CRITICAL: We tag this session as 'sql' to keep it isolated from RAG
            cur = conn.execute(
                "INSERT INTO sessions (user_id, title, session_type) VALUES (?, ?, 'sql')",
                (uid, title),
            )
            session_id = cur.lastrowid

        # Save the current exchange to the database
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
            (session_id, query),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
            (session_id, response),
        )
        conn.commit()
    except Exception as e:
        print(f"{Fore.RED}⚠️ Database Save Error: {e}{Style.RESET_ALL}")
    finally:
        conn.close()

    # Return response and session_id so the frontend can track the thread
    return jsonify({"output": response, "session_id": session_id})


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
        if ext == ".csv":
            df = pd.read_csv(
                path, nrows=100, on_bad_lines="skip", encoding_errors="replace"
            )
            # Add this line to handle NaN
            df = df.fillna("")
            data = [df.columns.tolist()] + df.values.tolist()
            return jsonify({"data": data, "total_rows": "Large File"})

        elif ext in [".xlsx", ".xls"]:
            xl = pd.ExcelFile(path)
            sheet_name = request.args.get("sheet") or xl.sheet_names[0]
            df = pd.read_excel(path, sheet_name=sheet_name, nrows=100)

            # --- CRITICAL FIX HERE ---
            # Replace NaN/Infinity with empty strings so JSON remains valid
            df = df.fillna("")

            data = [df.columns.tolist()] + df.values.tolist()
            return jsonify(
                {"data": data, "sheets": xl.sheet_names, "total_rows": "Large File"}
            )

        return jsonify({"error": "Unsupported preview format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/status")
def get_ingest_status():
    """Returns the current status and triggers a broadcast."""
    rag_info = rag.get_status() if rag else {"state": "idle"}
    sql_is_syncing = sql_system.is_syncing if sql_system else False

    return jsonify(
        {
            "is_busy": (rag_info.get("state") != "idle") or sql_is_syncing,
            "rag": rag_info,
            "sql_syncing": sql_is_syncing,
        }
    )


if __name__ == "__main__":
    is_production = os.getenv("APP_MODE", "development") == "production"

    print(f"\n{Style.BRIGHT}--- System Startup ---")

    if is_production:
        print(f"{Fore.GREEN}🚀 MODE: PRODUCTION (Reloader Disabled)")
        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )
    else:
        print(f"{Fore.YELLOW}🛠️ MODE: DEVELOPMENT (Reloader Enabled)")
        # Note: In Dev, the reloader will cause the "System Startup"
        # message to print twice. This is normal Flask behavior!
        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=True,
            use_reloader=True,
            log_output=True,
            allow_unsafe_werkzeug=True,
        )
