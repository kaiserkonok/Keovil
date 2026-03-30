import os
import sys
from pathlib import Path
from colorama import Fore, Style

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from keovil_web.server import socketio, app
except ImportError:
    try:
        from src.keovil_web.server import socketio, app
    except ImportError as e:
        print(
            f"{Fore.RED}CRITICAL: Failed to load keovil_web.server: {e}{Style.RESET_ALL}"
        )
        sys.exit(1)


def main():
    print(f"\n{Style.BRIGHT}--- Keovil System Startup ---")
    print(f"{Fore.GREEN}🚀 MODE: PRODUCTION (High-Performance Binary)")
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
