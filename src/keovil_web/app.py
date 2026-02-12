import os
import sys
from colorama import Fore, Style

# 1. 🛡️ FIX: Absolute Root Discovery
# This finds the 'Keovil' (root) directory regardless of where you call it from
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 2. Import from the COMPILED server.so
try:
    # Now that BASE_DIR is at the top of sys.path, 'src' is a valid top-level package
    from src.keovil_web.server import socketio, app
except ImportError as e:
    # Diagnostic print to help you if it still fails
    print(f"{Fore.YELLOW}DEBUG: sys.path is {sys.path[:2]}")
    print(f"{Fore.RED}CRITICAL: Failed to load secure kernel: {e}{Style.RESET_ALL}")
    sys.exit(1)

if __name__ == "__main__":
    print(f"\n{Style.BRIGHT}--- Keovil System Startup ---")

    # The rest of your logic is perfect...
    print(f"{Fore.GREEN}🚀 MODE: PRODUCTION (High-Performance Binary)")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
