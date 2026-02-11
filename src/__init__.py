import os, sys, hashlib


def _shield():
    S = "rtx-keovil-kevil-mommy-SHIELD-2026"
    H = "b6e792214fdc0151bc12ffe2fffb4c61d9de247b2975fe632b9608333c4d7afd"
    key_file = os.path.join(os.path.expanduser("~"), ".keovil", ".god_mode")

    print(key_file)

    is_admin = False
    try:
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                if hashlib.sha256((f.read().strip() + S).encode()).hexdigest() == H:
                    is_admin = True
    except:
        pass

    if not is_admin:
        # 1. This prints to the ACTUAL terminal
        print("Keovil Starting...")

        # 2. MANDATORY: Push the text out of the buffer now!
        sys.stdout.flush()

        # 3. NOW we swap the pipes. Everything after this goes to the black hole.
        null_fds = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fds, 1)  # Lock Stdout
        os.dup2(null_fds, 2)  # Lock Stderr

        # Backup for high-level Python calls
        sys.stdout = sys.stderr = open(os.devnull, 'w')
    else:
        # You'll see this because we haven't redirected the pipes for admins
        print("🔓 KEOVIL ADMIN: SECURE SESSION INITIALIZED")


_shield()