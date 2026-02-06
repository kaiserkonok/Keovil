import os, sys, hashlib


def _shield():
    S = "rtx-keovil-kevil-mommy-SHIELD-2026"  # Same salt as above
    H = "b6e792214fdc0151bc12ffe2fffb4c61d9de247b2975fe632b9608333c4d7afd"

    # Check for the secret file in your storage
    key_file = os.path.expanduser("~/.keovil/.god_mode")
    print("Starting Keovil Server....")

    try:
        with open(key_file, "r") as f:
            token = f.read().strip()
            # If the token + salt doesn't match the master hash...
            if hashlib.sha256((token + S).encode()).hexdigest() != H:
                raise Exception()
            else:
                print('Yoo Baby')
    except:
        print('Aha Keo..')
        # BLACK HOLE: Point everything to devnull
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = sys.stdout


_shield()