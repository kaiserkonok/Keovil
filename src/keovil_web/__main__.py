import os
import sys
from pathlib import Path

package_root = Path(__file__).parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from keovil_web.app import main

if __name__ == "__main__":
    main()
