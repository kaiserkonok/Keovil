from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import os
import shutil

# --- SECURITY OVERRIDE ---
Cython.Compiler.Options.emit_code_comments = False
Cython.Compiler.Options.docstrings = False

# We now map the NEW filenames to NEW binary names for maximum stealth.
extensions = [
    # Core Logic
    Extension("src.knowledge_engine", ["src/knowledge_engine.py"]),
    Extension("src.neural_db", ["src/neural_db.py"]),
    Extension("src.knowledge_splitter", ["src/knowledge_splitter.py"]),

    # Agents
    Extension("src.agents.db_agent", ["src/agents/db_agent.py"]),

    # Utils
    Extension("src.utils.document_processor", ["src/utils/document_processor.py"]),
    Extension("src.utils.file_collector", ["src/utils/file_collector.py"]),
    Extension("src.utils.model_engine", ["src/utils/model_engine.py"]),
]

try:
    setup(
        name="K-RAG-Core",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                'language_level': "3",
                'emit_code_comments': False,
                'boundscheck': False,
                'wraparound': False
            },
            nthreads=4
        ),
        script_args=['build_ext', '--inplace']
    )
finally:
    # --- AUTO-CLEANUP ---
    print("🧹 Cleaning up C source files and build artifacts...")
    if os.path.exists("build"):
        shutil.rmtree("build")

    for root, dirs, files in os.walk("src"):
        for file in files:
            # Remove intermediate C files
            if file.endswith(".c") or file.endswith(".cpp"):
                os.remove(os.path.join(root, file))
    print("✅ Build complete. Stealth binaries generated.")