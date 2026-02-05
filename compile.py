from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import os
import shutil

# --- SECURITY OVERRIDE ---
Cython.Compiler.Options.emit_code_comments = False
Cython.Compiler.Options.docstrings = False

extensions = [
    # Core Logic
    Extension("src.knowledge_engine", ["src/knowledge_engine.py"]),
    Extension("src.neural_db", ["src/neural_db.py"]),
    Extension("src.knowledge_splitter", ["src/knowledge_splitter.py"]),

    # Agents & Utils
    Extension("src.agents.db_agent", ["src/agents/db_agent.py"]),
    Extension("src.utils.document_processor", ["src/utils/document_processor.py"]),
    Extension("src.utils.file_collector", ["src/utils/file_collector.py"]),
    Extension("src.utils.model_engine", ["src/utils/model_engine.py"]),

    # 🛡️ THE PROTECTOR: Compiling the hardware lock & Flask logic
    Extension("src.keovil_web.server", ["src/keovil_web/server.py"]),
]

try:
    setup(
        name="Keovil-Core",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                'language_level': "3",
                'emit_code_comments': False,
                'boundscheck': False,
                'wraparound': False,
                'always_allow_keywords': True  # Required for Flask compatibility
            },
            nthreads=4
        ),
        script_args=['build_ext', '--inplace']
    )
finally:
    print("🧹 Cleaning up C source files...")
    if os.path.exists("build"):
        shutil.rmtree("build")
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".c") or file.endswith(".cpp"):
                os.remove(os.path.join(root, file))
    print("✅ Build complete. Hardware lock logic is now binary.")