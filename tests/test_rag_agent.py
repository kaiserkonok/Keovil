import sys
import os
import time
import shutil
from pathlib import Path

# 1. POINT TO THE SRC DIRECTORY
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"DEBUG: Root: {project_root}")
print(f"DEBUG: Looking in: {src_path}")

try:
    # Now that we added 'src' to the path, we can import directly
    from college_rag import CollegeRAG, Colors

    print("✅ Import Successful!")
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    sys.exit(1)

# ==========================================
# TEST CONFIGURATION
# ==========================================
TEST_ROOT = Path.home() / ".keovil_test_bench"
os.environ["APP_MODE"] = "test"
os.environ["STORAGE_BASE"] = str(TEST_ROOT)


def setup_test_bench():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True)
    (TEST_ROOT / "data").mkdir()
    (TEST_ROOT / "database").mkdir()


def generate_mass_data(file_count=100, lines_per_file=50):
    print(f"{Colors.OKCYAN}🏗️  Generating {file_count} test files...{Colors.ENDC}")
    data_dir = TEST_ROOT / "data"
    for i in range(file_count):
        with open(data_dir / f"test_doc_{i}.txt", "w") as f:
            f.write(f"Document ID: {i}\nSecret Key: KEOVIL-SIGMA-{i}\n")
            f.write(
                "Optimizing RAG performance for RTX 5060 Ti 16GB VRAM.\n"
                * lines_per_file
            )


def run_stress_test():
    setup_test_bench()
    generate_mass_data()

    print(f"\n{Colors.HEADER}🚀 INITIALIZING KEOVIL TEST ENGINE{Colors.ENDC}")
    # This will load ColBERT onto your GPU
    rag = CollegeRAG(top_k=3)

    # TEST A: Initial Sync Speed
    start_sync = time.time()
    print(f"\n{Colors.OKBLUE}[Test A] Vectorizing 100 Files...{Colors.ENDC}")
    rag._initial_sync()
    print(f"{Colors.OKGREEN}✅ Sync Time: {time.time() - start_sync:.2f}s{Colors.ENDC}")

    # TEST B: Retrieval
    print(f"\n{Colors.OKBLUE}[Test B] Testing Retrieval Accuracy...{Colors.ENDC}")
    q = "What is the secret key for document 88?"
    start_q = time.time()
    answer = rag.ask(q)
    print(f"{Colors.OKGREEN}🤖 Lora: {answer}{Colors.ENDC}")
    print(f"⏱️  Latency: {time.time() - start_q:.2f}s")


if __name__ == "__main__":
    run_stress_test()
