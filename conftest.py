import os
import sys

# Make the repo root importable so tests can `import main` / `import k8s_utils`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
