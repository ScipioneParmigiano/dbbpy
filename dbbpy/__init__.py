import importlib
import os
import sys

original_sys_path = sys.path.copy()

# Restrict sys.path to only the current directory
sys.path = [os.path.dirname(__file__)]

print("\n\n")
print(os.path.dirname(__file__))

try:
    # Attempt to load the real shared object from the current directory
    from .bindings import *
except ImportError:
    print(
        "The 'bindings' shared object could not be loaded. "
        "Please compile the C++ bindings by running 'compile_bindings' in a terminal and make sure the shared object file is in the library directory."
    )

finally:
    sys.path = original_sys_path
