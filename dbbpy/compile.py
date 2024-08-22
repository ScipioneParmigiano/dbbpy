import os
import sys
import numpy as np
import pybind11
from setuptools import Extension, Distribution
from setuptools.command.build_ext import build_ext

def get_mosek_include_dir():
    if sys.platform == "win32":
        return os.getenv('MOSEK_INCLUDE_DIR', r"C:\mosek\mosek\10.2\tools\platform\win64x86\h")
    elif sys.platform == "darwin":
        return os.getenv('MOSEK_INCLUDE_DIR', "/usr/local/mosek/10.2/tools/platform/osx64x86/h")
    else:  
        return os.getenv('MOSEK_INCLUDE_DIR', "/usr/include/mosek/10.2/tools/platform/linux64x86/h")

def get_mosek_lib_dir():
    if sys.platform == "win32":
        return os.getenv('MOSEK_LIB_DIR', r"C:\mosek\mosek\10.2\tools\platform\win64x86\bin")
    elif sys.platform == "darwin":
        return os.getenv('MOSEK_LIB_DIR', "/usr/local/mosek/10.2/tools/platform/osx64x86/bin")
    else:
        return os.getenv('MOSEK_LIB_DIR', "/usr/include/mosek/10.2/tools/platform/linux64x86/bin")

def get_mosek_libs():
    if sys.platform == "win32":
        return ["mosek64_10_2", "fusion64_10_2"]
    else: 
        return ["mosek64", "fusion64"]

def get_extra_link_args():
    if sys.platform == "win32":
        return []  # Windows typically doesn't need this
    else: 
        return ["-lmosek64", "-lfusion64"]

def get_main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        # For Windows
        cpp_file = os.path.join(script_dir, "src", "main.cpp")
    else:
        # For Unix-like systems
        cpp_file = os.path.join(script_dir, "src", "main.cpp")

    print(f"Path to the C++ source file: {cpp_file}")
    return [cpp_file]

build_dir = os.path.join(os.path.dirname(__file__), "build")
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)


def compile_bindings():
    eigen_include_dir = os.getenv('EIGEN_INCLUDE_DIR', r"C:\ProgramData\chocolatey\lib\eigen\include\eigen3")

    ext_modules = [
        Extension(
            "bindings",
            get_main(),  # C++ source file
            include_dirs=[
                np.get_include(),
                pybind11.get_include(),
                eigen_include_dir,
                get_mosek_include_dir()
            ],
            library_dirs=[get_mosek_lib_dir()],
            libraries=get_mosek_libs(),
            extra_compile_args=["-std=c++11"],
            extra_link_args=get_extra_link_args(),
            language="c++"
        ),
    ]

    # Create a dummy distribution instance
    dist = Distribution()
    dist.ext_modules = ext_modules

    # Create and run the build_ext command
    build_ext_cmd = build_ext(dist)
    
    # Override the build directory to dbbpy
    build_ext_cmd.build_lib = os.path.join(os.path.dirname(__file__))

    build_ext_cmd.finalize_options()
    build_ext_cmd.run()

    # Print the location of the compiled shared object
    for ext in ext_modules:
        shared_obj_path = build_ext_cmd.get_ext_fullpath(ext.name)
        print(f"Shared object compiled at: {shared_obj_path}")

if __name__ == "__main__":
    compile_bindings()


