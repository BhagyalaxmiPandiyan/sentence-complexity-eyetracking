"""
Script to patch nl-randvar.h (uncomment JointArrayRV/DelimitedJointArrayRV)
and compile the modelblocks left-corner parser binary.
"""
import os
import re
import subprocess
import sys

MBDIR = "/mnt/e/Mtech Project/Updated Folder/modelblocks-release"
RVTL = f"{MBDIR}/resource-rvtl"
INCLUDE = f"{MBDIR}/resource-lcparse/include"
SRC = f"{MBDIR}/resource-lcparse/src/parser-x-efabp.cpp"
OUT_BIN = "/mnt/e/Project/Final One/parser-x-efabp"
PATCHED_RVTL = "/tmp/rvtl-patched"

def patch_nl_randvar():
    src_path = f"{RVTL}/nl-randvar.h"
    dst_path = f"{PATCHED_RVTL}/nl-randvar.h"

    with open(src_path, "r") as f:
        lines = f.readlines()

    in_comment = False
    start_line = -1
    end_line = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "/*" and not in_comment and i > 500:
            in_comment = True
            start_line = i
        elif stripped == "*/" and in_comment:
            end_line = i
            in_comment = False
            break

    if start_line < 0 or end_line < 0:
        print(f"Could not find comment block (start={start_line}, end={end_line})")
        sys.exit(1)

    print(f"Uncommenting lines {start_line+1} to {end_line+1} in nl-randvar.h")
    lines[start_line] = "// (JointArrayRV uncommented for compilation)\n"
    lines[end_line] = "// (end JointArrayRV)\n"

    os.makedirs(PATCHED_RVTL, exist_ok=True)
    with open(dst_path, "w") as f:
        f.writelines(lines)
    print(f"Patched header written to {dst_path}")
    return dst_path

def compile_parser():
    # Also copy other headers from rvtl to the patched dir
    import shutil
    for fname in os.listdir(RVTL):
        src = os.path.join(RVTL, fname)
        dst = os.path.join(PATCHED_RVTL, fname)
        if fname != "nl-randvar.h" and os.path.isfile(src):
            shutil.copy2(src, dst)

    cmd = [
        "g++",
        f"-I{INCLUDE}",
        f"-I{PATCHED_RVTL}",
        "-Wall", "-fpermissive", "-std=c++11", "-lm",
        "-Wno-deprecated",
        SRC,
        "-o", OUT_BIN
    ]
    print("Compiling:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Compilation SUCCEEDED!")
        print(f"Binary at: {OUT_BIN}")
    else:
        print("Compilation FAILED:")
        print(result.stderr[:3000])
    return result.returncode

if __name__ == "__main__":
    patch_nl_randvar()
    rc = compile_parser()
    sys.exit(rc)
