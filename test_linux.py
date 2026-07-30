#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    if sys.platform != "linux":
        print("This script is intended for Linux.")
        return 1
    import argparse

    parser = argparse.ArgumentParser(description="Test Linux file move script")
    parser.add_argument("--dest", "-d", help="destination directory", default="/tmp/test_linux_destination")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="tmp_test_linux_", suffix=".txt", delete=False, dir="/tmp") as tmp_file:
        tmp_file.write(b"Linux temp file test\n")
        temp_path = Path(tmp_file.name)

    destination_path = dest_dir / temp_path.name
    shutil.move(str(temp_path), str(destination_path))
    print(f"Moved {temp_path} to {destination_path}")
    os.system("/usr/bin/shutdown")
    return 0


if __name__ == "__main__":
    main()
