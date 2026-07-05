# -*- coding: utf-8 -*-
"""Run all kreaphoton plain-assert tests in sequence (no pytest in the
ComfyUI embedded interpreter - planning-council H9/D6).

Run: <embedded python> tests/run_tests.py
"""
import subprocess
import sys
import os

TESTS = [
    "test_schedules.py",
    "test_guidance.py",
    "test_noise.py",
    "test_variety.py",
    "test_sampling.py",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable
    failures = []
    for t in TESTS:
        path = os.path.join(here, t)
        print("\n" + "#" * 78)
        print("# RUNNING: %s" % t)
        print("#" * 78)
        result = subprocess.run([python, path])
        if result.returncode != 0:
            failures.append(t)

    print("\n" + "=" * 78)
    if failures:
        print("run_tests: FAILED - %s" % ", ".join(failures))
        sys.exit(1)
    else:
        print("run_tests: ALL %d TEST FILES PASSED" % len(TESTS))


if __name__ == "__main__":
    main()
