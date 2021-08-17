#!/bin/bash -e

function run_unit_tests() {
  export PYTHONPATH=$(pwd):$PYTHONPATH
  python3 tests/run_tests.py
}

function run_lint_check() {
  bash .pfnci/linter.bash
}

function run_install_check() {
  python3 setup.py install --user
}

function run_all_checks() {
  run_unit_tests
  run_lint_check
  run_install_check
}

run_all_checks
