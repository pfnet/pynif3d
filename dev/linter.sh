#!/bin/bash
# ==================================================================================================
# This file is adapted from https://github.com/facebookresearch/pytorch3d/blob/master/dev/linter.sh.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Run this script at project root by "./dev/linter.sh" before you commit.
# ==================================================================================================

# Halt on error.
set -e

# Check that the right version of "black" is installed.
{
  V=$(black --version|cut '-d ' -f3)
  code='import distutils.version; assert "19.3" < distutils.version.LooseVersion("'$V'")'
  python3 -c "${code}" 2> /dev/null
} || {
  echo "Linter requires black 19.3b0 or higher!"
  exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=$(dirname "${DIR}")

echo "Running isort..."
isort -y -sp "${DIR}"

echo "Running black..."
black "${DIR}"

echo "Running flake..."
flake8 "${DIR}"

echo "Running clang-format ..."
clang_format=$(command -v clang-format-8 || echo clang-format)
source_files=$(find "${DIR}" -regex ".*\.\(cpp\|c\|cc\|cu\|cuh\|cxx\|h\|hh\|hpp\|hxx\|tcc\|mm\|m\)" -print)

# Fix the "cannot use -i when reading from stdin" error when no source files are found.
if [[ -n "${source_files}" ]]
then
  eval "${clang_format}" "${source_files}"
fi

# Run arc and pyre internally only.
if [[ -f tests/TARGETS ]]
then
  (cd "${DIR}"; command -v arc > /dev/null && arc lint) || true

  echo "Running pyre..."
  echo "To restart/kill pyre server, run 'pyre restart' or 'pyre kill' in fbcode/"
  ( cd ~/fbsource/fbcode; pyre -l vision/fair/pytorch3d/ )
fi
