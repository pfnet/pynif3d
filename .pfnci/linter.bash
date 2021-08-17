#!/bin/bash

# Make a copy of the current directory.
root_dir="$(pwd .)"
lint_dir="${root_dir}.lint"
cp -r "$root_dir" "$lint_dir"

# Run the lint checks. On error, terminate the script.
cd "$lint_dir" || exit 1
bash ./dev/linter.sh
exit_code=$?
echo "exit_code of linter is $exit_code"
[ $exit_code -eq 0 ] || exit 1
cd "$root_dir" || (echo "root_dir $root_dir cannot be accessed" && exit 1)

# Compare the files recursively and capture the exit code.
diff -qr "$root_dir" "$lint_dir"
exit_code=$?
echo "exit_code for diff command is $exit_code"

# Clean-up.
rm -rf "$lint_dir"
exit "$exit_code"