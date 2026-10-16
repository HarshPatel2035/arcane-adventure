#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
    print "Project environment is missing. Recreate .venv before starting."
    read "reply?Press Enter to close..."
    exit 1
fi
.venv/bin/python collect_data.py "$@"
result=$?
if (( result != 0 )); then
    read "reply?Collection failed. Press Enter to close..."
fi
exit $result
