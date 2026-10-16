#!/bin/zsh
cd -- "${0:A:h}" || exit 1
.venv/bin/python inference.py "$@"
result=$?
if (( result != 0 )); then
    read "reply?Recognition failed. Press Enter to close..."
fi
exit $result
