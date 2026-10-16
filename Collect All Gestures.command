#!/bin/zsh
cd -- "${0:A:h}" || exit 1
.venv/bin/python collect_data.py --guided "$@"
result=$?
if (( result != 0 )); then
    read "reply?Collection failed. Press Enter to close..."
fi
exit $result
