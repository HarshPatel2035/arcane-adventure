#!/bin/zsh
# Records label 5 "Idle / No Spell": relaxed hands, scratching, reaching for keys.
cd -- "${0:A:h}" || exit 1
.venv/bin/python collect_data.py --guided --include-idle "$@"
result=$?
if (( result != 0 )); then
    read "reply?Collection failed. Press Enter to close..."
fi
exit $result
