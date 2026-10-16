#!/bin/zsh
cd -- "${0:A:h}" || exit 1
print "Launching Arcane Adventure - The Enchanted Grove"
.venv/bin/python arcane_adventure.py "$@"
result=$?
if (( result != 0 )); then
    read "reply?Game failed. Press Enter to close..."
fi
exit $result
