#!/bin/zsh
cd -- "${0:A:h}" || exit 1
.venv/bin/python train_model.py "$@"
result=$?
read "reply?Press Enter to close..."
exit $result
