#!/bin/sh
python3 -c 'import numpy' 2> /dev/null || pip3 install numpy --user
python3 main.py $@
