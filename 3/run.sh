#!/bin/bash
function need {
python3 -c "import $1" 2> /dev/null || pip3 install $1 --user
}
need pip
need numpy
need scipy
need sklearn
need matplotlib
