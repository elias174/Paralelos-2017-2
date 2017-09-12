#!/bin/sh

make
echo ssh -C evalencia@manati \"cd CPrograms \&\& make $*\"
ssh -C evalencia@manati "cd CPrograms && make $*"
