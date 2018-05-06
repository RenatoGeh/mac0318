#!/bin/bash

reqs="ipykernel ipython ipywidgets jsonschema jupyter matplotlib notebook numpy opencv pandas pandocfilters pillow scikit scipy seaborn six tensorboard termcolor"

for l in $reqs; do
  pacman -Qs $l
done
