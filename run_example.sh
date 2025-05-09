#!/bin/sh

export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0

sproot=$PWD

cd examples
dirs=()
ndir=0
for FILE in `ls -l`
do
    if test -d $FILE; then
      dirs+=( $FILE )
      ndir=$[ndir+1]
    fi
done

re='^[0-9]+$'

if [[ $1 =~ $re ]] && [ $1 -gt 0 ] && [ $1 -le $ndir ]; then
  ie=$1
else
  echo "Enter the index of the example to run (1-$ndir):"

  for ((i=0; i<$ndir; i++)); do      
      echo "$[i+1]) ${dirs[i]}"
  done

  while true; do
    read ie
    if [[ $ie =~ $re ]] && [ $ie -gt 0 ] && [ $ie -le $ndir ]; then
      break
    else
      echo "Please input a number between 1 and $ndir"
    fi
  done
fi

dir=${dirs[ie-1]}
echo "Running example/"$dir

cd $dir

rm -rf output/*
rm -rf scratch/*

export PYTHONPATH=$sproot
if [[ $2 == "--submit" ]]; then
  python $sproot"/scripts/spsubmit" --workdir=$PWD --add_path=yes
else
  python $sproot"/scripts/sprun" --workdir=$PWD
fi

python $sproot"/scripts/spplot" $PWD/output $PWD/output_ref
