#/bin/bash

set +e
set -o xtrace

SECONDS=0

SCRIPT_PATH=`dirname $0`
KERNEL_PATH=`realpath $SCRIPT_PATH"/../kernels/pavo/"`

cd $KERNEL_PATH

if [ $# -gt 1 ]; then
    echo "please specify kernel-name to be built"
else
    topscc $KERNEL_PATH"/../"$1.cpp -arch gcu200 -ltops -std=c++17 -lpthread -O0 -o $KERNEL_PATH"/"$1.o0.out --save-temps
fi

# cleaning up tmp files
mv ./$1*.topsfb ./$1.topsfb
rm ./*.topsi
rm ./*.bc
rm ./*.o
rm ./*.s
rm ./*.tmp
rm a.out*
rm *.o0.out

duration=$SECONDS
echo "Execution time: $duration seconds"
