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
    opt/tops/bin/topscc $KERNEL_PATH"/../"$1.cpp -arch gcu200 -ltops -std=c++17 -lpthread -O3 -o $KERNEL_PATH"/"$1.o0.out --save-temps
fi

# cleaning up tmp files
# rm $KERNEL_PATH/$1.topsfb
mv $KERNEL_PATH/$1.cpp-tops-dtu-enflame-tops.topsfb $KERNEL_PATH/$1.topsfb
rm ./*.topsi
rm ./*.bc
rm ./*.o
rm ./*.s
rm ./*.tmp
rm a.out*
rm *.o0.out

duration=$SECONDS
echo "Execution time: $duration seconds"
