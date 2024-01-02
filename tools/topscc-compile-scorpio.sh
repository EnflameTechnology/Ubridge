#/bin/bash

set +e
set -o xtrace

SECONDS=0

SCRIPT_PATH=`dirname $0`
KERNEL_PATH=`realpath $SCRIPT_PATH"/../kernels/scorpio/"`
ATOMIC_INCLUDE_PATH="/opt/tops/include"
ATOMIC_LIB_PATH="/opt/tops/lib/acore"

cd $KERNEL_PATH

if [ $# -gt 1 ]; then
    echo "please specify kernel-name to be built"
else
    # topscc $KERNEL_PATH"/../"$1.cpp -arch gcu300 -ltops -std=c++17 -lpthread -O3 -o $KERNEL_PATH"/"$1.o0.out --save-temps
    topscc $KERNEL_PATH"/../"$1.cpp -arch gcu300 -O3 -std=c++17 -fPIC -ltops -o $KERNEL_PATH/$1.topsfb -Xclang -fallow-half-arguments-and-returns -D__GCU_ARCH__=300 -D__KRT_ARCH__=300 --tops-device-lib-path=$ATOMIC_LIB_PATH --tops-device-lib=libacore.bc -D__ATOMIC_OP__ -I$ATOMIC_INCLUDE_PATH --save-temps

fi

# cleaning up tmp files
rm $KERNEL_PATH/$1.topsfb
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
