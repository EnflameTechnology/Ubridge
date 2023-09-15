#/bin/bash

SCRIPT_PATH=`dirname $0`
KERNEL_PATH=`realpath $SCRIPT_PATH"/../kernels/dorado/"`

if [ $# -gt 1 ]; then
    echo "please specify kernel-name to be built"
else
    topscc $KERNEL_PATH"/../"$1.cpp -arch gcu210 -ltops -std=c++17 -lpthread -O0 -o $KERNEL_PATH"/"$1.o0.out --save-temps
fi

# cleaning up tmp files
mv $1*.topsfb $KERNEL_PATH/$1.topsfb
rm $1-tops-dtu-enflame-tops-gcu210.*
rm $1-host-x86_64-unknown-linux-gnu.*
rm $1-tops-dtu-enflame-tops-gcu210*
rm a.out*
