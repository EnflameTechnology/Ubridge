#/bin/bash

SCRIPT_PATH=`dirname $0`
KERNEL_PATH=`realpath $SCRIPT_PATH"/../kernels/scorpio/"`

set +e
set -o xtrace

SECONDS=0

# cleaning up tmp files
mv ./unary*.topsfb $KERNEL_PATH/unary.topsfb
mv ./activation*.topsfb $KERNEL_PATH/activation.topsfb
mv ./batch_matmul*.topsfb $KERNEL_PATH/batch_matmul.topsfb
mv ./convolution*.topsfb $KERNEL_PATH/convolution.topsfb
mv ./element*.topsfb $KERNEL_PATH/element.topsfb
mv ./transpose*.topsfb $KERNEL_PATH/transpose.topsfb
mv ./fused_batch_matmul*.topsfb $KERNEL_PATH/fused_batch_matmul.topsfb
mv ./transposed_matmul*.topsfb $KERNEL_PATH/transposed_matmul.topsfb

rm ./*.topsi
rm ./*.bc
rm ./*.o
rm ./*.s
rm ./*.tmp
rm ./*.topsfb
rm a.out*
rm temp.o0.out


duration=$SECONDS
echo "Execution time: $duration seconds"

