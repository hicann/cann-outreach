#! /bin/bash
SCRIPT_DIR=$(dirname "(realpath "$0")")
cd $SCRIPT_DIR || exit

rm -rf outputs
mkdir outputs

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

SOURCE_FILES=$(find . -name '*.cc')

g++ -std=c++14 -shared $SOURCE_FILES -o outputs/libcustom_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2