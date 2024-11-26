#!/bin/bash

# Update package lists and install required libraries
apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 software-properties-common wget

# Upgrade glibc
GLIBC_VERSION=2.38
cd /tmp
wget http://ftp.gnu.org/gnu/libc/glibc-$GLIBC_VERSION.tar.gz
tar -xvzf glibc-$GLIBC_VERSION.tar.gz
cd glibc-$GLIBC_VERSION
mkdir build
cd build
../configure --prefix=/opt/glibc-$GLIBC_VERSION
make -j$(nproc)
make install

# Export the new glibc path
export LD_LIBRARY_PATH=/opt/glibc-$GLIBC_VERSION/lib:$LD_LIBRARY_PATH
export PATH=/opt/glibc-$GLIBC_VERSION/bin:$PATH

# Verify the version
ldd --version
