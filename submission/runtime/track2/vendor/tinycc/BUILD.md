# Offline C launcher compiler

TinyCC upstream commit: `0fb54300b56512754221d80adda85ddb9815bceb`.
Source: https://repo.or.cz/tinycc.git/commit/0fb54300b56512754221d80adda85ddb9815bceb
Archive: `tinycc-source.tar.gz`, SHA256 `7cfbbc7e67ceffa035ef8681c551434ca8dbdc3b9bac44639913177b1a7bfee6`.
License: GNU LGPL; see accompanying `COPYING` and complete corresponding source.

Built from the unmodified source on Ubuntu 22.04 x86_64 with GCC 11.4:

```sh
mkdir source
tar -xzf tinycc-source.tar.gz -C source --strip-components=1
cd source
./configure --prefix=/absolute/path/to/install
make -j4
make install
```

The distributed files are `bin/tcc` and `lib/tcc` from that installation.
The `cc` wrapper supplies the relative `-B` path so the installation prefix
is not needed at runtime. Its highest required glibc symbol version is 2.34; the official
image uses Ubuntu 22.04 / glibc 2.35. This compiles only Triton's C launchers;
Triton's own toolchain compiles CUDA kernels. Nothing is downloaded or
installed during evaluation. The original uncompiled model is a fallback.
