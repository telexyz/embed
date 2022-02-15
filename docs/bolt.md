## Bolt: "Based On Lookup Tables"
https://github.com/dblalock/bolt

```sh
brew install bazel
cd bolt/cpp && bazel run :main
```

Bolt is an algorithm for compressing vectors of real-valued data and running mathematical operations directly on the compressed representations. If you have a large collection of mostly-dense vectors and can tolerate lossy compression, Bolt can probably save you 10-200x space and compute time. Bolt also has theoretical guarantees bounding the errors in its approximations.

EDIT: this repo now also features the source code for MADDNESS, our shiny new algorithm for approximate matrix multiplication. MADDNESS has no Python wrapper yet, and is referred to as "mithral" in the source code. Name changed because apparently I'm the only who gets Lord of the Rings references. MADDNESS runs ridiculously fast and, under reasonable assumptions, requires zero multiply-adds. Realistically, it'll be most useful for speeding up neural net inference on CPUs, but it'll take another couple papers to get it there; we need to generalize it to convolution and write the CUDA kernels to allow GPU training.

- - -

Bolt currently only supports machines with AVX2 instructions, which basically means x86 machines from fall 2013 or later. Contributions for ARM support are welcome. Also note that the Bolt Python wrapper is currently configured to require Clang, since GCC apparently runs into issues.