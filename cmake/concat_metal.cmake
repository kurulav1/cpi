# Concatenate the Metal shader family files into one translation unit for the offline compiler.
# Invoked as: cmake -DOUT=<combined> "-DPARTS=<a;b;c>" -P concat_metal.cmake
#
# The runtime loader (runtime/metal_context.mm) does the same concatenation in filename order for
# the no-toolchain path; this keeps the offline metallib byte-for-byte the same source. The parts
# carry no #include, so concatenation is the whole story; no include resolution either side.
file(WRITE "${OUT}" "")
foreach(part IN LISTS PARTS)
  file(READ "${part}" contents)
  file(APPEND "${OUT}" "${contents}")
  file(APPEND "${OUT}" "\n")
endforeach()
