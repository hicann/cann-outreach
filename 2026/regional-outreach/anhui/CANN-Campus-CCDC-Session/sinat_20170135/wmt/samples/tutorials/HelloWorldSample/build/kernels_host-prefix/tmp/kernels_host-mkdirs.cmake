# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/ascendc_kernel_cmake/host_project")
  file(MAKE_DIRECTORY "/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/ascendc_kernel_cmake/host_project")
endif()
file(MAKE_DIRECTORY
  "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/src/kernels_host-build"
  "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix"
  "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/tmp"
  "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/src/kernels_host-stamp"
  "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/src"
  "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/src/kernels_host-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/src/kernels_host-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/class/26EAIHFSIC001/wmt/samples/tutorials/HelloWorldSample/build/kernels_host-prefix/src/kernels_host-stamp${cfgdir}") # cfgdir has leading slash
endif()
