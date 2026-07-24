# vla.cpp tests, enabled with -DVLA_BUILD_TESTS=ON. predict_check needs a real
# GGUF so it is built but not run by ctest; the unit tests below are pure.

find_package(Python3 COMPONENTS Interpreter)

add_executable(vla_predict_check predict_check.cpp)
target_link_libraries(vla_predict_check PRIVATE vla_core)
target_compile_options(vla_predict_check PRIVATE -Wall -Wextra)

add_executable(octo_load_check octo_load_check.cpp)
target_include_directories(octo_load_check PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(octo_load_check PRIVATE vla_core)
target_compile_options(octo_load_check PRIVATE -Wall -Wextra)

add_executable(octo_parity_dump octo_parity_dump.cpp)
target_include_directories(octo_parity_dump PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(octo_parity_dump PRIVATE vla_core)
target_compile_options(octo_parity_dump PRIVATE -Wall -Wextra)
if(Python3_FOUND AND DEFINED ENV{VLA_OCTO_GOLDEN_DIR} AND EXISTS "$ENV{VLA_OCTO_GOLDEN_DIR}/example_batch")
  add_test(
    NAME octo_tokenizer_parity_example_batch
    COMMAND ${CMAKE_COMMAND}
      -DCKPT=${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
      -DCASE_DIR=$ENV{VLA_OCTO_GOLDEN_DIR}/example_batch
      -DDUMP_DIR=${CMAKE_BINARY_DIR}/octo_parity/example_batch
      -DDUMPER=$<TARGET_FILE:octo_parity_dump>
      -DVERIFY=${CMAKE_SOURCE_DIR}/scripts/verify_octo_parity.py
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -P ${CMAKE_SOURCE_DIR}/tests/run_octo_parity.cmake)
endif()

add_executable(test_vision_common test_vision_common.cpp)
target_include_directories(test_vision_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_vision_common PRIVATE -Wall -Wextra)
add_test(NAME vision_common COMMAND test_vision_common)
