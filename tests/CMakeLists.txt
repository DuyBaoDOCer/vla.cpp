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
# All shipped golden traces (tier1 + tier2/bridge_debug) were generated against
# "bridge_dataset" statistics (manifest metadata.unnormalization.dataset /
# metadata.dataset, confirmed for every case below) -- passed explicitly here
# rather than left to octo_parity_dump's default, so the key is visible and easy
# to audit if a future golden case uses a different dataset.
set(OCTO_UNNORM_DATASET bridge_dataset)

if(Python3_FOUND AND DEFINED ENV{VLA_OCTO_GOLDEN_DIR})
  foreach(OCTO_PARITY_CASE example_batch synthetic_zero synthetic_random second_timestep_padded wrist_valid_synthetic synthetic_ramp)
    if(EXISTS "$ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy")
      # Oracle mode: --t5-inject feeds the golden tokens downstream; native T5 output is still
      # dumped as "t5.out" and checked against golden independently (M5, TIP-006).
      add_test(
        NAME octo_parity_${OCTO_PARITY_CASE}
        COMMAND ${CMAKE_COMMAND}
          -DCKPT=${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
          -DCASE_DIR=$ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}
          -DT5_INJECT=$ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy
          -DTRANSFORMER_TOL=2e-3
          -DUNNORM_DATASET=${OCTO_UNNORM_DATASET}
          -DDUMP_DIR=${CMAKE_BINARY_DIR}/octo_parity/${OCTO_PARITY_CASE}
          -DDUMPER=$<TARGET_FILE:octo_parity_dump>
          -DVERIFY=${CMAKE_SOURCE_DIR}/scripts/verify_octo_parity.py
          -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
          -P ${CMAKE_SOURCE_DIR}/tests/run_octo_parity.cmake)
      # Native mode: no --t5-inject, T5-base encoder output drives the whole pipeline end-to-end.
      add_test(
        NAME octo_parity_native_${OCTO_PARITY_CASE}
        COMMAND ${CMAKE_COMMAND}
          -DCKPT=${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
          -DCASE_DIR=$ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}
          -DTRANSFORMER_TOL=2e-3
          -DUNNORM_DATASET=${OCTO_UNNORM_DATASET}
          -DDUMP_DIR=${CMAKE_BINARY_DIR}/octo_parity_native/${OCTO_PARITY_CASE}
          -DDUMPER=$<TARGET_FILE:octo_parity_dump>
          -DVERIFY=${CMAKE_SOURCE_DIR}/scripts/verify_octo_parity.py
          -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
          -P ${CMAKE_SOURCE_DIR}/tests/run_octo_parity.cmake)
    endif()
  endforeach()
endif()

# tier2/bridge_debug: real (non-synthetic) images + label evaluation, exercised the
# same way as tier1 (oracle + native), one directory level up from VLA_OCTO_GOLDEN_DIR
# (.../golden_traces/octo_small_tier1 -> .../golden_traces/octo_small_tier2/bridge_debug),
# gated on VLA_OCTO_GOLDEN_DIR_TIER2 so it stays opt-in when that data isn't present.
if(Python3_FOUND AND DEFINED ENV{VLA_OCTO_GOLDEN_DIR_TIER2})
  set(OCTO_BRIDGE_DEBUG_DIR "$ENV{VLA_OCTO_GOLDEN_DIR_TIER2}/bridge_debug")
  # bridge_debug's own dataset_statistics.action.{mean,std} (recorded per-case in its
  # golden manifest, sourced from its debug-only "./tests/debug_dataset") do NOT match
  # the octo-small-1.5 checkpoint's "bridge_dataset" stats embedded in the GGUF -- both
  # are labeled "bridge_dataset" but are numerically different snapshots (verified: our
  # unnormalize formula reproduces bridge_debug's golden action_final_unnormalized to
  # ~2e-9 when fed bridge_debug's OWN recorded mean/std, so the formula is correct; the
  # mismatch is golden-data provenance, not a vla.cpp defect -- see TIP-008 Completion
  # Report. All 6 tier1 cases' own recorded stats match the GGUF exactly, so this is
  # specific to the bridge_debug trace, not our GGUF conversion.
  set(OCTO_BRIDGE_DEBUG_KNOWN_MISMATCH
    "action_final_unnormalized:bridge_debug's golden dataset_statistics differs numerically from the checkpoint's embedded bridge_dataset stats (see TIP-008 report); unnormalize formula itself is verified correct")
  if(EXISTS "${OCTO_BRIDGE_DEBUG_DIR}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy")
    add_test(
      NAME octo_parity_bridge_debug
      COMMAND ${CMAKE_COMMAND}
        -DCKPT=${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
        -DCASE_DIR=${OCTO_BRIDGE_DEBUG_DIR}
        -DT5_INJECT=${OCTO_BRIDGE_DEBUG_DIR}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy
        -DTRANSFORMER_TOL=2e-3
        -DUNNORM_DATASET=${OCTO_UNNORM_DATASET}
        -DKNOWN_MISMATCH=${OCTO_BRIDGE_DEBUG_KNOWN_MISMATCH}
        -DDUMP_DIR=${CMAKE_BINARY_DIR}/octo_parity/bridge_debug
        -DDUMPER=$<TARGET_FILE:octo_parity_dump>
        -DVERIFY=${CMAKE_SOURCE_DIR}/scripts/verify_octo_parity.py
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -P ${CMAKE_SOURCE_DIR}/tests/run_octo_parity.cmake)
    add_test(
      NAME octo_parity_native_bridge_debug
      COMMAND ${CMAKE_COMMAND}
        -DCKPT=${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
        -DCASE_DIR=${OCTO_BRIDGE_DEBUG_DIR}
        -DTRANSFORMER_TOL=2e-3
        -DUNNORM_DATASET=${OCTO_UNNORM_DATASET}
        -DKNOWN_MISMATCH=${OCTO_BRIDGE_DEBUG_KNOWN_MISMATCH}
        -DDUMP_DIR=${CMAKE_BINARY_DIR}/octo_parity_native/bridge_debug
        -DDUMPER=$<TARGET_FILE:octo_parity_dump>
        -DVERIFY=${CMAKE_SOURCE_DIR}/scripts/verify_octo_parity.py
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -P ${CMAKE_SOURCE_DIR}/tests/run_octo_parity.cmake)
  endif()
endif()

add_executable(test_vision_common test_vision_common.cpp)
target_include_directories(test_vision_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_vision_common PRIVATE -Wall -Wextra)
add_test(NAME vision_common COMMAND test_vision_common)
