# vla.cpp tests, enabled with -DVLA_BUILD_TESTS=ON. predict_check needs a real
# GGUF so it is built but not run by ctest; the unit tests below are pure.

find_package(Python3 COMPONENTS Interpreter)

add_executable(vla_predict_check predict_check.cpp)
target_link_libraries(vla_predict_check PRIVATE vla_core)
target_compile_options(vla_predict_check PRIVATE -Wall -Wextra)

add_executable(octo_bench octo_bench.cpp)
target_include_directories(octo_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(octo_bench PRIVATE vla_core)
target_compile_options(octo_bench PRIVATE -Wall -Wextra)

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

# TIP-ND1-B: registers one ctest case running octo_parity_dump (now resident-only -- the
# gốc/"_resident" split this helper used to register in parallel, TIP-ND1-A, was collapsed
# back to one test per case once TIP-ND1-A proved resident-on-CPU is bit-exact with the
# now-deleted gốc path and octo.cpp was unified onto it, TIP-ND1-B). Test names are unchanged
# from before TIP-ND1-A.
function(add_octo_parity_test)
  set(oneValueArgs NAME CKPT CASE_DIR T5_INJECT TRANSFORMER_TOL UNNORM_DATASET KNOWN_MISMATCH EXCLUDE DUMP_DIR)
  cmake_parse_arguments(APT "" "${oneValueArgs}" "" ${ARGN})
  add_test(
    NAME ${APT_NAME}
    COMMAND ${CMAKE_COMMAND}
      -DCKPT=${APT_CKPT}
      -DCASE_DIR=${APT_CASE_DIR}
      -DT5_INJECT=${APT_T5_INJECT}
      -DTRANSFORMER_TOL=${APT_TRANSFORMER_TOL}
      -DUNNORM_DATASET=${APT_UNNORM_DATASET}
      -DKNOWN_MISMATCH=${APT_KNOWN_MISMATCH}
      -DEXCLUDE=${APT_EXCLUDE}
      -DDUMP_DIR=${APT_DUMP_DIR}
      -DDUMPER=$<TARGET_FILE:octo_parity_dump>
      -DVERIFY=${CMAKE_SOURCE_DIR}/scripts/verify_octo_parity.py
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -P ${CMAKE_SOURCE_DIR}/tests/run_octo_parity.cmake)
endfunction()

if(Python3_FOUND AND DEFINED ENV{VLA_OCTO_GOLDEN_DIR})
  foreach(OCTO_PARITY_CASE example_batch synthetic_zero synthetic_random second_timestep_padded wrist_valid_synthetic synthetic_ramp)
    if(EXISTS "$ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy")
      # Oracle mode: --t5-inject feeds the golden tokens downstream; native T5 output is still
      # dumped as "t5.out" and checked against golden independently (M5, TIP-006).
      add_octo_parity_test(
        NAME octo_parity_${OCTO_PARITY_CASE}
        CKPT ${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
        CASE_DIR $ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}
        T5_INJECT $ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy
        TRANSFORMER_TOL 2e-3
        UNNORM_DATASET ${OCTO_UNNORM_DATASET}
        DUMP_DIR ${CMAKE_BINARY_DIR}/octo_parity/${OCTO_PARITY_CASE})
      # Native mode: no --t5-inject, T5-base encoder output drives the whole pipeline end-to-end.
      add_octo_parity_test(
        NAME octo_parity_native_${OCTO_PARITY_CASE}
        CKPT ${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
        CASE_DIR $ENV{VLA_OCTO_GOLDEN_DIR}/${OCTO_PARITY_CASE}
        TRANSFORMER_TOL 2e-3
        UNNORM_DATASET ${OCTO_UNNORM_DATASET}
        DUMP_DIR ${CMAKE_BINARY_DIR}/octo_parity_native/${OCTO_PARITY_CASE})
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
    add_octo_parity_test(
      NAME octo_parity_bridge_debug
      CKPT ${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
      CASE_DIR ${OCTO_BRIDGE_DEBUG_DIR}
      T5_INJECT ${OCTO_BRIDGE_DEBUG_DIR}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy
      TRANSFORMER_TOL 2e-3
      UNNORM_DATASET ${OCTO_UNNORM_DATASET}
      KNOWN_MISMATCH ${OCTO_BRIDGE_DEBUG_KNOWN_MISMATCH}
      DUMP_DIR ${CMAKE_BINARY_DIR}/octo_parity/bridge_debug)
    add_octo_parity_test(
      NAME octo_parity_native_bridge_debug
      CKPT ${CMAKE_SOURCE_DIR}/octo-small-1.5-f32.gguf
      CASE_DIR ${OCTO_BRIDGE_DEBUG_DIR}
      TRANSFORMER_TOL 2e-3
      UNNORM_DATASET ${OCTO_UNNORM_DATASET}
      KNOWN_MISMATCH ${OCTO_BRIDGE_DEBUG_KNOWN_MISMATCH}
      DUMP_DIR ${CMAKE_BINARY_DIR}/octo_parity_native/bridge_debug)
  endif()
endif()

# LIBERO (window=1): golden traces from the cyrusneary octo-finetuned-libero checkpoint
# (TIP-GOLD), a genuinely single-camera model -- its example_batch/finetune_config never fed
# a wrist observation at all, so the golden has no octo_transformer.obs_wrist.* /
# block_transformer...obs_wrist tensors, and its block_transformer.{input_tokens,attention_mask,
# output_tokens} are shorter than the port's (vla.cpp always includes masked-but-present wrist
# tokens in the full sequence -- numerically equivalent for downstream outputs, since a masked
# token contributes nothing, but shape-incompatible for those specific full-sequence/wrist
# boundaries). OCTO_LIBERO_EXCLUDE drops exactly those from comparison; every other boundary
# (obs_primary, task_language, readout, diff.*, final_action(+unnormalized)) is still compared
# at the same tolerance as bridge. See TIP-HARNESS report for the full rationale.
# Gated on two env vars since neither the golden traces nor the LIBERO GGUF are committed:
#   VLA_OCTO_LIBERO_GOLDEN_DIR -- root containing <case>/tensors/ + manifest.json per case
#   VLA_OCTO_LIBERO_CKPT       -- path to the window=1 GGUF (scripts/convert_octo_to_gguf.py
#                                 --ckpt <cyrusneary experiment dir> --step 60000)
set(OCTO_LIBERO_UNNORM_DATASET libero_object)
set(OCTO_LIBERO_EXCLUDE "bt.input,bt.mask,bt.output,obs_wrist")

if(Python3_FOUND AND DEFINED ENV{VLA_OCTO_LIBERO_GOLDEN_DIR} AND DEFINED ENV{VLA_OCTO_LIBERO_CKPT})
  foreach(OCTO_LIBERO_CASE example_frame synthetic_zero synthetic_ramp)
    if(EXISTS "$ENV{VLA_OCTO_LIBERO_GOLDEN_DIR}/${OCTO_LIBERO_CASE}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy")
      # Oracle mode: --t5-inject feeds the golden tokens downstream; native T5 output is still
      # dumped as "t5.out" and checked against golden independently.
      add_octo_parity_test(
        NAME octo_parity_libero_${OCTO_LIBERO_CASE}
        CKPT $ENV{VLA_OCTO_LIBERO_CKPT}
        CASE_DIR $ENV{VLA_OCTO_LIBERO_GOLDEN_DIR}/${OCTO_LIBERO_CASE}
        T5_INJECT $ENV{VLA_OCTO_LIBERO_GOLDEN_DIR}/${OCTO_LIBERO_CASE}/tensors/octo_transformer.task_language.tokens_after_tokenizer.npy
        TRANSFORMER_TOL 2e-3
        UNNORM_DATASET ${OCTO_LIBERO_UNNORM_DATASET}
        EXCLUDE ${OCTO_LIBERO_EXCLUDE}
        DUMP_DIR ${CMAKE_BINARY_DIR}/octo_parity_libero/${OCTO_LIBERO_CASE})
      # Native mode: no --t5-inject, T5-base encoder output drives the whole pipeline end-to-end.
      add_octo_parity_test(
        NAME octo_parity_libero_native_${OCTO_LIBERO_CASE}
        CKPT $ENV{VLA_OCTO_LIBERO_CKPT}
        CASE_DIR $ENV{VLA_OCTO_LIBERO_GOLDEN_DIR}/${OCTO_LIBERO_CASE}
        TRANSFORMER_TOL 2e-3
        UNNORM_DATASET ${OCTO_LIBERO_UNNORM_DATASET}
        EXCLUDE ${OCTO_LIBERO_EXCLUDE}
        DUMP_DIR ${CMAKE_BINARY_DIR}/octo_parity_libero_native/${OCTO_LIBERO_CASE})
    endif()
  endforeach()
endif()

add_executable(test_vision_common test_vision_common.cpp)
target_include_directories(test_vision_common PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_vision_common PRIVATE -Wall -Wextra)
add_test(NAME vision_common COMMAND test_vision_common)

# Model-free tripwire for the final-action-slice formula (Blueprint risk item 1445/1492).
# Always registered (no golden dir / GGUF needed), runs in well under a second.
add_executable(octo_action_slice_test octo_action_slice_test.cpp)
target_compile_options(octo_action_slice_test PRIVATE -Wall -Wextra)
add_test(NAME octo_action_slice COMMAND octo_action_slice_test)
