if(NOT EXISTS "${CKPT}")
  message(FATAL_ERROR "Octo L1 GGUF not found: ${CKPT}")
endif()

get_filename_component(DUMP_PARENT "${DUMP_DIR}" DIRECTORY)
file(MAKE_DIRECTORY "${DUMP_PARENT}")

set(UNNORM_DATASET_ARGS "")
if(DEFINED UNNORM_DATASET AND NOT "${UNNORM_DATASET}" STREQUAL "")
  list(APPEND UNNORM_DATASET_ARGS --unnorm-dataset "${UNNORM_DATASET}")
endif()

# TIP-06: octo_l1_parity_dump runs octo_dump_l1_stagewise_case_resident -- the head_type=l1
# resident stage functions (proprio tokenizer, L1 MAPHead), NOT the diffusion path. See
# octo.cpp's own head_type=="l1" gate in that function for the call-site proof this
# grep-verified at TIP-06 report time.
execute_process(
  COMMAND "${DUMPER}" --ckpt "${CKPT}" --case "${CASE_DIR}" ${UNNORM_DATASET_ARGS} --out "${DUMP_DIR}"
  RESULT_VARIABLE dump_rc
)
if(NOT dump_rc EQUAL 0)
  message(FATAL_ERROR "octo_l1_parity_dump failed: ${dump_rc}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" "${VERIFY}" --golden "${CASE_DIR}" --dump "${DUMP_DIR}" --tol "${TOL}"
  RESULT_VARIABLE verify_rc
)
if(NOT verify_rc EQUAL 0)
  message(FATAL_ERROR "verify_octo_l1_parity.py failed: ${verify_rc}")
endif()
