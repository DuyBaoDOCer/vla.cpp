if(NOT EXISTS "${CKPT}")
  message(FATAL_ERROR "Octo GGUF not found: ${CKPT}")
endif()

get_filename_component(DUMP_PARENT "${DUMP_DIR}" DIRECTORY)
file(MAKE_DIRECTORY "${DUMP_PARENT}")

set(T5_ARGS "")
if(DEFINED T5_INJECT AND NOT "${T5_INJECT}" STREQUAL "")
  list(APPEND T5_ARGS --t5-inject "${T5_INJECT}")
endif()

set(TRANSFORMER_TOL_ARGS "")
if(DEFINED TRANSFORMER_TOL AND NOT "${TRANSFORMER_TOL}" STREQUAL "")
  list(APPEND TRANSFORMER_TOL_ARGS --transformer-tol "${TRANSFORMER_TOL}")
endif()

set(UNNORM_DATASET_ARGS "")
if(DEFINED UNNORM_DATASET AND NOT "${UNNORM_DATASET}" STREQUAL "")
  list(APPEND UNNORM_DATASET_ARGS --unnorm-dataset "${UNNORM_DATASET}")
endif()

set(KNOWN_MISMATCH_ARGS "")
if(DEFINED KNOWN_MISMATCH AND NOT "${KNOWN_MISMATCH}" STREQUAL "")
  list(APPEND KNOWN_MISMATCH_ARGS --known-mismatch "${KNOWN_MISMATCH}")
endif()

set(EXCLUDE_ARGS "")
if(DEFINED EXCLUDE AND NOT "${EXCLUDE}" STREQUAL "")
  list(APPEND EXCLUDE_ARGS --exclude "${EXCLUDE}")
endif()

# TIP-ND1-B: octo_parity_dump always runs the resident (m->backend) path now -- the gốc
# disk-read dump path it used to alternate with (RESIDENT=1 toggle, TIP-ND1-A) was deleted
# once TIP-ND1-A proved resident-on-CPU is bit-exact with it (20/20 golden cases).
execute_process(
  COMMAND "${DUMPER}" --ckpt "${CKPT}" --case "${CASE_DIR}" ${T5_ARGS} ${UNNORM_DATASET_ARGS} --out "${DUMP_DIR}"
  RESULT_VARIABLE dump_rc
)
if(NOT dump_rc EQUAL 0)
  message(FATAL_ERROR "octo_parity_dump failed: ${dump_rc}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" "${VERIFY}" --golden "${CASE_DIR}" --dump "${DUMP_DIR}" ${TRANSFORMER_TOL_ARGS} ${KNOWN_MISMATCH_ARGS} ${EXCLUDE_ARGS} --report "${DUMP_DIR}/parity_report.json"
  RESULT_VARIABLE verify_rc
)
if(NOT verify_rc EQUAL 0)
  message(FATAL_ERROR "verify_octo_parity.py failed: ${verify_rc}")
endif()
