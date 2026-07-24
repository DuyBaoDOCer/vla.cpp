if(NOT EXISTS "${CKPT}")
  message(FATAL_ERROR "Octo GGUF not found: ${CKPT}")
endif()

get_filename_component(DUMP_PARENT "${DUMP_DIR}" DIRECTORY)
file(MAKE_DIRECTORY "${DUMP_PARENT}")

execute_process(
  COMMAND "${DUMPER}" --ckpt "${CKPT}" --case "${CASE_DIR}" --out "${DUMP_DIR}"
  RESULT_VARIABLE dump_rc
)
if(NOT dump_rc EQUAL 0)
  message(FATAL_ERROR "octo_parity_dump failed: ${dump_rc}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" "${VERIFY}" --golden "${CASE_DIR}" --dump "${DUMP_DIR}" --report "${DUMP_DIR}/parity_report.json"
  RESULT_VARIABLE verify_rc
)
if(NOT verify_rc EQUAL 0)
  message(FATAL_ERROR "verify_octo_parity.py failed: ${verify_rc}")
endif()
