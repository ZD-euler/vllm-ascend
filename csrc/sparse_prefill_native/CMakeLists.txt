# Counted union and original-page copies, built only for the supported A3 target.
# Keep these separate from the ordinary operator library: they are loaded
# lazily only when selected-KV prefill is explicitly enabled.
set(SELECTED_KV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
ascendc_library(sparse_prefill_union_pipeline SHARED "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_union_pipeline.cpp")
ascendc_library(sparse_prefill_fused_fixed SHARED "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_fused_fixed.cpp")
ascendc_library(dsa_packed_inplace SHARED "${SELECTED_KV_SOURCE_DIR}/packed_inplace.cpp")
ascendc_library(dsa_selected_li_pages SHARED "${SELECTED_KV_SOURCE_DIR}/li_page_gather.cpp")

# Address/status checks are required for correctness, including Release builds.
# The generated host wrappers also include these sources; keep the contract
# local to these three targets on both compilation sides.
foreach(selected_kv_target IN ITEMS sparse_prefill_union_pipeline sparse_prefill_fused_fixed dsa_packed_inplace)
    ascendc_compile_definitions(${selected_kv_target} PRIVATE ASCENDC_DUMP=1)
    ascendc_compile_options(${selected_kv_target} PRIVATE
        -UNDEBUG
        -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1)
endforeach()

# Keep the tested main-kernel build settings unchanged; LI is a separate target.
ascendc_compile_definitions(dsa_selected_li_pages PRIVATE ASCENDC_DUMP=1)
ascendc_compile_options(dsa_selected_li_pages PRIVATE
    -UNDEBUG
    -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1)

set(SELECTED_KV_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
if(NOT DEFINED VLLM_ASCEND_PACKAGE_PATH)
    set(VLLM_ASCEND_PACKAGE_PATH "${VLLM_ASCEND_INSTALL_PATH}/vllm_ascend")
endif()
add_custom_command(TARGET sparse_prefill_fused_fixed POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SELECTED_KV_LIBRARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/sparse_prefill_fused_fixed_merge_obj_dir/device_aiv.o"
        "${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o")

add_library(sparse_prefill_fused_fixed_launch SHARED
    "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_fused_fixed_launch.cpp"
    "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_batch_queued.cpp"
    "${SELECTED_KV_SOURCE_DIR}/li_page_gather_queued.cpp"
    "${SELECTED_KV_SOURCE_DIR}/hybm_address_export.cpp")
target_include_directories(sparse_prefill_fused_fixed_launch PRIVATE
    "${ASCEND_HOME_PATH}/include" ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_PATH}/include")
target_link_directories(sparse_prefill_fused_fixed_launch PRIVATE
    "${ASCEND_HOME_PATH}/lib64" "${TORCH_NPU_PATH}/lib")
target_link_libraries(sparse_prefill_fused_fixed_launch PRIVATE
    ascendcl sparse_prefill_union_pipeline dsa_selected_li_pages ${TORCH_LIBRARIES} torch_npu)
add_dependencies(sparse_prefill_fused_fixed_launch sparse_prefill_fused_fixed)
separate_arguments(torch_abi_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(sparse_prefill_fused_fixed_launch PRIVATE ${torch_abi_flags})

set(SELECTED_KV_TARGETS sparse_prefill_union_pipeline sparse_prefill_fused_fixed
    sparse_prefill_fused_fixed_launch dsa_packed_inplace dsa_selected_li_pages)
set_target_properties(${SELECTED_KV_TARGETS} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SELECTED_KV_LIBRARY_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE INSTALL_RPATH "$ORIGIN")
add_dependencies(vllm_ascend_C ${SELECTED_KV_TARGETS})
# The fixed kernel target produces the device object loaded through ACL.
# Its generated host .so is a build artifact, not a runtime dependency.
set(SELECTED_KV_RUNTIME_TARGETS ${SELECTED_KV_TARGETS})
list(REMOVE_ITEM SELECTED_KV_RUNTIME_TARGETS sparse_prefill_fused_fixed)
install(TARGETS ${SELECTED_KV_RUNTIME_TARGETS} LIBRARY DESTINATION "${VLLM_ASCEND_PACKAGE_PATH}/lib")
install(FILES "${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o"
    DESTINATION "${VLLM_ASCEND_PACKAGE_PATH}/lib")
