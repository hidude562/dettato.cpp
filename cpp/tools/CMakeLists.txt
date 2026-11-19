# Every tools/<name>.cpp becomes the executable dettato_<name>.
file(GLOB DETTATO_TOOLS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(src ${DETTATO_TOOLS})
    get_filename_component(name ${src} NAME_WE)
    add_executable(dettato_${name} ${src})
    target_link_libraries(dettato_${name} PRIVATE dettato_ggml)
    target_include_directories(dettato_${name} PRIVATE ../src)
endforeach()
