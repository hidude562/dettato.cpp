include(FetchContent)

# Pinned by commit, like ggml: a tag can move.
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        b3fb4b9feafcd8d91c5cb510a4775143fdbef02f   # v3.11.0
)
FetchContent_MakeAvailable(Catch2)

file(GLOB DETTATO_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
add_executable(dettato_tests ${DETTATO_TEST_SOURCES})
target_link_libraries(dettato_tests PRIVATE dettato_ggml Catch2::Catch2WithMain)
target_include_directories(dettato_tests PRIVATE ../src)

# The [model] tests run when DETTATO_TEST_MODEL names a dettato GGUF (environment variable), and skip otherwise.
add_test(NAME dettato_tests COMMAND dettato_tests --skip-benchmarks)
set_tests_properties(dettato_tests PROPERTIES SKIP_RETURN_CODE 4)
