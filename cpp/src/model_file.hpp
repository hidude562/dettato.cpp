#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml-backend.h>
#include <ggml.h>

struct gguf_context;

namespace dtt
{

/**
 * A dettato GGUF: every tensor uploaded once to the backend's memory, the metadata readable by key. Tensors the CPU
 * front ends need (filterbanks, windows) are also readable back as host copies.
 */
class ModelFile
{
public:
    ModelFile(const std::string& path, ggml_backend_t backend);
    ~ModelFile();
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    /** @return The tensor; throws if the file has no such tensor. */
    ggml_tensor* tensor(const std::string& name) const;
    ggml_tensor* findTensor(const std::string& name) const;

    /** @return A tensor's data as float32 on the host (any stored float type). */
    std::vector<float> hostF32(const std::string& name) const;

    bool has(const std::string& key) const;
    uint32_t u32(const std::string& key) const;
    float f32(const std::string& key) const;
    std::string str(const std::string& key) const;
    std::vector<int32_t> i32s(const std::string& key) const;

    size_t bytes() const { return mBytes; }

private:
    int64_t keyId(const std::string& key) const;

    gguf_context* mGguf = nullptr;
    ggml_context* mCtx = nullptr;
    ggml_backend_buffer_t mBuffer = nullptr;
    std::unordered_map<std::string, ggml_tensor*> mTensors;
    size_t mBytes = 0;
};

} // namespace dtt
