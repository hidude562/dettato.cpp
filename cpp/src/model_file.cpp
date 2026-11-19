#include "model_file.hpp"

#include <cstdio>
#include <memory>

#include <ggml.h>
#include <gguf.h>

#include "common.hpp"

namespace dtt
{

ModelFile::ModelFile(const std::string& path, ggml_backend_t backend)
{
    gguf_init_params params = {/*no_alloc*/ true, &mCtx};
    mGguf = gguf_init_from_file(path.c_str(), params);
    DTT_CHECK(mGguf != nullptr, "cannot read %s as GGUF", path.c_str());

    const int64_t ver = gguf_find_key(mGguf, "dettato.format_version");
    DTT_CHECK(ver >= 0, "%s is not a dettato checkpoint", path.c_str());
    DTT_CHECK(gguf_get_val_u32(mGguf, ver) == 1, "unsupported dettato checkpoint version %u", gguf_get_val_u32(mGguf, ver));

    mBuffer = ggml_backend_alloc_ctx_tensors(mCtx, backend);
    DTT_CHECK(mBuffer != nullptr, "cannot allocate the weights on the device");
    ggml_backend_buffer_set_usage(mBuffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::unique_ptr<FILE, int (*)(FILE*)> f(ggml_fopen(path.c_str(), "rb"), &std::fclose); // UTF-8 name, like gguf_init_from_file
    DTT_CHECK(f != nullptr, "cannot open %s", path.c_str());
    const size_t data0 = gguf_get_data_offset(mGguf);
    std::vector<uint8_t> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(mGguf); i++) {
        const char* name = gguf_get_tensor_name(mGguf, i);
        ggml_tensor* t = ggml_get_tensor(mCtx, name);
        DTT_CHECK(t != nullptr, "tensor %s missing from context", name);
        const size_t n = ggml_nbytes(t);
        tmp.resize(n);
#if defined(_WIN32)
        DTT_CHECK(_fseeki64(f.get(), static_cast<long long>(data0 + gguf_get_tensor_offset(mGguf, i)), SEEK_SET) == 0, "seek failed");
#else
        DTT_CHECK(fseeko(f.get(), static_cast<off_t>(data0 + gguf_get_tensor_offset(mGguf, i)), SEEK_SET) == 0, "seek failed");
#endif
        DTT_CHECK(std::fread(tmp.data(), 1, n, f.get()) == n, "short read for %s", name);
        ggml_backend_tensor_set(t, tmp.data(), 0, n);
        mTensors.emplace(name, t);
        mBytes += n;
    }
}

ModelFile::~ModelFile()
{
    if (mBuffer != nullptr) {
        ggml_backend_buffer_free(mBuffer);
    }
    if (mCtx != nullptr) {
        ggml_free(mCtx);
    }
    if (mGguf != nullptr) {
        gguf_free(mGguf);
    }
}

ggml_tensor* ModelFile::findTensor(const std::string& name) const
{
    auto it = mTensors.find(name);
    return it == mTensors.end() ? nullptr : it->second;
}

ggml_tensor* ModelFile::tensor(const std::string& name) const
{
    ggml_tensor* t = findTensor(name);
    DTT_CHECK(t != nullptr, "checkpoint has no tensor %s", name.c_str());
    return t;
}

std::vector<float> ModelFile::hostF32(const std::string& name) const
{
    ggml_tensor* t = tensor(name);
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    } else {
        DTT_CHECK(t->type == GGML_TYPE_F16, "%s: unsupported host type %s", name.c_str(), ggml_type_name(t->type));
        std::vector<ggml_fp16_t> h(static_cast<size_t>(n));
        ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(h.data(), out.data(), n);
    }
    return out;
}

int64_t ModelFile::keyId(const std::string& key) const
{
    const int64_t id = gguf_find_key(mGguf, key.c_str());
    DTT_CHECK(id >= 0, "checkpoint has no key %s", key.c_str());
    return id;
}

bool ModelFile::has(const std::string& key) const
{
    return gguf_find_key(mGguf, key.c_str()) >= 0;
}

uint32_t ModelFile::u32(const std::string& key) const
{
    return gguf_get_val_u32(mGguf, keyId(key));
}

float ModelFile::f32(const std::string& key) const
{
    return gguf_get_val_f32(mGguf, keyId(key));
}

std::string ModelFile::str(const std::string& key) const
{
    return gguf_get_val_str(mGguf, keyId(key));
}

std::vector<int32_t> ModelFile::i32s(const std::string& key) const
{
    const int64_t id = keyId(key);
    DTT_CHECK(gguf_get_arr_type(mGguf, id) == GGUF_TYPE_INT32, "%s is not an int32 array", key.c_str());
    const auto* p = static_cast<const int32_t*>(gguf_get_arr_data(mGguf, id));
    return std::vector<int32_t>(p, p + gguf_get_arr_n(mGguf, id));
}

} // namespace dtt
