#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
namespace cvinfer::wevisone {
class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& path);
    ~Tokenizer();
    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& tokens) const;
    int64_t token_id(const std::string& special) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
