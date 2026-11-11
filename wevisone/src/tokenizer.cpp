#include "tokenizer.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace cvinfer::wevisone {
namespace {
std::string pair_key(const std::string& left, const std::string& right) {
    // Raw BPE pieces can contain NUL; a NUL separator is not unambiguous.
    return std::to_string(left.size()) + ':' + left + right;
}
std::string nfc(const std::string& text) {
    if (text.empty()) return text;
#if defined(_WIN32)
    if (text.size() > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("Tokenizer input is too large");
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (!count) throw std::invalid_argument("Invalid UTF-8 tokenizer input");
    std::vector<wchar_t> wide(count);
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), wide.data(), count))
        throw std::invalid_argument("Invalid UTF-8 tokenizer input");
    const int capacity = NormalizeString(NormalizationC, wide.data(), count, nullptr, 0);
    if (capacity <= 0) throw std::runtime_error("Cannot size NFC normalization");
    std::vector<wchar_t> normalized(capacity);
    int length = NormalizeString(NormalizationC, wide.data(), count, normalized.data(), capacity);
    if (length < 0) { normalized.resize(static_cast<size_t>(-length)); length = NormalizeString(NormalizationC, wide.data(), count, normalized.data(), static_cast<int>(normalized.size())); }
    if (length <= 0) throw std::runtime_error("Cannot normalize UTF-8 text to NFC");
    const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, normalized.data(), length, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) throw std::runtime_error("Cannot encode NFC tokenizer input");
    std::string result(static_cast<size_t>(bytes), '\0');
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, normalized.data(), length, result.data(), bytes, nullptr, nullptr))
        throw std::runtime_error("Cannot encode NFC tokenizer input");
    return result;
#else
    throw std::runtime_error("NFC WeVisOne tokenizer currently requires Windows NormalizeString");
#endif
}
std::string unhex(const std::string& s) {
    if (s.size() % 2) throw std::runtime_error("Invalid WeVisOne tokenizer hex");
    std::string result;
    for (size_t i = 0; i < s.size(); i += 2) {
        size_t used{};
        const auto value = std::stoul(s.substr(i, 2), &used, 16);
        if (used != 2) throw std::runtime_error("Invalid WeVisOne tokenizer hex");
        result.push_back(static_cast<char>(value));
    }
    return result;
}
}
struct Tokenizer::Impl {
    std::unordered_map<std::string, int64_t> vocab, special;
    std::unordered_map<int64_t, std::string> inverse;
    std::unordered_map<std::string, int> ranks;
    pcre2_code* pattern{};
    bool normalize_nfc{false};
    ~Impl() { if (pattern) pcre2_code_free(pattern); }
    void bpe(const std::string& text, std::vector<int64_t>& ids) const {
        std::vector<std::string> pieces;
        for (unsigned char byte : text) pieces.emplace_back(1, static_cast<char>(byte));
        while (pieces.size() > 1) {
            size_t at = pieces.size();
            int best = std::numeric_limits<int>::max();
            for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                const auto found = ranks.find(pair_key(pieces[i], pieces[i + 1]));
                if (found != ranks.end() && found->second < best) { best = found->second; at = i; }
            }
            if (at == pieces.size()) break;
            const auto left = pieces[at], right = pieces[at + 1];
            for (size_t i = 0; i + 1 < pieces.size();) {
                if (pieces[i] == left && pieces[i + 1] == right) {
                    pieces[i] += pieces[i + 1]; pieces.erase(pieces.begin() + i + 1);
                } else ++i;
            }
        }
        for (const auto& piece : pieces) {
            const auto it = vocab.find(piece);
            if (it == vocab.end()) throw std::runtime_error("WeVisOne byte BPE vocabulary mismatch");
            ids.push_back(it->second);
        }
    }
    void ordinary(const std::string& input, std::vector<int64_t>& ids) const {
        const auto text = normalize_nfc ? nfc(input) : input;
        auto* raw = pcre2_match_data_create_from_pattern(pattern, nullptr);
        if (!raw) throw std::bad_alloc();
        std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(raw, pcre2_match_data_free);
        size_t offset = 0;
        while (offset < text.size()) {
            const int rc = pcre2_match(pattern, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset, 0, raw, nullptr);
            if (rc < 0) throw std::invalid_argument("Invalid UTF-8 or unmatched WeVisOne tokenizer input");
            const auto* range = pcre2_get_ovector_pointer(raw);
            if (range[0] != offset || range[1] <= offset) throw std::runtime_error("WeVisOne tokenizer split is incomplete");
            bpe(text.substr(offset, range[1] - offset), ids); offset = range[1];
        }
    }
};
Tokenizer::Tokenizer(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Missing WeVisOne native tokenizer");
    const auto j = nlohmann::json::parse(stream);
    if (j.at("format") != "cvinfer-qwen-byte-bpe-v1") throw std::runtime_error("Unsupported WeVisOne tokenizer format");
    if (j.contains("normalizer") && !j.at("normalizer").is_null()) {
        if (j.at("normalizer") != nlohmann::json{{"type", "NFC"}}) throw std::runtime_error("Unsupported WeVisOne tokenizer normalization");
        impl_->normalize_nfc = true;
    }
    for (const auto& item : j.at("vocab").items()) {
        const auto raw = unhex(item.key()); const auto id = item.value().get<int64_t>();
        impl_->vocab.emplace(raw, id); impl_->inverse.emplace(id, raw);
    }
    int rank = 0;
    for (const auto& pair : j.at("merges"))
        impl_->ranks.emplace(pair_key(unhex(pair.at(0).get<std::string>()), unhex(pair.at(1).get<std::string>())), rank++);
    for (const auto& item : j.at("special").items()) {
        const auto id = item.value().get<int64_t>();
        impl_->special.emplace(item.key(), id); impl_->inverse[id] = item.key();
    }
    if (impl_->vocab.size() < 256) throw std::runtime_error("Incomplete WeVisOne byte vocabulary");
    int error{}; PCRE2_SIZE at{};
    const auto regex = j.at("pattern").get<std::string>();
    impl_->pattern = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex.data()), regex.size(), PCRE2_UTF | PCRE2_UCP, &error, &at, nullptr);
    if (!impl_->pattern) throw std::runtime_error("Unsupported WeVisOne tokenizer regex at " + std::to_string(at));
}
Tokenizer::~Tokenizer() = default;
std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int64_t> ids;
    size_t position = 0;
    while (position < text.size()) {
        size_t earliest = text.size(); const std::string* selected = nullptr; int64_t id{};
        for (const auto& token : impl_->special) {
            const auto at = text.find(token.first, position);
            if (at < earliest || (at == earliest && selected && token.first.size() > selected->size())) {
                earliest = at; selected = &token.first; id = token.second;
            }
        }
        impl_->ordinary(text.substr(position, earliest - position), ids);
        if (!selected) break;
        ids.push_back(id); position = earliest + selected->size();
    }
    return ids;
}
std::string Tokenizer::decode(const std::vector<int64_t>& tokens) const {
    std::string text;
    for (const auto token : tokens) {
        const auto found = impl_->inverse.find(token);
        if (found == impl_->inverse.end()) throw std::runtime_error("Invalid WeVisOne output token");
        text += found->second;
    }
    return text;
}
int64_t Tokenizer::token_id(const std::string& text) const {
    const auto ids = encode(text);
    if (ids.size() != 1) throw std::runtime_error("WeVisOne tokenizer must contain single token " + text);
    return ids.front();
}
}
