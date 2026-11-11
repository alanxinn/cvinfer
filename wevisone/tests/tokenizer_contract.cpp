#include "tokenizer.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: wevisone_tokenizer_contract <native-tokenizer.json> <official-cases.json>");
        cvinfer::wevisone::Tokenizer tokenizer(std::filesystem::u8path(argv[1]));
        std::ifstream input(std::filesystem::u8path(argv[2]));
        const auto fixture = nlohmann::json::parse(input);
        size_t passed = 0;
        for (const auto& item : fixture.at("items")) {
            const auto text = item.at("text").get<std::string>();
            const auto ids = tokenizer.encode(text);
            const auto expected = item.at("ids").get<std::vector<int64_t>>();
            if (ids != expected) throw std::runtime_error("Official token IDs differ for " + item.at("text").dump()
                + " expected=" + nlohmann::json(expected).dump() + " actual=" + nlohmann::json(ids).dump());
            if (tokenizer.decode(ids) != item.at("decoded").get<std::string>())
                throw std::runtime_error("Official decoded text differs for " + item.at("text").dump());
            ++passed;
        }
        std::cout << "PASS official tokenizer conformance: " << passed << " cases\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
