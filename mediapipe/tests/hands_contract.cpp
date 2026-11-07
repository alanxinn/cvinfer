#include "mediapipe/hands.h"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;
using cvinfer::mediapipe::HandDetector;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> std::vector<T> read(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    check(bool(input), "Fixture is missing");
    const auto bytes = static_cast<std::size_t>(input.tellg());
    check(bytes % sizeof(T) == 0, "Invalid fixture size");
    std::vector<T> result(bytes / sizeof(T));
    input.seekg(0);
    if (bytes) check(bool(input.read(reinterpret_cast<char*>(result.data()), bytes)), "Fixture read failed");
    return result;
}
void compare(const std::vector<hands_hand_t>& actual, const std::vector<hands_hand_t>& expected) {
    check(actual.size() == expected.size(), "Hand count differs across interfaces");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        std::array<float, 133> aa{}, ee{};
        std::memcpy(aa.data(), &actual[i], sizeof(hands_hand_t));
        std::memcpy(ee.data(), &expected[i], sizeof(hands_hand_t));
        for (std::size_t j = 0; j < aa.size(); ++j)
            check(std::isfinite(aa[j]) && std::abs(aa[j] - ee[j]) <= 1e-6F,
                  "C++/C ABI output differs from Python native fixture");
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: mediapipe_hands_contract engine-directory fp32|fp16");
        const auto root = fs::u8path(argv[1]);
        const std::string precision = argv[2];
        check(precision == "fp32" || precision == "fp16", "Invalid precision");
        const auto palm = (root / ("palm-" + precision + ".engine")).u8string();
        const auto hand = (root / ("hand-" + precision + ".engine")).u8string();
        HandDetector initial(palm, hand);
        HandDetector detector(std::move(initial));
        bool rejected = false;
        try { initial.infer_bgr(nullptr, 1, 1); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "Moved-from C++ detector accepted inference");
        hands_model_t* raw = nullptr;
        check(hands_model_create(palm.c_str(), hand.c_str(), nullptr, &raw) == 0, hands_last_error());
        const auto destroy = [](hands_model_t* model) { hands_model_destroy(model); };
        std::unique_ptr<hands_model_t, decltype(destroy)> owner(raw, destroy);
        const auto fixtures = root / "interface-fixtures";
        std::ifstream manifest(fixtures / ("manifest-" + precision + ".tsv"));
        check(bool(manifest), "Interface fixture manifest is missing");
        std::string name;
        int width, height, count, cases = 0;
        std::vector<hands_hand_t> saved;
        std::vector<uint8_t> first_pixels;
        int first_width = 0, first_height = 0;
        while (manifest >> name >> width >> height >> count) {
            auto pixels = read<uint8_t>(fixtures / (name + ".bgr"));
            auto expected = read<hands_hand_t>(fixtures / (name + ".hands.bin"));
            check(pixels.size() == std::size_t(width) * height * 3 && expected.size() == std::size_t(count), "Invalid fixture dimensions");
            auto result = detector.infer_bgr(pixels.data(), width, height);
            compare(result, expected);
            std::vector<hands_hand_t> output(2);
            std::size_t copied = 999;
            check(hands_model_infer_bgr(raw, pixels.data(), width, height, 3, output.data(), output.size(), &copied) == 0, hands_last_error());
            output.resize(copied);
            compare(output, expected);
            if (!cases) { saved = result; first_pixels = pixels; first_width = width; first_height = height; }
            ++cases;
        }
        check(cases == 5 && saved.size() == 2, "Incomplete real-image fixtures");
        std::size_t count_out = 999;
        check(hands_model_infer_bgr(raw, first_pixels.data(), first_width, first_height, 3, nullptr, 0, &count_out) == 4 && count_out == 2,
              "C ABI capacity query failed");
        hands_hand_t sentinel;
        std::memset(&sentinel, 0x5a, sizeof(sentinel));
        const auto original_sentinel = sentinel;
        check(hands_model_infer_bgr(raw, first_pixels.data(), first_width, first_height, 3, &sentinel, 1, &count_out) == 4 && count_out == 2,
              "C ABI short buffer was accepted");
        check(std::memcmp(&sentinel, &original_sentinel, sizeof(sentinel)) == 0, "C ABI partially wrote a short buffer");
        check(hands_model_infer_bgr(raw, nullptr, 1, 1, 3, nullptr, 0, &count_out) == 1 && count_out == 0, "C ABI null image was accepted");
        check(hands_model_infer_bgr(raw, first_pixels.data(), first_width, first_height, 4, nullptr, 0, &count_out) == 1, "C ABI invalid channels were accepted");
        check(hands_model_infer_bgr(raw, first_pixels.data(), first_width, first_height, 3, nullptr, 0, nullptr) == 1, "C ABI null count was accepted");
        check(hands_model_infer_bgr(nullptr, first_pixels.data(), first_width, first_height, 3, nullptr, 0, &count_out) == 1, "C ABI null model was accepted");
        std::vector<hands_hand_t> recovered(2);
        check(hands_model_infer_bgr(raw, first_pixels.data(), first_width, first_height, 3, recovered.data(), 2, &count_out) == 0, hands_last_error());
        check(std::strlen(hands_last_error()) == 0, "C ABI retained a stale error");
        compare(recovered, saved);
        hands_config_t config;
        hands_config_init(&config);
        check(config.max_hands == 2 && config.palm_threshold == .5F, "C ABI defaults changed");
        config.max_hands = 0;
        hands_model_t* invalid = nullptr;
        check(hands_model_create(palm.c_str(), hand.c_str(), &config, &invalid) == 1 && invalid == nullptr, "Invalid config was accepted");
        config.max_hands = 1; config.palm_threshold = std::numeric_limits<float>::quiet_NaN();
        check(hands_model_create(palm.c_str(), hand.c_str(), &config, &invalid) == 1 && invalid == nullptr, "NaN threshold was accepted");
        check(hands_model_create("", hand.c_str(), nullptr, &invalid) == 1 && invalid == nullptr, "Empty engine path was accepted");
        check(hands_model_create(hand.c_str(), palm.c_str(), nullptr, &invalid) == 2 && invalid == nullptr, "Swapped engine contracts were accepted");
        hands_config_init(&config); config.max_hands = 1;
        HandDetector limited(palm, hand, config);
        check(limited.infer_bgr(first_pixels.data(), first_width, first_height).size() == 1, "C++ max_hands failed");
        owner.reset();
        compare(saved, read<hands_hand_t>(fixtures / (precision + "-original.hands.bin")));
        hands_model_destroy(nullptr);
        std::cout << "PASS: " << precision << " C++ and C ABI; 5 image cases, 133 floats/hand, move/lifetime, max_hands, errors/recovery, no partial writes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
