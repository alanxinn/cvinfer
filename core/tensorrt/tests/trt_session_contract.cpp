#include "core/trt_core.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: trt_session_contract dynamic-identity.engine");
        auto session = cvinfer::core::TrtSession::load(argv[1]);
        require(session.input_names() == std::vector<std::string>{"input"}, "input names");
        require(session.output_names() == std::vector<std::string>{"output"}, "output names");
        require(session.tensor_shape("input") == std::vector<int64_t>({2, 3}), "profile optimal shape");
        for (const int rows : {1, 4, 2}) {
            std::vector<float> values(rows * 3);
            for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i) / 3;
            session.set_shape("input", {rows, 3});
            session.set_input("input", values.data(), values.size() * sizeof(float));
            session.infer();
            const auto output = session.get_output("output");
            require(output.shape == std::vector<int64_t>({rows, 3}), "dynamic output shape");
            require(output.data_type == cvinfer::core::TensorDataType::Float32, "output dtype");
            require(output.data.size() == values.size() * sizeof(float), "output bytes");
            require(std::memcmp(output.data.data(), values.data(), output.data.size()) == 0, "host output values");
            void* device = nullptr;
            cuda_check(cudaMalloc(&device, output.data.size()));
            try {
                cuda_check(cudaMemcpy(device, values.data(), output.data.size(), cudaMemcpyHostToDevice));
                session.set_input_device("input", device, output.data.size());
                session.set_output_host_copy(false);
                session.infer_async();
                session.synchronize();
                require(!session.is_pending(), "synchronous compatibility");
                require(session.outputs().empty(), "disabled host copy");
                require(session.device_output_shape("output") == output.shape, "device shape");
                std::vector<float> actual(values.size());
                cuda_check(cudaMemcpy(actual.data(), session.device_output("output"), output.data.size(), cudaMemcpyDeviceToHost));
                require(actual == values, "external device input/output");
                session.set_output_host_copy(true);
                session.set_input("input", values.data(), output.data.size());
                session.infer();
                require(session.get_output_ref("output").data == output.data, "device to host input switching");
            } catch (...) { cudaFree(device); throw; }
            cuda_check(cudaFree(device));
            session.set_input("input", values.data(), sizeof(float));
            bool rejected = false;
            try { session.infer(); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "input byte mismatch must be rejected");
        }
        bool rejected = false;
        try { session.set_shape("output", {1, 3}); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "output shape mutation must be rejected");
        std::cout << "PASS: dynamic shapes, host/device input, host-copy toggle, buffer reuse and validation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
