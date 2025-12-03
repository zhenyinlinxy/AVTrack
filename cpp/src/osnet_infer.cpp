/**
 * @file osnet_infer.cpp
 * @brief Implementation of OSNet ReID inference
 */

#include "osnet_infer.h"
#include <stdexcept>
#include <iostream>

namespace robust_tracker {

OSNetInfer::OSNetInfer(const std::string& model_path,
                         const std::vector<std::string>& providers) {
    try {
        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OSNetInfer");
        
        // Configure session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Add execution providers
        for (const auto& provider : providers) {
            if (provider == "CUDAExecutionProvider") {
#ifdef USE_CUDA
                OrtCUDAProviderOptions cuda_options;
                session_options_->AppendExecutionProvider_CUDA(cuda_options);
#endif
            } else if (provider == "TensorRTExecutionProvider") {
#ifdef USE_TENSORRT
                OrtTensorRTProviderOptions trt_options;
                session_options_->AppendExecutionProvider_TensorRT(trt_options);
#endif
            }
            // CPUExecutionProvider is added by default
        }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Get input info
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_str_.push_back(std::string(input_name.get()));
        }
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }
        
        // Get output info
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_str_.push_back(std::string(output_name.get()));
        }
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }
        
        // Get input dimensions (assume first input, NCHW format)
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_shape = tensor_info.GetShape();
        
        if (input_shape.size() >= 4) {
            // Handle dynamic dimensions
            input_channels_ = (input_shape[1] > 0) ? static_cast<int>(input_shape[1]) : 3;
            input_height_ = (input_shape[2] > 0) ? static_cast<int>(input_shape[2]) : 256;
            input_width_ = (input_shape[3] > 0) ? static_cast<int>(input_shape[3]) : 128;
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        session_.reset();
    } catch (const std::exception& e) {
        std::cerr << "Error loading OSNet model: " << e.what() << std::endl;
        session_.reset();
    }
}

OSNetInfer::~OSNetInfer() = default;

std::vector<float> OSNetInfer::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));
    
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    // Prepare output tensor (NCHW format)
    std::vector<float> tensor(input_channels_ * input_height_ * input_width_);
    
    // Split channels and normalize
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    size_t channel_size = input_height_ * input_width_;
    for (int c = 0; c < input_channels_; ++c) {
        cv::Mat normalized = (channels[c] - mean_[c]) / std_[c];
        std::memcpy(tensor.data() + c * channel_size, 
                    normalized.data, 
                    channel_size * sizeof(float));
    }
    
    return tensor;
}

std::vector<float> OSNetInfer::extractFeature(const cv::Mat& image) {
    if (!isValid()) {
        return {};
    }
    
    try {
        // Preprocess image
        std::vector<float> input_tensor = preprocess(image);
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, input_channels_, input_height_, input_width_};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor.data(),
            input_tensor.size(),
            input_shape.data(),
            input_shape.size());
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_ort,
            1,
            output_names_.data(),
            output_names_.size());
        
        // Get output
        if (output_tensors.empty()) {
            return {};
        }
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        std::vector<float> feature(output_data, output_data + output_size);
        
        return feature;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::vector<float>> OSNetInfer::extractFeatures(const std::vector<cv::Mat>& images) {
    std::vector<std::vector<float>> features;
    features.reserve(images.size());
    
    for (const auto& image : images) {
        features.push_back(extractFeature(image));
    }
    
    return features;
}

float OSNetInfer::cosineSimilarity(const std::vector<float>& feat1,
                                     const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size() || feat1.empty()) {
        return 0.0f;
    }
    
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < feat1.size(); ++i) {
        dot += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < 1e-8f || norm2 < 1e-8f) {
        return 0.0f;
    }
    
    return dot / (norm1 * norm2);
}

} // namespace robust_tracker
