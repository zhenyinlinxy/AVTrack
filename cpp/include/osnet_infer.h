/**
 * @file osnet_infer.h
 * @brief OSNet ReID model inference wrapper
 * 
 * Provides ONNX Runtime based inference for OSNet ReID model
 */

#ifndef OSNET_INFER_H
#define OSNET_INFER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace robust_tracker {

/**
 * @class OSNetInfer
 * @brief OSNet model inference class for person/object re-identification
 * 
 * This class wraps the ONNX Runtime for running OSNet model inference
 * to extract ReID features from image patches.
 */
class OSNetInfer {
public:
    /**
     * @brief Constructor
     * @param model_path Path to the ONNX model file
     * @param providers List of execution providers (e.g., "CPUExecutionProvider")
     */
    OSNetInfer(const std::string& model_path,
               const std::vector<std::string>& providers = {"CPUExecutionProvider"});
    
    /**
     * @brief Destructor
     */
    ~OSNetInfer();
    
    /**
     * @brief Extract ReID feature from an image patch
     * @param image Image patch (BGR format)
     * @return Feature vector (typically 512-dimensional)
     */
    std::vector<float> extractFeature(const cv::Mat& image);
    
    /**
     * @brief Extract ReID features from multiple image patches
     * @param images Vector of image patches (BGR format)
     * @return Vector of feature vectors
     */
    std::vector<std::vector<float>> extractFeatures(const std::vector<cv::Mat>& images);
    
    /**
     * @brief Compute cosine similarity between two feature vectors
     * @param feat1 First feature vector
     * @param feat2 Second feature vector
     * @return Cosine similarity (-1.0 to 1.0)
     */
    static float cosineSimilarity(const std::vector<float>& feat1,
                                   const std::vector<float>& feat2);
    
    /**
     * @brief Check if model is loaded successfully
     * @return true if model is ready for inference
     */
    bool isValid() const { return session_ != nullptr; }
    
    /**
     * @brief Get the input size expected by the model
     * @return Input size as pair (height, width)
     */
    std::pair<int, int> getInputSize() const { return {input_height_, input_width_}; }

private:
    /**
     * @brief Preprocess image for model input
     * @param image Input image (BGR)
     * @return Preprocessed tensor data
     */
    std::vector<float> preprocess(const cv::Mat& image);
    
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    
    int input_height_ = 256;
    int input_width_ = 128;
    int input_channels_ = 3;
    
    // Normalization parameters (ImageNet)
    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
};

} // namespace robust_tracker

#endif // OSNET_INFER_H
