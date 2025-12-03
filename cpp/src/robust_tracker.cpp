/**
 * @file robust_tracker.cpp
 * @brief Implementation of robust visual object tracker
 */

#include "robust_tracker.h"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace robust_tracker {

RobustTracker::RobustTracker(const std::string& onnx_path,
                               const TrackerConfig& config,
                               const std::vector<std::string>& providers)
    : config_(config)
    , tracking_state_(TrackingState::INIT)
    , frame_id_(0)
    , lost_count_(0)
    , initialized_(false)
    , feat_size_(config.search_size / 16) {  // Assuming stride of 16
    
    try {
        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RobustTracker");
        
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
            // CPUExecutionProvider is default
        }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), *session_options_);
        
        // Get input/output names
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_str_.push_back(std::string(input_name.get()));
        }
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }
        
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_str_.push_back(std::string(output_name.get()));
        }
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }
        
        // Initialize ReID feature bank
        reid_bank_ = std::make_unique<ReIDFeatureBank>(
            config_.reid_bank_size,
            config_.reid_update_interval,
            config_.reid_similarity_threshold);
        
        // Initialize Hanning window
        hann_window_ = hann2d(feat_size_, feat_size_);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        session_.reset();
    } catch (const std::exception& e) {
        std::cerr << "Error initializing tracker: " << e.what() << std::endl;
        session_.reset();
    }
}

RobustTracker::~RobustTracker() = default;

bool RobustTracker::setReIDModel(const std::string& reid_onnx_path) {
    try {
        reid_model_ = std::make_unique<OSNetInfer>(reid_onnx_path);
        return reid_model_->isValid();
    } catch (const std::exception& e) {
        std::cerr << "Error loading ReID model: " << e.what() << std::endl;
        reid_model_.reset();
        return false;
    }
}

bool RobustTracker::initialize(const cv::Mat& image, const BBox& target_box) {
    if (!session_) {
        std::cerr << "Tracker session not initialized" << std::endl;
        return false;
    }
    
    // Store initial state
    state_ = target_box;
    frame_id_ = 0;
    lost_count_ = 0;
    tracking_state_ = TrackingState::TRACKING;
    
    // Extract template
    current_template_ = extractTemplate(image, target_box);
    
    // Reset and initialize ReID bank
    reid_bank_->reset();
    
    // Extract and register initial ReID feature
    if (reid_model_ && reid_model_->isValid()) {
        cv::Mat target_patch = image(target_box.toRect()).clone();
        std::vector<float> feature = extractReIDFeature(target_patch);
        if (!feature.empty()) {
            reid_bank_->registerFeature(feature);
        }
    }
    
    initialized_ = true;
    return true;
}

TrackingResult RobustTracker::track(const cv::Mat& image) {
    if (!initialized_) {
        TrackingResult result;
        result.success = false;
        result.state = TrackingState::INIT;
        return result;
    }
    
    ++frame_id_;
    
    TrackingResult result;
    
    // Choose tracking mode based on state
    switch (tracking_state_) {
        case TrackingState::TRACKING:
        case TrackingState::UNCERTAIN:
            result = normalTrack(image);
            break;
        case TrackingState::LOST:
            result = research(image);
            break;
        default:
            result = normalTrack(image);
            break;
    }
    
    return result;
}

TrackingResult RobustTracker::normalTrack(const cv::Mat& image) {
    TrackingResult result;
    result.state = tracking_state_;
    result.success = true;
    
    // Sample search region
    float resize_factor;
    cv::Mat search_patch = sampleTargetHisi(image, state_, config_.search_factor,
                                             config_.search_size, resize_factor);
    
    // Preprocess patches
    std::vector<float> template_tensor = preprocessCropBgrToNchw(current_template_, 
                                                                  config_.template_size,
                                                                  mean_, std_);
    std::vector<float> search_tensor = preprocessCropBgrToNchw(search_patch,
                                                                config_.search_size,
                                                                mean_, std_);
    
    // Run inference
    std::vector<float> score_map, size_map, offset_map;
    runInference(template_tensor, search_tensor, score_map, size_map, offset_map);
    
    // Apply Hanning window to score map
    for (int i = 0; i < feat_size_; ++i) {
        for (int j = 0; j < feat_size_; ++j) {
            score_map[i * feat_size_ + j] *= hann_window_.at<float>(i, j);
        }
    }
    
    // Calculate bounding box
    float max_score;
    BBox pred_box = calBbox(score_map, size_map, offset_map, feat_size_, 
                            config_.search_size, max_score);
    
    // Map back to original image coordinates
    BBox mapped_box = mapBoxBack(pred_box, state_, resize_factor, config_.search_size);
    
    // Clip to image boundaries
    int H = image.rows;
    int W = image.cols;
    BBox clipped_box = clipBox(mapped_box, H, W);
    
    // ReID verification
    bool reid_match = true;
    if (reid_model_ && reid_model_->isValid() && max_score < config_.conf_high) {
        cv::Rect box_rect = clipped_box.toRect();
        box_rect &= cv::Rect(0, 0, W, H);  // Ensure within bounds
        if (box_rect.width > 0 && box_rect.height > 0) {
            cv::Mat target_patch = image(box_rect).clone();
            std::vector<float> feature = extractReIDFeature(target_patch);
            reid_match = isReIDMatch(feature);
            
            // Update ReID bank periodically during good tracking
            if (reid_match && max_score > config_.conf_medium && 
                frame_id_ % config_.reid_update_interval == 0) {
                reid_bank_->registerFeature(feature);
            }
        }
    }
    
    // Update tracking state
    updateTrackingState(max_score, reid_match);
    
    // Update state
    if (tracking_state_ != TrackingState::LOST) {
        state_ = clipped_box;
        
        // Update template if confidence is high
        if (max_score > config_.conf_high) {
            current_template_ = extractTemplate(image, state_);
        }
    }
    
    result.box = state_;
    result.confidence = max_score;
    result.state = tracking_state_;
    
    return result;
}

TrackingResult RobustTracker::research(const cv::Mat& image) {
    // When lost, try to find target again using ReID or larger search area
    ++lost_count_;
    
    TrackingResult result;
    result.state = TrackingState::LOST;
    result.box = state_;
    result.confidence = 0.0f;
    result.success = false;
    
    // If lost for too long, give up
    if (lost_count_ > config_.max_lost_frames) {
        return result;
    }
    
    // Try tracking with current template (larger search area)
    float resize_factor;
    float extended_factor = config_.search_factor * 1.5f;  // Extend search area
    cv::Mat search_patch = sampleTargetHisi(image, state_, extended_factor,
                                             config_.search_size, resize_factor);
    
    // Preprocess patches
    std::vector<float> template_tensor = preprocessCropBgrToNchw(current_template_,
                                                                  config_.template_size,
                                                                  mean_, std_);
    std::vector<float> search_tensor = preprocessCropBgrToNchw(search_patch,
                                                                config_.search_size,
                                                                mean_, std_);
    
    // Run inference
    std::vector<float> score_map, size_map, offset_map;
    runInference(template_tensor, search_tensor, score_map, size_map, offset_map);
    
    // Apply Hanning window
    for (int i = 0; i < feat_size_; ++i) {
        for (int j = 0; j < feat_size_; ++j) {
            score_map[i * feat_size_ + j] *= hann_window_.at<float>(i, j);
        }
    }
    
    // Calculate bounding box
    float max_score;
    BBox pred_box = calBbox(score_map, size_map, offset_map, feat_size_,
                            config_.search_size, max_score);
    
    // Map back to original image
    BBox mapped_box = mapBoxBack(pred_box, state_, resize_factor, config_.search_size);
    BBox clipped_box = clipBox(mapped_box, image.rows, image.cols);
    
    // ReID verification for recovery
    bool reid_match = false;
    if (reid_model_ && reid_model_->isValid() && max_score > config_.conf_low) {
        cv::Rect box_rect = clipped_box.toRect();
        box_rect &= cv::Rect(0, 0, image.cols, image.rows);
        if (box_rect.width > 0 && box_rect.height > 0) {
            cv::Mat target_patch = image(box_rect).clone();
            std::vector<float> feature = extractReIDFeature(target_patch);
            reid_match = isReIDMatch(feature);
        }
    }
    
    // Check if target is found again
    if (max_score > config_.conf_medium && reid_match) {
        // Target recovered
        state_ = clipped_box;
        tracking_state_ = TrackingState::UNCERTAIN;  // Start with uncertain
        lost_count_ = 0;
        
        result.box = state_;
        result.confidence = max_score;
        result.state = TrackingState::UNCERTAIN;
        result.success = true;
    }
    
    return result;
}

void RobustTracker::updateTrackingState(float confidence, bool reid_match) {
    if (confidence > config_.conf_high) {
        tracking_state_ = TrackingState::TRACKING;
        lost_count_ = 0;
    } else if (confidence > config_.conf_medium) {
        if (reid_match) {
            tracking_state_ = TrackingState::TRACKING;
            lost_count_ = 0;
        } else {
            tracking_state_ = TrackingState::UNCERTAIN;
        }
    } else if (confidence > config_.conf_low) {
        if (reid_match) {
            tracking_state_ = TrackingState::UNCERTAIN;
        } else {
            tracking_state_ = TrackingState::LOST;
            ++lost_count_;
        }
    } else {
        tracking_state_ = TrackingState::LOST;
        ++lost_count_;
    }
}

TrackingResult RobustTracker::trackWithTemplate(const cv::Mat& image, const cv::Mat& template_patch) {
    // Use provided template for tracking
    cv::Mat saved_template = current_template_.clone();
    current_template_ = template_patch.clone();
    
    TrackingResult result = normalTrack(image);
    
    // Restore original template
    current_template_ = saved_template;
    
    return result;
}

cv::Mat RobustTracker::extractTemplate(const cv::Mat& image, const BBox& box) {
    float resize_factor;
    return sampleTargetHisi(image, box, config_.template_factor, 
                            config_.template_size, resize_factor);
}

std::vector<float> RobustTracker::extractReIDFeature(const cv::Mat& image) {
    if (!reid_model_ || !reid_model_->isValid()) {
        return {};
    }
    
    return reid_model_->extractFeature(image);
}

bool RobustTracker::isReIDMatch(const std::vector<float>& feature) {
    if (feature.empty()) {
        return true;  // No feature, assume match
    }
    
    return reid_bank_->isSimilar(feature);
}

void RobustTracker::runInference(const std::vector<float>& template_tensor,
                                   const std::vector<float>& search_tensor,
                                   std::vector<float>& score_map,
                                   std::vector<float>& size_map,
                                   std::vector<float>& offset_map) {
    if (!session_) {
        return;
    }
    
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        // Prepare input tensors
        std::vector<int64_t> template_shape = {1, 3, config_.template_size, config_.template_size};
        std::vector<int64_t> search_shape = {1, 3, config_.search_size, config_.search_size};
        
        // Create copies since CreateTensor needs non-const pointers
        std::vector<float> template_data = template_tensor;
        std::vector<float> search_data = search_tensor;
        
        Ort::Value template_ort = Ort::Value::CreateTensor<float>(
            memory_info, template_data.data(), template_data.size(),
            template_shape.data(), template_shape.size());
        
        Ort::Value search_ort = Ort::Value::CreateTensor<float>(
            memory_info, search_data.data(), search_data.size(),
            search_shape.data(), search_shape.size());
        
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(template_ort));
        inputs.push_back(std::move(search_ort));
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            inputs.data(),
            inputs.size(),
            output_names_.data(),
            output_names_.size());
        
        // Parse outputs (assuming 3 outputs: score_map, size_map, offset_map)
        if (output_tensors.size() >= 3) {
            // Score map
            {
                float* data = output_tensors[0].GetTensorMutableData<float>();
                auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t size = 1;
                for (auto d : shape) size *= d;
                score_map.assign(data, data + size);
            }
            
            // Size map
            {
                float* data = output_tensors[1].GetTensorMutableData<float>();
                auto shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
                size_t size = 1;
                for (auto d : shape) size *= d;
                size_map.assign(data, data + size);
            }
            
            // Offset map
            {
                float* data = output_tensors[2].GetTensorMutableData<float>();
                auto shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
                size_t size = 1;
                for (auto d : shape) size *= d;
                offset_map.assign(data, data + size);
            }
        } else if (output_tensors.size() == 1) {
            // Single output case - parse based on expected dimensions
            float* data = output_tensors[0].GetTensorMutableData<float>();
            auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // Assuming output is [B, 5, H, W] where channels are [score, w, h, ox, oy]
            if (shape.size() >= 4 && shape[1] >= 5) {
                size_t channel_size = shape[2] * shape[3];
                
                score_map.assign(data, data + channel_size);
                size_map.assign(data + channel_size, data + 3 * channel_size);
                offset_map.assign(data + 3 * channel_size, data + 5 * channel_size);
            }
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference error: " << e.what() << std::endl;
    }
}

// ============================================================================
// Processing Functions
// ============================================================================

int processVideoRobust(RobustTracker& tracker,
                       const std::string& video_path,
                       const std::string& save_path,
                       bool display) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video: " << video_path << std::endl;
        return 0;
    }
    
    cv::VideoWriter writer;
    if (!save_path.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(save_path, fourcc, fps, cv::Size(width, height));
    }
    
    cv::Mat frame;
    int frame_count = 0;
    
    // Read first frame
    if (!cap.read(frame)) {
        return 0;
    }
    
    // Select ROI for initialization
    cv::Rect roi = cv::selectROI("Select Target", frame, false);
    cv::destroyWindow("Select Target");
    
    if (roi.width == 0 || roi.height == 0) {
        return 0;
    }
    
    // Initialize tracker
    BBox init_box = BBox::fromRect(roi);
    if (!tracker.initialize(frame, init_box)) {
        std::cerr << "Failed to initialize tracker" << std::endl;
        return 0;
    }
    
    // Draw first frame
    drawTrackingResult(frame, init_box, 
                        trackingStateToString(TrackingState::TRACKING), 
                        1.0f, cv::Scalar(0, 255, 0));
    
    if (display) {
        cv::imshow("Tracking", frame);
    }
    if (writer.isOpened()) {
        writer.write(frame);
    }
    ++frame_count;
    
    // Process remaining frames
    while (cap.read(frame)) {
        TrackingResult result = tracker.track(frame);
        
        // Choose color based on state
        cv::Scalar color;
        switch (result.state) {
            case TrackingState::TRACKING:
                color = cv::Scalar(0, 255, 0);  // Green
                break;
            case TrackingState::UNCERTAIN:
                color = cv::Scalar(0, 255, 255);  // Yellow
                break;
            case TrackingState::LOST:
                color = cv::Scalar(0, 0, 255);  // Red
                break;
            default:
                color = cv::Scalar(128, 128, 128);  // Gray
                break;
        }
        
        drawTrackingResult(frame, result.box,
                            trackingStateToString(result.state),
                            result.confidence, color);
        
        if (display) {
            cv::imshow("Tracking", frame);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {  // ESC or 'q' to quit
                break;
            }
        }
        
        if (writer.isOpened()) {
            writer.write(frame);
        }
        
        ++frame_count;
    }
    
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    if (display) {
        cv::destroyAllWindows();
    }
    
    return frame_count;
}

int batchProcessFolderRobust(RobustTracker& tracker,
                              const std::string& folder_path,
                              const std::string& save_path,
                              bool display) {
    // List image files
    std::vector<std::string> image_files;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_regular_file() && isImageFile(entry.path().string())) {
            image_files.push_back(entry.path().string());
        }
    }
    
    if (image_files.empty()) {
        std::cerr << "No image files found in: " << folder_path << std::endl;
        return 0;
    }
    
    // Sort files
    std::sort(image_files.begin(), image_files.end());
    
    // Create output directory if needed
    if (!save_path.empty()) {
        std::filesystem::create_directories(save_path);
    }
    
    int frame_count = 0;
    
    // Read first image
    cv::Mat frame = cv::imread(image_files[0]);
    if (frame.empty()) {
        std::cerr << "Failed to read first image" << std::endl;
        return 0;
    }
    
    // Select ROI for initialization
    cv::Rect roi = cv::selectROI("Select Target", frame, false);
    cv::destroyWindow("Select Target");
    
    if (roi.width == 0 || roi.height == 0) {
        return 0;
    }
    
    // Initialize tracker
    BBox init_box = BBox::fromRect(roi);
    if (!tracker.initialize(frame, init_box)) {
        std::cerr << "Failed to initialize tracker" << std::endl;
        return 0;
    }
    
    // Process all images
    for (size_t i = 0; i < image_files.size(); ++i) {
        frame = cv::imread(image_files[i]);
        if (frame.empty()) {
            continue;
        }
        
        TrackingResult result;
        if (i == 0) {
            result.box = init_box;
            result.confidence = 1.0f;
            result.state = TrackingState::TRACKING;
        } else {
            result = tracker.track(frame);
        }
        
        // Choose color based on state
        cv::Scalar color;
        switch (result.state) {
            case TrackingState::TRACKING:
                color = cv::Scalar(0, 255, 0);
                break;
            case TrackingState::UNCERTAIN:
                color = cv::Scalar(0, 255, 255);
                break;
            case TrackingState::LOST:
                color = cv::Scalar(0, 0, 255);
                break;
            default:
                color = cv::Scalar(128, 128, 128);
                break;
        }
        
        drawTrackingResult(frame, result.box,
                            trackingStateToString(result.state),
                            result.confidence, color);
        
        if (display) {
            cv::imshow("Tracking", frame);
            int key = cv::waitKey(30);
            if (key == 27 || key == 'q') {
                break;
            }
        }
        
        if (!save_path.empty()) {
            std::string output_file = save_path + "/" + 
                std::filesystem::path(image_files[i]).filename().string();
            cv::imwrite(output_file, frame);
        }
        
        ++frame_count;
    }
    
    if (display) {
        cv::destroyAllWindows();
    }
    
    return frame_count;
}

} // namespace robust_tracker
