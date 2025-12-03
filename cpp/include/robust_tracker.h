/**
 * @file robust_tracker.h
 * @brief Robust visual object tracker with ReID support
 * 
 * Main tracking class that combines ONNX-based tracking with ReID verification
 */

#ifndef ROBUST_TRACKER_H
#define ROBUST_TRACKER_H

#include "tracking_state.h"
#include "reid_feature_bank.h"
#include "osnet_infer.h"
#include "utils.h"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace robust_tracker {

/**
 * @struct TrackerConfig
 * @brief Configuration parameters for the robust tracker
 */
struct TrackerConfig {
    // Template and search sizes
    int template_size = 128;      ///< Template patch size
    int search_size = 256;        ///< Search region size
    float template_factor = 2.0f; ///< Template area factor
    float search_factor = 4.0f;   ///< Search area factor
    
    // Confidence thresholds
    float conf_high = 0.8f;    ///< High confidence threshold
    float conf_medium = 0.3f;  ///< Medium confidence threshold  
    float conf_low = 0.15f;    ///< Low confidence threshold
    
    // Lost tracking parameters
    int max_lost_frames = 3000;  ///< Max frames before giving up
    
    // ReID parameters
    int reid_bank_size = 10;           ///< ReID feature bank size
    int reid_update_interval = 20;     ///< Frames between ReID updates
    float reid_similarity_threshold = 0.7f;  ///< ReID matching threshold
};

/**
 * @struct TrackingResult
 * @brief Result of tracking operation
 */
struct TrackingResult {
    BBox box;                    ///< Bounding box (x, y, w, h)
    float confidence;            ///< Tracking confidence
    TrackingState state;         ///< Current tracking state
    bool success;                ///< Whether tracking succeeded
};

/**
 * @class RobustTracker
 * @brief Robust visual object tracker with ReID-based verification
 * 
 * This tracker combines template-based tracking with ReID features
 * to robustly track objects through occlusions and appearance changes.
 */
class RobustTracker {
public:
    /**
     * @brief Constructor
     * @param onnx_path Path to tracking ONNX model
     * @param config Tracker configuration
     * @param providers ONNX execution providers
     */
    RobustTracker(const std::string& onnx_path,
                  const TrackerConfig& config = TrackerConfig(),
                  const std::vector<std::string>& providers = {"CPUExecutionProvider"});
    
    /**
     * @brief Destructor
     */
    ~RobustTracker();
    
    /**
     * @brief Initialize tracker with first frame and target box
     * @param image First frame (BGR)
     * @param target_box Initial target bounding box [x, y, w, h]
     * @return true if initialization succeeded
     */
    bool initialize(const cv::Mat& image, const BBox& target_box);
    
    /**
     * @brief Track target in new frame
     * @param image Current frame (BGR)
     * @return Tracking result
     */
    TrackingResult track(const cv::Mat& image);
    
    /**
     * @brief Set ReID model for verification
     * @param reid_onnx_path Path to ReID ONNX model
     * @return true if ReID model loaded successfully
     */
    bool setReIDModel(const std::string& reid_onnx_path);
    
    /**
     * @brief Get current tracking state
     * @return Current tracking state
     */
    TrackingState getState() const { return tracking_state_; }
    
    /**
     * @brief Get current bounding box
     * @return Current target bounding box
     */
    BBox getCurrentBox() const { return state_; }
    
    /**
     * @brief Get frame count since initialization
     * @return Frame count
     */
    int getFrameCount() const { return frame_id_; }
    
    /**
     * @brief Check if tracker is initialized
     * @return true if initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Get configuration
     * @return Tracker configuration
     */
    const TrackerConfig& getConfig() const { return config_; }

private:
    /**
     * @brief Normal tracking mode
     * @param image Current frame
     * @return Tracking result
     */
    TrackingResult normalTrack(const cv::Mat& image);
    
    /**
     * @brief Re-search for lost target
     * @param image Current frame
     * @return Tracking result
     */
    TrackingResult research(const cv::Mat& image);
    
    /**
     * @brief Update tracking state based on confidence
     * @param confidence Current tracking confidence
     * @param reid_match Whether ReID verification passed
     */
    void updateTrackingState(float confidence, bool reid_match);
    
    /**
     * @brief Track using template matching
     * @param image Current frame
     * @param template_patch Template patch
     * @return Tracking result
     */
    TrackingResult trackWithTemplate(const cv::Mat& image, const cv::Mat& template_patch);
    
    /**
     * @brief Extract template from image
     * @param image Source image
     * @param box Target bounding box
     * @return Template patch
     */
    cv::Mat extractTemplate(const cv::Mat& image, const BBox& box);
    
    /**
     * @brief Extract ReID feature from image patch
     * @param image Image patch
     * @return Feature vector
     */
    std::vector<float> extractReIDFeature(const cv::Mat& image);
    
    /**
     * @brief Check if ReID feature matches stored features
     * @param feature Feature to check
     * @return true if matches
     */
    bool isReIDMatch(const std::vector<float>& feature);
    
    /**
     * @brief Run ONNX inference
     * @param template_tensor Template tensor
     * @param search_tensor Search tensor
     * @param score_map Output score map
     * @param size_map Output size map
     * @param offset_map Output offset map
     */
    void runInference(const std::vector<float>& template_tensor,
                      const std::vector<float>& search_tensor,
                      std::vector<float>& score_map,
                      std::vector<float>& size_map,
                      std::vector<float>& offset_map);
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    
    // ReID components
    std::unique_ptr<OSNetInfer> reid_model_;
    std::unique_ptr<ReIDFeatureBank> reid_bank_;
    
    // Tracking state
    TrackerConfig config_;
    TrackingState tracking_state_;
    BBox state_;                    ///< Current bounding box
    cv::Mat current_template_;      ///< Current template patch
    cv::Mat hann_window_;           ///< Hanning window for score weighting
    
    int frame_id_;
    int lost_count_;
    bool initialized_;
    
    // Model parameters
    int feat_size_;                 ///< Feature map size
    
    // Normalization parameters
    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
};

// ============================================================================
// Processing Functions
// ============================================================================

/**
 * @brief Process a video file with robust tracker
 * 
 * @param tracker Tracker instance
 * @param video_path Path to video file
 * @param save_path Output video path (empty for no save)
 * @param display Whether to display results
 * @return Number of frames processed
 */
int processVideoRobust(RobustTracker& tracker,
                       const std::string& video_path,
                       const std::string& save_path = "",
                       bool display = true);

/**
 * @brief Process a folder of images with robust tracker
 * 
 * @param tracker Tracker instance
 * @param folder_path Path to image folder
 * @param save_path Output folder path (empty for no save)
 * @param display Whether to display results
 * @return Number of frames processed
 */
int batchProcessFolderRobust(RobustTracker& tracker,
                              const std::string& folder_path,
                              const std::string& save_path = "",
                              bool display = true);

} // namespace robust_tracker

#endif // ROBUST_TRACKER_H
