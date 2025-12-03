/**
 * @file utils.h
 * @brief Utility functions for robust tracker
 * 
 * Common utility functions for image processing and box operations
 */

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

namespace robust_tracker {

/**
 * @brief Bounding box structure (x, y, width, height)
 */
struct BBox {
    float x, y, w, h;
    
    BBox() : x(0), y(0), w(0), h(0) {}
    BBox(float x_, float y_, float w_, float h_) : x(x_), y(y_), w(w_), h(h_) {}
    
    float cx() const { return x + w * 0.5f; }
    float cy() const { return y + h * 0.5f; }
    float area() const { return w * h; }
    
    cv::Rect toRect() const {
        return cv::Rect(static_cast<int>(x), static_cast<int>(y),
                        static_cast<int>(w), static_cast<int>(h));
    }
    
    static BBox fromRect(const cv::Rect& rect) {
        return BBox(static_cast<float>(rect.x), static_cast<float>(rect.y),
                    static_cast<float>(rect.width), static_cast<float>(rect.height));
    }
};

/**
 * @brief Align value to multiple of 16 (for HiSilicon platform)
 * @param value Value to align
 * @return Aligned value
 */
inline int alignTo16(int value) {
    return ((value + 15) / 16) * 16;
}

/**
 * @brief Sample target region with HiSilicon platform alignment
 * 
 * Extracts a square crop centered at target box, with size aligned to 16
 * 
 * @param image Input image (BGR)
 * @param target_box Target bounding box [x, y, w, h]
 * @param search_area_factor Ratio of crop size to target size
 * @param output_size Output size (will be aligned to 16)
 * @param resize_factor Output resize factor
 * @return Cropped and resized image patch
 */
cv::Mat sampleTargetHisi(const cv::Mat& image,
                          const BBox& target_box,
                          float search_area_factor,
                          int output_size,
                          float& resize_factor);

/**
 * @brief Preprocess image from BGR to NCHW format for ONNX inference
 * 
 * @param image Input image (BGR, HxWxC)
 * @param output_size Target size for resizing
 * @param mean Normalization mean values
 * @param std Normalization std values
 * @return Preprocessed tensor data (NCHW format, float)
 */
std::vector<float> preprocessCropBgrToNchw(const cv::Mat& image,
                                            int output_size,
                                            const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
                                            const std::vector<float>& std = {0.229f, 0.224f, 0.225f});

/**
 * @brief Calculate bounding box from score, size and offset maps
 * 
 * @param score_map Score map from model (1, 1, H, W)
 * @param size_map Size map from model (1, 2, H, W)
 * @param offset_map Offset map from model (1, 2, H, W)
 * @param search_size Search region size
 * @param max_score Output maximum score
 * @return Normalized bounding box (cx, cy, w, h)
 */
BBox calBbox(const std::vector<float>& score_map,
             const std::vector<float>& size_map,
             const std::vector<float>& offset_map,
             int feat_size,
             int search_size,
             float& max_score);

/**
 * @brief Map bounding box back to original image coordinates
 * 
 * @param pred_box Predicted box in crop coordinates (cx, cy, w, h)
 * @param prev_box Previous target box in image coordinates
 * @param resize_factor Resize factor from sampling
 * @param search_size Search region size
 * @return Box in original image coordinates (x, y, w, h)
 */
BBox mapBoxBack(const BBox& pred_box,
                const BBox& prev_box,
                float resize_factor,
                int search_size);

/**
 * @brief Clip bounding box to image boundaries
 * 
 * @param box Input bounding box
 * @param H Image height
 * @param W Image width
 * @param margin Minimum margin
 * @return Clipped bounding box
 */
BBox clipBox(const BBox& box, int H, int W, int margin = 10);

/**
 * @brief Draw tracking result on image
 * 
 * @param image Image to draw on (will be modified)
 * @param box Bounding box to draw
 * @param state_str State string to display
 * @param confidence Confidence score
 * @param color Box color
 */
void drawTrackingResult(cv::Mat& image,
                         const BBox& box,
                         const std::string& state_str,
                         float confidence,
                         const cv::Scalar& color = cv::Scalar(0, 255, 0));

/**
 * @brief Generate 2D Hanning window
 * 
 * @param size Window size (height, width)
 * @return Hanning window matrix
 */
cv::Mat hann2d(int height, int width);

/**
 * @brief Compute cosine similarity between two vectors
 * @param a First vector
 * @param b Second vector
 * @return Cosine similarity (-1.0 to 1.0)
 */
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

/**
 * @brief L2 normalize a feature vector in-place
 * @param feature Feature vector to normalize
 */
void l2Normalize(std::vector<float>& feature);

/**
 * @brief Get file extension
 * @param filename Filename
 * @return Extension (including dot)
 */
std::string getFileExtension(const std::string& filename);

/**
 * @brief Check if file is a video
 * @param filename Filename
 * @return true if video file
 */
bool isVideoFile(const std::string& filename);

/**
 * @brief Check if file is an image
 * @param filename Filename
 * @return true if image file
 */
bool isImageFile(const std::string& filename);

/**
 * @brief List files in a directory
 * @param directory Directory path
 * @param extension Optional extension filter
 * @return List of file paths
 */
std::vector<std::string> listFiles(const std::string& directory,
                                    const std::string& extension = "");

} // namespace robust_tracker

#endif // UTILS_H
