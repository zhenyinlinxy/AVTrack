/**
 * @file utils.cpp
 * @brief Implementation of utility functions
 */

#include "utils.h"
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace robust_tracker {

cv::Mat sampleTargetHisi(const cv::Mat& image,
                          const BBox& target_box,
                          float search_area_factor,
                          int output_size,
                          float& resize_factor) {
    float x = target_box.x;
    float y = target_box.y;
    float w = target_box.w;
    float h = target_box.h;
    
    // Calculate crop size based on target size and factor
    float crop_sz = std::ceil(std::sqrt(w * h) * search_area_factor);
    
    if (crop_sz < 1) {
        crop_sz = 1;
    }
    
    // Align to 16 for HiSilicon platform
    int crop_sz_aligned = alignTo16(static_cast<int>(crop_sz));
    
    // Calculate crop region
    float cx = x + w * 0.5f;
    float cy = y + h * 0.5f;
    
    int x1 = static_cast<int>(std::round(cx - crop_sz_aligned * 0.5f));
    int x2 = x1 + crop_sz_aligned;
    int y1 = static_cast<int>(std::round(cy - crop_sz_aligned * 0.5f));
    int y2 = y1 + crop_sz_aligned;
    
    // Calculate padding
    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(0, x2 - image.cols);
    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(0, y2 - image.rows);
    
    // Adjust coordinates for valid region
    int x1_valid = x1 + x1_pad;
    int x2_valid = x2 - x2_pad;
    int y1_valid = y1 + y1_pad;
    int y2_valid = y2 - y2_pad;
    
    // Extract valid region
    cv::Rect roi(x1_valid, y1_valid, x2_valid - x1_valid, y2_valid - y1_valid);
    cv::Mat im_crop = image(roi).clone();
    
    // Add padding with constant border (black)
    cv::Mat im_crop_padded;
    cv::copyMakeBorder(im_crop, im_crop_padded, 
                        y1_pad, y2_pad, x1_pad, x2_pad,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    // Resize to output size (also aligned to 16)
    int output_size_aligned = alignTo16(output_size);
    resize_factor = static_cast<float>(output_size_aligned) / crop_sz_aligned;
    
    cv::Mat resized;
    cv::resize(im_crop_padded, resized, cv::Size(output_size_aligned, output_size_aligned));
    
    return resized;
}

std::vector<float> preprocessCropBgrToNchw(const cv::Mat& image,
                                            int output_size,
                                            const std::vector<float>& mean,
                                            const std::vector<float>& std) {
    cv::Mat resized;
    if (image.cols != output_size || image.rows != output_size) {
        cv::resize(image, resized, cv::Size(output_size, output_size));
    } else {
        resized = image;
    }
    
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to float [0, 1]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    // Prepare NCHW tensor
    int channels = 3;
    int height = output_size;
    int width = output_size;
    std::vector<float> tensor(channels * height * width);
    
    // Split channels
    std::vector<cv::Mat> channels_mat(3);
    cv::split(float_img, channels_mat);
    
    // Normalize and copy to tensor
    size_t channel_size = height * width;
    for (int c = 0; c < channels; ++c) {
        cv::Mat normalized = (channels_mat[c] - mean[c]) / std[c];
        
        // Copy to tensor in NCHW format
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                tensor[c * channel_size + h * width + w] = normalized.at<float>(h, w);
            }
        }
    }
    
    return tensor;
}

BBox calBbox(const std::vector<float>& score_map,
             const std::vector<float>& size_map,
             const std::vector<float>& offset_map,
             int feat_size,
             int search_size,
             float& max_score) {
    // Find max score position
    max_score = -1.0f;
    int max_idx = 0;
    
    for (size_t i = 0; i < score_map.size(); ++i) {
        if (score_map[i] > max_score) {
            max_score = score_map[i];
            max_idx = static_cast<int>(i);
        }
    }
    
    // Calculate position
    int idx_y = max_idx / feat_size;
    int idx_x = max_idx % feat_size;
    
    // Get size from size_map (2 channels: w, h)
    int size_offset = feat_size * feat_size;
    float pred_w = size_map[max_idx];                    // Channel 0: width
    float pred_h = size_map[size_offset + max_idx];      // Channel 1: height
    
    // Get offset from offset_map (2 channels: dx, dy)
    float offset_x = offset_map[max_idx];                 // Channel 0: dx
    float offset_y = offset_map[size_offset + max_idx];   // Channel 1: dy
    
    // Calculate center position (normalized to feature map)
    float cx = (static_cast<float>(idx_x) + offset_x + 0.5f) / feat_size;
    float cy = (static_cast<float>(idx_y) + offset_y + 0.5f) / feat_size;
    
    // Scale to search size
    cx *= search_size;
    cy *= search_size;
    float w = pred_w * search_size;
    float h = pred_h * search_size;
    
    return BBox(cx, cy, w, h);  // Return as cx, cy, w, h
}

BBox mapBoxBack(const BBox& pred_box,
                const BBox& prev_box,
                float resize_factor,
                int search_size) {
    // pred_box is in (cx, cy, w, h) format in search region coordinates
    float cx_prev = prev_box.cx();
    float cy_prev = prev_box.cy();
    
    float half_side = 0.5f * search_size / resize_factor;
    
    // Map center back to image coordinates
    float cx_real = pred_box.x / resize_factor + (cx_prev - half_side);
    float cy_real = pred_box.y / resize_factor + (cy_prev - half_side);
    float w = pred_box.w / resize_factor;
    float h = pred_box.h / resize_factor;
    
    // Convert to (x, y, w, h) format
    return BBox(cx_real - w * 0.5f, cy_real - h * 0.5f, w, h);
}

BBox clipBox(const BBox& box, int H, int W, int margin) {
    float x1 = box.x;
    float y1 = box.y;
    float x2 = x1 + box.w;
    float y2 = y1 + box.h;
    
    x1 = std::min(std::max(0.0f, x1), static_cast<float>(W - margin));
    x2 = std::min(std::max(static_cast<float>(margin), x2), static_cast<float>(W));
    y1 = std::min(std::max(0.0f, y1), static_cast<float>(H - margin));
    y2 = std::min(std::max(static_cast<float>(margin), y2), static_cast<float>(H));
    
    float w = std::max(static_cast<float>(margin), x2 - x1);
    float h = std::max(static_cast<float>(margin), y2 - y1);
    
    return BBox(x1, y1, w, h);
}

void drawTrackingResult(cv::Mat& image,
                         const BBox& box,
                         const std::string& state_str,
                         float confidence,
                         const cv::Scalar& color) {
    // Draw bounding box
    cv::rectangle(image, box.toRect(), color, 2);
    
    // Prepare text
    char text[128];
    std::snprintf(text, sizeof(text), "%s: %.2f", state_str.c_str(), confidence);
    
    // Draw text background
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    cv::Rect text_bg(static_cast<int>(box.x), 
                      static_cast<int>(box.y) - text_size.height - 5,
                      text_size.width + 4, 
                      text_size.height + 4);
    
    if (text_bg.y < 0) {
        text_bg.y = static_cast<int>(box.y + box.h) + 2;
    }
    
    cv::rectangle(image, text_bg, color, cv::FILLED);
    
    // Draw text
    cv::putText(image, text, 
                cv::Point(text_bg.x + 2, text_bg.y + text_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
}

cv::Mat hann2d(int height, int width) {
    cv::Mat hann_y = cv::Mat::zeros(height, 1, CV_32F);
    cv::Mat hann_x = cv::Mat::zeros(1, width, CV_32F);
    
    // Create 1D Hanning windows
    for (int i = 0; i < height; ++i) {
        hann_y.at<float>(i) = 0.5f * (1.0f - std::cos(2.0f * CV_PI * i / (height - 1)));
    }
    
    for (int j = 0; j < width; ++j) {
        hann_x.at<float>(j) = 0.5f * (1.0f - std::cos(2.0f * CV_PI * j / (width - 1)));
    }
    
    // Create 2D Hanning window (outer product)
    cv::Mat hann2d = hann_y * hann_x;
    
    return hann2d;
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return 0.0f;
    }
    
    return dot / (norm_a * norm_b);
}

void l2Normalize(std::vector<float>& feature) {
    float norm = 0.0f;
    for (float v : feature) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-8f) {
        for (float& v : feature) {
            v /= norm;
        }
    }
}

std::string getFileExtension(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) {
        return "";
    }
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

bool isVideoFile(const std::string& filename) {
    std::string ext = getFileExtension(filename);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || 
           ext == ".mkv" || ext == ".wmv" || ext == ".flv" ||
           ext == ".webm" || ext == ".m4v";
}

bool isImageFile(const std::string& filename) {
    std::string ext = getFileExtension(filename);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
           ext == ".bmp" || ext == ".tiff" || ext == ".tif" ||
           ext == ".webp";
}

std::vector<std::string> listFiles(const std::string& directory,
                                    const std::string& extension) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().string();
                if (extension.empty() || getFileExtension(filename) == extension) {
                    files.push_back(filename);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing directory: " << e.what() << std::endl;
    }
    
    // Sort files
    std::sort(files.begin(), files.end());
    
    return files;
}

} // namespace robust_tracker
