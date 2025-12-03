/**
 * @file reid_feature_bank.cpp
 * @brief Implementation of ReID feature bank
 */

#include "reid_feature_bank.h"
#include <stdexcept>

namespace robust_tracker {

ReIDFeatureBank::ReIDFeatureBank(int bank_size, 
                                   int update_interval,
                                   float similarity_threshold)
    : bank_size_(bank_size)
    , update_interval_(update_interval)
    , similarity_threshold_(similarity_threshold) {
    if (bank_size <= 0) {
        throw std::invalid_argument("Bank size must be positive");
    }
    if (similarity_threshold < 0.0f || similarity_threshold > 1.0f) {
        throw std::invalid_argument("Similarity threshold must be between 0 and 1");
    }
}

void ReIDFeatureBank::reset() {
    features_.clear();
}

void ReIDFeatureBank::registerFeature(const std::vector<float>& feature) {
    if (feature.empty()) {
        return;
    }
    
    // Add feature to the bank
    features_.push_back(feature);
    
    // Remove oldest if exceeding capacity
    while (static_cast<int>(features_.size()) > bank_size_) {
        features_.pop_front();
    }
}

bool ReIDFeatureBank::isSimilar(const std::vector<float>& feature) const {
    // If feature bank is empty, assume match (no reference to compare)
    if (features_.empty()) {
        return true;
    }
    
    // If input feature is empty, cannot match
    if (feature.empty()) {
        return false;
    }
    
    return getMaxSimilarity(feature) >= similarity_threshold_;
}

float ReIDFeatureBank::getMaxSimilarity(const std::vector<float>& feature) const {
    if (feature.empty() || features_.empty()) {
        return 0.0f;
    }
    
    float max_sim = -1.0f;
    for (const auto& stored_feature : features_) {
        float sim = cosineSimilarity(feature, stored_feature);
        if (sim > max_sim) {
            max_sim = sim;
        }
    }
    
    return max_sim;
}

float ReIDFeatureBank::cosineSimilarity(const std::vector<float>& a, 
                                          const std::vector<float>& b) {
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

} // namespace robust_tracker
