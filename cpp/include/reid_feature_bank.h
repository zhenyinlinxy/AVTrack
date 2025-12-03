/**
 * @file reid_feature_bank.h
 * @brief ReID feature bank for target re-identification
 * 
 * Manages ReID features for robust target matching
 */

#ifndef REID_FEATURE_BANK_H
#define REID_FEATURE_BANK_H

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace robust_tracker {

/**
 * @class ReIDFeatureBank
 * @brief Manages a bank of ReID features for target matching
 * 
 * This class maintains a circular buffer of feature vectors and provides
 * methods for feature registration, similarity computation, and matching.
 */
class ReIDFeatureBank {
public:
    /**
     * @brief Constructor
     * @param bank_size Maximum number of features to store
     * @param update_interval Frames between feature updates
     * @param similarity_threshold Threshold for feature matching
     */
    ReIDFeatureBank(int bank_size = 10, 
                    int update_interval = 20,
                    float similarity_threshold = 0.7f);
    
    /**
     * @brief Reset the feature bank
     */
    void reset();
    
    /**
     * @brief Register a new feature in the bank
     * @param feature Feature vector to register
     */
    void registerFeature(const std::vector<float>& feature);
    
    /**
     * @brief Check if a feature is similar to features in the bank
     * @param feature Feature vector to compare
     * @return true if similar to stored features
     */
    bool isSimilar(const std::vector<float>& feature) const;
    
    /**
     * @brief Get the maximum similarity with stored features
     * @param feature Feature vector to compare
     * @return Maximum similarity score (0.0 to 1.0)
     */
    float getMaxSimilarity(const std::vector<float>& feature) const;
    
    /**
     * @brief Get the number of features stored
     * @return Number of features in the bank
     */
    size_t size() const { return features_.size(); }
    
    /**
     * @brief Check if the bank is empty
     * @return true if no features stored
     */
    bool empty() const { return features_.empty(); }
    
    /**
     * @brief Get the bank size limit
     * @return Maximum number of features
     */
    int getBankSize() const { return bank_size_; }
    
    /**
     * @brief Get the update interval
     * @return Update interval in frames
     */
    int getUpdateInterval() const { return update_interval_; }
    
    /**
     * @brief Get the similarity threshold
     * @return Similarity threshold value
     */
    float getSimilarityThreshold() const { return similarity_threshold_; }

private:
    /**
     * @brief Compute cosine similarity between two vectors
     * @param a First vector
     * @param b Second vector
     * @return Cosine similarity (-1.0 to 1.0)
     */
    static float cosineSimilarity(const std::vector<float>& a, 
                                   const std::vector<float>& b);
    
    std::deque<std::vector<float>> features_;  ///< Feature buffer
    int bank_size_;                             ///< Maximum features to store
    int update_interval_;                       ///< Frames between updates
    float similarity_threshold_;                ///< Matching threshold
};

} // namespace robust_tracker

#endif // REID_FEATURE_BANK_H
