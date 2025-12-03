/**
 * @file tracking_state.h
 * @brief Tracking state enumeration for robust tracker
 * 
 * Defines the possible states of the tracking system
 */

#ifndef TRACKING_STATE_H
#define TRACKING_STATE_H

namespace robust_tracker {

/**
 * @enum TrackingState
 * @brief Enumeration representing the current state of the tracker
 */
enum class TrackingState {
    INIT = 0,      ///< Initial state, tracker not yet initialized
    TRACKING = 1,  ///< Normal tracking state with high confidence
    UNCERTAIN = 2, ///< Uncertain tracking, target may be occluded
    LOST = 3       ///< Target is lost
};

/**
 * @brief Convert TrackingState to string representation
 * @param state The tracking state to convert
 * @return String representation of the state
 */
inline const char* trackingStateToString(TrackingState state) {
    switch (state) {
        case TrackingState::INIT:      return "INIT";
        case TrackingState::TRACKING:  return "TRACKING";
        case TrackingState::UNCERTAIN: return "UNCERTAIN";
        case TrackingState::LOST:      return "LOST";
        default:                       return "UNKNOWN";
    }
}

} // namespace robust_tracker

#endif // TRACKING_STATE_H
