/**
 * @file main.cpp
 * @brief Main entry point for robust tracker application
 * 
 * Command-line application for visual object tracking with ReID support
 */

#include "robust_tracker.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief Simple command-line argument parser
 */
class ArgParser {
public:
    ArgParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            args_.push_back(std::string(argv[i]));
        }
    }
    
    bool hasOption(const std::string& option) const {
        return std::find(args_.begin(), args_.end(), option) != args_.end();
    }
    
    std::string getOption(const std::string& option, const std::string& default_value = "") const {
        auto it = std::find(args_.begin(), args_.end(), option);
        if (it != args_.end() && ++it != args_.end()) {
            return *it;
        }
        return default_value;
    }
    
    int getOptionInt(const std::string& option, int default_value) const {
        std::string val = getOption(option);
        if (val.empty()) return default_value;
        try {
            return std::stoi(val);
        } catch (...) {
            return default_value;
        }
    }
    
    float getOptionFloat(const std::string& option, float default_value) const {
        std::string val = getOption(option);
        if (val.empty()) return default_value;
        try {
            return std::stof(val);
        } catch (...) {
            return default_value;
        }
    }
    
    std::vector<std::string> getOptionList(const std::string& option) const {
        std::vector<std::string> result;
        auto it = std::find(args_.begin(), args_.end(), option);
        if (it != args_.end()) {
            ++it;
            while (it != args_.end() && (*it)[0] != '-') {
                result.push_back(*it);
                ++it;
            }
        }
        return result;
    }

private:
    std::vector<std::string> args_;
};

void printUsage(const char* program_name) {
    std::cout << "Robust Visual Object Tracker with ReID Support\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  --onnx <path>              Path to tracking ONNX model\n";
    std::cout << "  --video <path>             Path to video file\n";
    std::cout << "  OR\n";
    std::cout << "  --folder <path>            Path to image folder\n\n";
    std::cout << "Optional:\n";
    std::cout << "  --providers <list>         ONNX execution providers (default: CPUExecutionProvider)\n";
    std::cout << "  --save <path>              Output path for saving results\n";
    std::cout << "  --template-size <int>      Template size (default: 128)\n";
    std::cout << "  --search-size <int>        Search region size (default: 256)\n";
    std::cout << "  --template-factor <float>  Template area factor (default: 2.0)\n";
    std::cout << "  --search-factor <float>    Search area factor (default: 4.0)\n";
    std::cout << "  --conf-high <float>        High confidence threshold (default: 0.8)\n";
    std::cout << "  --conf-medium <float>      Medium confidence threshold (default: 0.3)\n";
    std::cout << "  --conf-low <float>         Low confidence threshold (default: 0.15)\n";
    std::cout << "  --max-lost-frames <int>    Maximum lost frames before giving up (default: 3000)\n";
    std::cout << "  --reid-onnx <path>         Path to ReID ONNX model\n";
    std::cout << "  --reid-bank-size <int>     ReID feature bank size (default: 10)\n";
    std::cout << "  --reid-update-interval <int>  ReID update interval in frames (default: 20)\n";
    std::cout << "  --reid-similarity-threshold <float>  ReID similarity threshold (default: 0.7)\n";
    std::cout << "  --no-display               Disable visualization window\n";
    std::cout << "  --help                     Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --onnx model.onnx --video input.mp4\n";
    std::cout << "  " << program_name << " --onnx model.onnx --video input.mp4 --save output.mp4\n";
    std::cout << "  " << program_name << " --onnx model.onnx --folder images/ --reid-onnx osnet.onnx\n";
    std::cout << "  " << program_name << " --onnx model.onnx --video input.mp4 --providers CUDAExecutionProvider CPUExecutionProvider\n";
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    
    // Check for help
    if (parser.hasOption("--help") || parser.hasOption("-h") || argc < 2) {
        printUsage(argv[0]);
        return 0;
    }
    
    // Get required parameters
    std::string onnx_path = parser.getOption("--onnx");
    std::string video_path = parser.getOption("--video");
    std::string folder_path = parser.getOption("--folder");
    
    // Validate required parameters
    if (onnx_path.empty()) {
        std::cerr << "Error: --onnx parameter is required\n";
        printUsage(argv[0]);
        return 1;
    }
    
    if (video_path.empty() && folder_path.empty()) {
        std::cerr << "Error: Either --video or --folder parameter is required\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Check if ONNX file exists
    if (!fs::exists(onnx_path)) {
        std::cerr << "Error: ONNX model file not found: " << onnx_path << std::endl;
        return 1;
    }
    
    // Get optional parameters
    std::vector<std::string> providers = parser.getOptionList("--providers");
    if (providers.empty()) {
        providers.push_back("CPUExecutionProvider");
    }
    
    std::string save_path = parser.getOption("--save");
    bool display = !parser.hasOption("--no-display");
    
    // Build tracker configuration
    robust_tracker::TrackerConfig config;
    config.template_size = parser.getOptionInt("--template-size", 128);
    config.search_size = parser.getOptionInt("--search-size", 256);
    config.template_factor = parser.getOptionFloat("--template-factor", 2.0f);
    config.search_factor = parser.getOptionFloat("--search-factor", 4.0f);
    config.conf_high = parser.getOptionFloat("--conf-high", 0.8f);
    config.conf_medium = parser.getOptionFloat("--conf-medium", 0.3f);
    config.conf_low = parser.getOptionFloat("--conf-low", 0.15f);
    config.max_lost_frames = parser.getOptionInt("--max-lost-frames", 3000);
    config.reid_bank_size = parser.getOptionInt("--reid-bank-size", 10);
    config.reid_update_interval = parser.getOptionInt("--reid-update-interval", 20);
    config.reid_similarity_threshold = parser.getOptionFloat("--reid-similarity-threshold", 0.7f);
    
    // Print configuration
    std::cout << "=== Robust Tracker Configuration ===\n";
    std::cout << "ONNX Model: " << onnx_path << "\n";
    std::cout << "Template Size: " << config.template_size << "\n";
    std::cout << "Search Size: " << config.search_size << "\n";
    std::cout << "Template Factor: " << config.template_factor << "\n";
    std::cout << "Search Factor: " << config.search_factor << "\n";
    std::cout << "Confidence Thresholds: high=" << config.conf_high 
              << ", medium=" << config.conf_medium 
              << ", low=" << config.conf_low << "\n";
    std::cout << "Max Lost Frames: " << config.max_lost_frames << "\n";
    std::cout << "Providers: ";
    for (const auto& p : providers) std::cout << p << " ";
    std::cout << "\n";
    
    // Create tracker
    std::cout << "\nInitializing tracker...\n";
    robust_tracker::RobustTracker tracker(onnx_path, config, providers);
    
    // Load ReID model if specified
    std::string reid_onnx_path = parser.getOption("--reid-onnx");
    if (!reid_onnx_path.empty()) {
        if (!fs::exists(reid_onnx_path)) {
            std::cerr << "Warning: ReID model file not found: " << reid_onnx_path << std::endl;
        } else {
            std::cout << "Loading ReID model: " << reid_onnx_path << "\n";
            if (tracker.setReIDModel(reid_onnx_path)) {
                std::cout << "ReID model loaded successfully\n";
            } else {
                std::cerr << "Warning: Failed to load ReID model\n";
            }
        }
    }
    
    // Process video or folder
    int frame_count = 0;
    
    if (!video_path.empty()) {
        if (!fs::exists(video_path)) {
            std::cerr << "Error: Video file not found: " << video_path << std::endl;
            return 1;
        }
        std::cout << "\nProcessing video: " << video_path << "\n";
        frame_count = robust_tracker::processVideoRobust(tracker, video_path, save_path, display);
    } else {
        if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
            std::cerr << "Error: Folder not found or not a directory: " << folder_path << std::endl;
            return 1;
        }
        std::cout << "\nProcessing folder: " << folder_path << "\n";
        frame_count = robust_tracker::batchProcessFolderRobust(tracker, folder_path, save_path, display);
    }
    
    std::cout << "\nProcessed " << frame_count << " frames\n";
    
    if (!save_path.empty()) {
        std::cout << "Output saved to: " << save_path << "\n";
    }
    
    return 0;
}
