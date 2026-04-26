#ifndef TURBOQUANT_HPP
#define TURBOQUANT_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

class TurboQuant {
public:
    int d; 
    std::vector<float> centroids;
    std::vector<float> rotation_matrix; // Pi Matrix
    std::vector<int8_t> qjl_signs;      // Stage 3 residuals

    TurboQuant(int dim, int b) : d(dim) {
        float scale = 1.0f / std::sqrt((float)d);
        // Step 1: Centroids (Lloyd-Max)
        if (b == 1) centroids = {-0.798f * scale, 0.798f * scale};
        else centroids = {-1.51f * scale, -0.453f * scale, 0.453f * scale, 1.51f * scale};

        // Step 2: Rotation Matrix Generation
        rotation_matrix.resize(d * d);
        std::mt19937 gen(42); 
        std::normal_distribution<float> dist(0.0f, scale);
        for (float& val : rotation_matrix) val = dist(gen);
    }

    // Stage 1 & 2: Rotate and Quantize (MSE)
    std::vector<int> compress_mse(const std::vector<float>& x) {
        std::vector<float> y = rotate(x);
        std::vector<int> indices(d);
        for (int i = 0; i < d; ++i) {
            indices[i] = find_nearest(y[i]);
        }
        return indices;
    }

    // Stage 3: Residuals with QJL (Algorithm 2)
    // Stage 3: Residuals with QJL in Rotated Space
    void apply_qjl_residual(const std::vector<float>& y, const std::vector<int>& indices) {
        qjl_signs.clear();
        for (int i = 0; i < d; ++i) {
            // y[i] original rotated value hai, centroids[indices[i]] uska quantized version
            float residual = y[i] - centroids[indices[i]];
            qjl_signs.push_back((residual >= 0) ? 1 : -1); 
        }
    }
    // TurboQuant class ke andar ye add karo:
    std::vector<float> decompress_with_qjl(const std::vector<int>& indices, float qjl_delta = 0.1f) {
        std::vector<float> reconstructed(d);
        for (int i = 0; i < d; ++i) {
            // Base quantization value
            float val = centroids[indices[i]];
            
            // Stage 3 Correction: Residual sign ko scale karke add karo
            // Delta (h) ek hyperparameter hai jo error ko minimize karta hai
            val += qjl_signs[i] * qjl_delta;
            
            reconstructed[i] = val;
        }
        return reconstructed;
    }

private:
    std::vector<float> rotate(const std::vector<float>& x) {
        std::vector<float> y(d, 0.0f);
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                y[i] += rotation_matrix[i * d + j] * x[j];
            }
        }
        return y;
    }

    int find_nearest(float val) {
        int best_idx = 0;
        float min_dist = std::abs(val - centroids[0]);
        for (int i = 1; i < centroids.size(); ++i) {
            float dist = std::abs(val - centroids[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
        return best_idx;
    }
};

#endif