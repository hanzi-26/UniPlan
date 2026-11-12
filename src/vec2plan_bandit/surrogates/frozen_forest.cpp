// Single-thread batch inference for numeric, symmetric CatBoost ensembles.
// Model weights are supplied unchanged. No fast-math or reduced tree count.
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" int predict_frozen_forest(
    const float* x, int rows, int columns, int models, int comparisons,
    const int32_t* feature, const float* border,
    const int32_t* model_start, const int32_t* split_start,
    const int32_t* leaf_start, const int32_t* depth,
    const int32_t* split, const double* leaf,
    const double* scale, const double* bias, double* output) {
    try {
        std::vector<uint8_t> bits(size_t(comparisons) * rows);
        std::vector<uint32_t> index(rows);
        for (int i = 0; i < comparisons; ++i) {
            const float* column = x + size_t(feature[i]) * rows;
            uint8_t* result = bits.data() + size_t(i) * rows;
            for (int row = 0; row < rows; ++row)
                result[row] = column[row] > border[i];
        }
        for (int model = 0; model < models; ++model) {
            double* result = output + size_t(model) * rows;
            for (int row = 0; row < rows; ++row) result[row] = 0.;
            for (int tree = model_start[model]; tree < model_start[model + 1]; ++tree) {
                for (int row = 0; row < rows; ++row) index[row] = 0;
                const int32_t* tests = split + split_start[tree];
                for (int level = 0; level < depth[tree]; ++level) {
                    const uint8_t* outcomes = bits.data() + size_t(tests[level]) * rows;
                    for (int row = 0; row < rows; ++row)
                        index[row] |= uint32_t(outcomes[row]) << level;
                }
                const double* values = leaf + leaf_start[tree];
                for (int row = 0; row < rows; ++row) result[row] += values[index[row]];
            }
            for (int row = 0; row < rows; ++row)
                result[row] = result[row] * scale[model] + bias[model];
        }
        return 0;
    } catch (...) { return 1; }
}
