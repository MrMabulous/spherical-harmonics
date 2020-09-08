#include "spherical_harmonics.h"

#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API
#endif

extern "C"
{

EXPORT_API int GetCoefficientCount(int order) {
  return sh::GetCoefficientCount(order);
}

EXPORT_API void ProjectWeightedSparseSamples(
    uint32_t order,
    float* dirs,
    float* values,
    float* weights,
    float* coeffs_out,
    uint32_t count
    ) {
    sh::algn_vector<sh::Vector3<float>> dir_vec;
    dir_vec.resize(count);
    memcpy((void*)dir_vec.data(), dirs, count * 3 * sizeof(float));
    sh::algn_vector<float> value_vec;
    value_vec.resize(count);
    memcpy(value_vec.data(), values, count * sizeof(float));
    sh::algn_vector<float> weight_vec;
    weight_vec.resize(count);
    memcpy(weight_vec.data(), weights, count * sizeof(float));
    std::unique_ptr<sh::algn_vector<float>> res =
        sh::ProjectWeightedSparseSamples(order, dir_vec, value_vec, weight_vec);
    memcpy(coeffs_out, res->data(), res->size() * sizeof(float));
}

/*
EXPORT_API void ResampleWeightedSparseSamples(
    uint32_t order,
    float* dirs,
    float* values,
    float* weights,
    float* coeffs_out,
    uint32_t count
    ) {

    sh::SphericalFunction<float, float> resampling = [&](float phi, float theta) => {
      sh::Vector3f d = sh::ToVector(phi, theta);
    }

    sh::algn_vector<sh::Vector3<float>> dir_vec;
    dir_vec.resize(count);
    memcpy((void*)dir_vec.data(), dirs, count * 3 * sizeof(float));
    sh::algn_vector<float> value_vec;
    value_vec.resize(count);
    memcpy(value_vec.data(), values, count * sizeof(float));
    sh::algn_vector<float> weight_vec;
    weight_vec.resize(count);
    memcpy(weight_vec.data(), weights, count * sizeof(float));
    std::unique_ptr<sh::algn_vector<float>> res =
        sh::ProjectWeightedSparseSamples(order, dir_vec, value_vec, weight_vec);
    memcpy(coeffs_out, res->data(), res->size() * sizeof(float));
}
*/

}  // extern "C"