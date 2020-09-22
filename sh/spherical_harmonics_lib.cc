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

EXPORT_API void ProjectWeightedSparseSamplesLinear(
    uint32_t order,
    float* dirs,
    float* r_values,
    float* g_values,
    float* b_values,
    float* weights,
    float* r_coeffs_out,
    float* g_coeffs_out,
    float* b_coeffs_out,
    uint32_t count,
    uint32_t min_samples_per_basis
    ) {
    sh::algn_vector<sh::Vector3<float>> dir_vec;
    dir_vec.resize(count);
    memcpy((void*)dir_vec.data(), dirs, count * 3 * sizeof(float));

    sh::algn_vector<float> r_value_vec;
    r_value_vec.resize(count);
    memcpy(r_value_vec.data(), r_values, count * sizeof(float));

    sh::algn_vector<float> g_value_vec;
    g_value_vec.resize(count);
    memcpy(g_value_vec.data(), g_values, count * sizeof(float));

    sh::algn_vector<float> b_value_vec;
    b_value_vec.resize(count);
    memcpy(b_value_vec.data(), b_values, count * sizeof(float));

    sh::algn_vector<float> weight_vec;
    weight_vec.resize(count);
    memcpy(weight_vec.data(), weights, count * sizeof(float));

    sh::algn_vector<float> r_coeffs_out_vec;
    sh::algn_vector<float> g_coeffs_out_vec;
    sh::algn_vector<float> b_coeffs_out_vec;

    sh::algn_vector<size_t> index_array;
    index_array.reserve(count);
    for(int i=0; i<count; i++) {
      index_array.push_back(i);
    }

    sh::algn_vector<size_t> num_values_array;
    num_values_array.push_back(count);

    int num_problems = 1;
    switch(order) {
      case 0:
        sh::ProjectWeightedSparseSampleStream<float, 0>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 1:
        sh::ProjectWeightedSparseSampleStream<float, 1>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 2:
        sh::ProjectWeightedSparseSampleStream<float, 2>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 3:
        sh::ProjectWeightedSparseSampleStream<float, 3>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 4:
        sh::ProjectWeightedSparseSampleStream<float, 4>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
    }

    memcpy(r_coeffs_out, r_coeffs_out_vec.data(), r_coeffs_out_vec.size() * sizeof(float));
    memcpy(g_coeffs_out, g_coeffs_out_vec.data(), g_coeffs_out_vec.size() * sizeof(float));
    memcpy(b_coeffs_out, b_coeffs_out_vec.data(), b_coeffs_out_vec.size() * sizeof(float));
}

EXPORT_API void ProjectWeightedSparseSamplesConstrained(
    uint32_t order,
    float* dirs,
    float* r_values,
    float* g_values,
    float* b_values,
    float* weights,
    float* r_coeffs_out,
    float* g_coeffs_out,
    float* b_coeffs_out,
    uint32_t count,
    uint32_t min_samples_per_basis
    ) {
    sh::algn_vector<sh::Vector3<float>> dir_vec;
    dir_vec.resize(count);
    memcpy((void*)dir_vec.data(), dirs, count * 3 * sizeof(float));

    sh::algn_vector<float> r_value_vec;
    r_value_vec.resize(count);
    memcpy(r_value_vec.data(), r_values, count * sizeof(float));

    sh::algn_vector<float> g_value_vec;
    g_value_vec.resize(count);
    memcpy(g_value_vec.data(), g_values, count * sizeof(float));

    sh::algn_vector<float> b_value_vec;
    b_value_vec.resize(count);
    memcpy(b_value_vec.data(), b_values, count * sizeof(float));

    sh::algn_vector<float> weight_vec;
    weight_vec.resize(count);
    memcpy(weight_vec.data(), weights, count * sizeof(float));

    sh::algn_vector<float> r_coeffs_out_vec;
    sh::algn_vector<float> g_coeffs_out_vec;
    sh::algn_vector<float> b_coeffs_out_vec;

    sh::algn_vector<size_t> index_array;
    index_array.reserve(count);
    for(int i=0; i<count; i++) {
      index_array.push_back(i);
    }

    sh::algn_vector<size_t> num_values_array;
    num_values_array.push_back(count);

    int num_problems = 1;
    switch(order) {
      case 0:
        sh::ProjectConstrainedWeightedSparseSampleStream<float, 0>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 1:
        sh::ProjectConstrainedWeightedSparseSampleStream<float, 1>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 2:
        sh::ProjectConstrainedWeightedSparseSampleStream<float, 2>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 3:
        sh::ProjectConstrainedWeightedSparseSampleStream<float, 3>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
      case 4:
        sh::ProjectConstrainedWeightedSparseSampleStream<float, 4>(
            num_problems, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
            index_array, num_values_array,
            &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec, sh::SolverType::kJacobiSVD,
            min_samples_per_basis);
      break;
    }

    memcpy(r_coeffs_out, r_coeffs_out_vec.data(), r_coeffs_out_vec.size() * sizeof(float));
    memcpy(g_coeffs_out, g_coeffs_out_vec.data(), g_coeffs_out_vec.size() * sizeof(float));
    memcpy(b_coeffs_out, b_coeffs_out_vec.data(), b_coeffs_out_vec.size() * sizeof(float));
}

EXPORT_API void ProjectWeightedSparseSamplesMultiple(
    uint32_t order,
    float* dirs,
    float* r_values,
    float* g_values,
    float* b_values,
    float* weights,
    float* r_coeffs_out,
    float* g_coeffs_out,
    float* b_coeffs_out,
    uint32_t* values_per_problem,
    uint32_t dir_count,
    uint32_t problem_count,
    uint32_t min_samples_per_basis
    ) {
    sh::algn_vector<sh::Vector3<float>> dir_vec;
    dir_vec.resize(dir_count);
    memcpy((void*)dir_vec.data(), dirs, dir_count * 3 * sizeof(float));

    sh::algn_vector<float> r_value_vec;
    r_value_vec.resize(dir_count);
    memcpy(r_value_vec.data(), r_values, dir_count * sizeof(float));

    sh::algn_vector<float> g_value_vec;
    g_value_vec.resize(dir_count);
    memcpy(g_value_vec.data(), g_values, dir_count * sizeof(float));

    sh::algn_vector<float> b_value_vec;
    b_value_vec.resize(dir_count);
    memcpy(b_value_vec.data(), b_values, dir_count * sizeof(float));

    sh::algn_vector<float> weight_vec;
    weight_vec.resize(dir_count);
    memcpy(weight_vec.data(), weights, dir_count * sizeof(float));

    sh::algn_vector<float> r_coeffs_out_vec;
    sh::algn_vector<float> g_coeffs_out_vec;
    sh::algn_vector<float> b_coeffs_out_vec;

    sh::algn_vector<size_t> index_array;
    index_array.reserve(dir_count);
    int idx = 0;
    sh::algn_vector<size_t> num_values_array;
    num_values_array.reserve(problem_count);
    for(int i=0; i<problem_count; i++) {
      size_t num_values = values_per_problem[i];
      num_values_array.push_back(num_values);
      for(int k=0; k<num_values; k++) {
        index_array.push_back(idx);
        idx++;
      }
    }

    sh::ProjectMultipleWeightedSparseSamples<float, 4>(
        problem_count, dir_vec, r_value_vec, g_value_vec, b_value_vec, weight_vec,
        index_array, num_values_array,
        &r_coeffs_out_vec, &g_coeffs_out_vec, &b_coeffs_out_vec,
        min_samples_per_basis);

    memcpy(r_coeffs_out, r_coeffs_out_vec.data(), r_coeffs_out_vec.size() * sizeof(float));
    memcpy(g_coeffs_out, g_coeffs_out_vec.data(), g_coeffs_out_vec.size() * sizeof(float));
    memcpy(b_coeffs_out, b_coeffs_out_vec.data(), b_coeffs_out_vec.size() * sizeof(float));
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