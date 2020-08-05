#include "sh/spherical_harmonics.h"
#include "gtest/gtest.h"

#include <random>
#include <chrono>

namespace sh {

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

struct Result {
  Result() : duration(0), error(0) {}
  std::chrono::duration<double> duration;
  double error;
};

template <typename T>
class TypedSphericalHarmonicsBechmark : public testing::Test { };

typedef ::testing::Types<double, float> Implementations;
TYPED_TEST_CASE(TypedSphericalHarmonicsBechmark, Implementations);

TYPED_TEST(TypedSphericalHarmonicsBechmark, ProjectSparseSamples) {

  printf("benchmarking performance for %s precision\n", std::is_same<TypeParam, float>::value ? "float" : "double");

  constexpr size_t num_solvers = 10;

  static const SolverType solvers[num_solvers] =
      { SolverType::kJacobiSVD,
        SolverType::kBdcsSVD,
        SolverType::kHouseholderQR,
        SolverType::kColPivHouseholderQR,
        SolverType::kFullPivHouseholderQR,
        SolverType::kLDLT,
        SolverType::kLLT,
        SolverType::kCompleteOrthogonalDecomposition,
        SolverType::kPartialPivLU,
        SolverType::kFullPivLU };

  static const std::string solver_names[num_solvers] = 
      { "kJacobiSVD                       ",
        "kBdcsSVD                         ",
        "kHouseholderQR                   ",
        "kColPivHouseholderQR             ",
        "kFullPivHouseholderQR            ",
        "kLDLT                            ",
        "kLLT                             ",
        "kCompleteOrthogonalDecomposition ",
        "kPartialPivLU                    ",
        "kFullPivLU                       " };

  constexpr int order = 4;

  // Generate sparse samples
  algn_vector<Vector3<TypeParam>> sample_dirs;
  algn_vector<TypeParam> sample_vals;
  TypeParam lower_bound = static_cast<TypeParam>(-2);
  TypeParam upper_bound = static_cast<TypeParam>(2);
  std::uniform_real_distribution<TypeParam> unif(lower_bound,upper_bound);
  std::default_random_engine re;
  TypeParam rn = unif(re);
  Result results[num_solvers];
  
  // run each solver n times and average
  const int n = 10;
  for(int i=1; i<4000;) {
    sample_dirs.clear();
    sample_vals.clear();
    const int samples = i;

    for(int j = 0; j < n; j++) {
      for(int d = 0; d < i; d++) {
        sample_vals.push_back(unif(re));
        Vector3<TypeParam> dir(unif(re), unif(re), unif(re));
        dir.normalize();
        sample_dirs.push_back(dir);
      }

      for(int s = 0; s < num_solvers; s++) {
        algn_vector<TypeParam> solution;
        auto start = std::chrono::high_resolution_clock::now();
        ProjectSparseSamples(order, sample_dirs, sample_vals, &solution, solvers[s]);
        auto finish = std::chrono::high_resolution_clock::now();
        results[s].duration += finish - start;

        double error = 0.0;
        for(int d = 0; d<samples; d++) {
          double diff = sample_vals[d] - EvalSHSum(order, solution, sample_dirs[d]);
          error += diff * diff;
        }
        error /= samples;
        results[s].error += error;
      }
    }

    std::chrono::duration<double> min_duration = std::chrono::duration<double>::max();
    double min_error = std::numeric_limits<double>::max();
    for(int s = 0; s < num_solvers; s++) {
      if(results[s].duration < min_duration)
        min_duration = results[s].duration;
      if(results[s].error < min_error)
        min_error = results[s].error; 
    }

    double min_ms = std::chrono::duration_cast<std::chrono::microseconds>(min_duration).count()/(n * 1000.0f);
    min_error /= n;
    for(int s = 0; s < num_solvers; s++) {
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(results[s].duration).count()/(n * 1000.0f);
      double error = results[s].error / n;
      printf("%s%d samples: %08.5f milliseconds (%05.2fx) avg square error: %011.8f (%08.6fx)\n", solver_names[s].c_str(), samples, ms, ms / min_ms, error, error / min_error);
    }

    printf("\n");

    if(i<10)
      i++;
    else if(i < 100)
      i+=10;
    else if(i < 1000)
      i+=100;
    else
      i+=1000;
  }
}

}  // namespace sh
