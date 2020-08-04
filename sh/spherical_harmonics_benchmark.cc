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
  std::vector<Vector3<TypeParam>> sample_dirs;
  std::vector<TypeParam> sample_vals;
  TypeParam lower_bound = static_cast<TypeParam>(-2);
  TypeParam upper_bound = static_cast<TypeParam>(2);
  std::uniform_real_distribution<TypeParam> unif(lower_bound,upper_bound);
  std::default_random_engine re;
  TypeParam rn = unif(re);
  Result results[num_solvers];
  for(int i=1; i<400;) {
    sample_dirs.clear();
    sample_vals.clear();
    for (int t = 0; t < i; t++) {
      TypeParam theta = static_cast<TypeParam>((t + 0.5) * M_PI / i);
      for (int p = 0; p < 10; p++) {
        TypeParam phi = static_cast<TypeParam>((p + t/static_cast<double>(i)) * 2.0 * M_PI / 10.0);
        Vector3<TypeParam> dir = ToVector(phi, theta);
        sample_dirs.push_back(dir);
        sample_vals.push_back(rn);
        rn = rn * static_cast<TypeParam>(0.9) + unif(re) * static_cast<TypeParam>(0.1);
      }
    }

    //run each solver 5 times and average
    const int n = 10;
    const int samples = i*10;

    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> min_duration = std::chrono::duration<double>::max();
    double min_error = std::numeric_limits<double>::max();
    for(int s = 0; s < num_solvers; s++) {
      std::unique_ptr<std::vector<TypeParam>> solution;
      start = std::chrono::high_resolution_clock::now();
      for(int k=0; k<n; k++) {
        solution = ProjectSparseSamples(order, sample_dirs, sample_vals, solvers[s]);
      }
      finish = std::chrono::high_resolution_clock::now();
      results[s].duration = finish - start;

      double error = 0.0;
      for(int d = 0; d<samples; d++) {
        double diff = sample_vals[d] - EvalSHSum(order, *solution, sample_dirs[d]);
        error += diff * diff;
      }
      error /= samples;
      results[s].error = error;
      
      if(results[s].duration < min_duration)
        min_duration = results[s].duration;

      if(error < min_error)
        min_error = error;
    }

    double min_ms = std::chrono::duration_cast<std::chrono::microseconds>(min_duration).count()/(n * 1000.0f);
    for(int s = 0; s < num_solvers; s++) {
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(results[s].duration).count()/(n * 1000.0f);
      double error = results[s].error;
      const char* col = (ms == min_ms) ? ANSI_COLOR_GREEN : ANSI_COLOR_RESET;
      printf("%s%d samples: %08.5f milliseconds %s(%05.2fx)%s avg square error: %011.8f (%08.6fx)\n", solver_names[s].c_str(), samples, ms, col, ms / min_ms, ANSI_COLOR_RESET, error, error / min_error);
    }

    printf("\n");

    if(i<10)
      i++;
    else if(i < 100)
      i+=10;
    else
      i+=100;
  }
}

}  // namespace sh
