// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sh/spherical_harmonics.h"

#include <iostream>
#include <limits>
#include <random>

namespace sh {

namespace {

// Number of precomputed factorials and double-factorials that can be
// returned in constant time.
const int kCacheSize = 16;

const int kHardCodedOrderLimit = 4;

const int kIrradianceOrder = 2;
const int kIrradianceCoeffCount = GetCoefficientCount(kIrradianceOrder);

// For efficiency, the cosine lobe for normal = (0, 0, 1) as the first 9
// spherical harmonic coefficients are hardcoded below. This was computed by
// evaluating:
//   ProjectFunction(kIrradianceOrder, [] (double phi, double theta) {
//     return Clamp(Eigen::Vector3d::UnitZ().dot(ToVector(phi, theta)), 
//                  0.0, 1.0);
//   }, 10000000);
template <typename T>
const std::vector<T> cosine_lobe = { static_cast<T>(0.886227),
                                     static_cast<T>(0.0),
                                     static_cast<T>(1.02333),
                                     static_cast<T>(0.0),
                                     static_cast<T>(0.0),
                                     static_cast<T>(0.0),
                                     static_cast<T>(0.495416),
                                     static_cast<T>(0.0),
                                     static_cast<T>(0.0) };

// A zero template is required for EvalSHSum to handle its template
// instantiations and a type's default constructor does not necessarily
// initialize to zero.
template<typename T> T Zero();
template<> double Zero() { return 0.0; }
template<> float Zero() { return 0.0; }
template<> Eigen::Array3f Zero() { return Eigen::Array3f::Zero(); }

template <class T>
using VectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Usage: CHECK(bool, string message);
// Note that it must end a semi-colon, making it look like a
// valid C++ statement (hence the awkward do() while(false)).
#ifndef NDEBUG
# define CHECK(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "Check failed (" #condition ") in " << __FILE__ \
        << ":" << __LINE__ << ", message: " << message << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(false)
#else
# define CHECK(condition, message) do {} while(false)
#endif

// Clamp the first argument to be greater than or equal to the second
// and less than or equal to the third.
template <typename T>
T Clamp(T val, T min, T max) {
  if (val < min) {
    val = min;
  }
  if (val > max) {
    val = max;
  }
  return val;
}

// Return true if the first value is within epsilon of the second value.
template <typename T>
bool NearByMargin(T actual, T expected) {
  T diff = actual - expected;
  if (diff < 0.0) {
    diff = -diff;
  }
  // 5 bits of error in mantissa (source of '32 *')
  return diff < 32 * std::numeric_limits<T>::epsilon();
}

// Return floating mod x % m.
template <typename T>
T FastFMod(T x, T m) {
  return x - (m * floor(x / m));
}

// Hardcoded spherical harmonic functions for low orders (l is first number
// and m is second number (sign encoded as preceeding 'p' or 'n')).
//
// As polynomials they are evaluated more efficiently in cartesian coordinates,
// assuming that @d is unit. This is not verified for efficiency.
template <typename T>
T HardcodedSH00(const Vector3<T>& d) {
  // 0.5 * sqrt(1/pi)
  return static_cast<T>(0.282095);
}

template <typename T>
T HardcodedSH1n1(const Vector3<T>& d) {
  // -sqrt(3/(4pi)) * y
  return static_cast<T>(-0.488603) * d.y();
}

template <typename T>
T HardcodedSH10(const Vector3<T>& d) {
  // sqrt(3/(4pi)) * z
  return static_cast<T>(0.488603) * d.z();
}

template <typename T>
T HardcodedSH1p1(const Vector3<T>& d) {
  // -sqrt(3/(4pi)) * x
  return static_cast<T>(-0.488603) * d.x();
}

template <typename T>
T HardcodedSH2n2(const Vector3<T>& d) {
  // 0.5 * sqrt(15/pi) * x * y
  return static_cast<T>(1.092548) * d.x() * d.y();
}

template <typename T>
T HardcodedSH2n1(const Vector3<T>& d) {
  // -0.5 * sqrt(15/pi) * y * z
  return static_cast<T>(-1.092548) * d.y() * d.z();
}

template <typename T>
T HardcodedSH20(const Vector3<T>& d) {
  // 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
  return static_cast<T>(0.315392) * 
      (-d.x() * d.x() - d.y() * d.y() + 2.0 * d.z() * d.z());
}

template <typename T>
T HardcodedSH2p1(const Vector3<T>& d) {
  // -0.5 * sqrt(15/pi) * x * z
  return static_cast<T>(-1.092548) * d.x() * d.z();
}

template <typename T>
T HardcodedSH2p2(const Vector3<T>& d) {
  // 0.25 * sqrt(15/pi) * (x^2 - y^2)
  return static_cast<T>(0.546274) * (d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH3n3(const Vector3<T>& d) {
  // -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
  return static_cast<T>(-0.590044) *
      d.y() * (3.0 * d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH3n2(const Vector3<T>& d) {
  // 0.5 * sqrt(105/pi) * x * y * z
  return static_cast<T>(2.890611) * d.x() * d.y() * d.z();
}

template <typename T>
T HardcodedSH3n1(const Vector3<T>& d) {
  // -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
  return static_cast<T>(-0.457046) *
      d.y() * (4.0 * d.z() * d.z() - d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH30(const Vector3<T>& d) {
  // 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
  return static_cast<T>(0.373176) *
      d.z() * (2.0 * d.z() * d.z() - 3.0 * d.x() * d.x() - 3.0 * d.y() * d.y());
}

template <typename T>
T HardcodedSH3p1(const Vector3<T>& d) {
  // -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
  return static_cast<T>(-0.457046) *
      d.x() * (4.0 * d.z() * d.z() - d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH3p2(const Vector3<T>& d) {
  // 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
  return static_cast<T>(1.445306) * d.z() * (d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH3p3(const Vector3<T>& d) {
  // -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
  return static_cast<T>(-0.590044) *
      d.x() * (d.x() * d.x() - 3.0 * d.y() * d.y());
}

template <typename T>
T HardcodedSH4n4(const Vector3<T>& d) {
  // 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
  return static_cast<T>(2.503343) *
      d.x() * d.y() * (d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH4n3(const Vector3<T>& d) {
  // -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
  return static_cast<T>(-1.770131) *
      d.y() * d.z() * (3.0 * d.x() * d.x() - d.y() * d.y());
}

template <typename T>
T HardcodedSH4n2(const Vector3<T>& d) {
  // 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
  return static_cast<T>(0.946175) *
      d.x() * d.y() * (7.0 * d.z() * d.z() - 1.0);
}

template <typename T>
T HardcodedSH4n1(const Vector3<T>& d) {
  // -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
  return static_cast<T>(-0.669047) *
      d.y() * d.z() * (7.0 * d.z() * d.z() - 3.0);
}

template <typename T>
T HardcodedSH40(const Vector3<T>& d) {
  // 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
  T z2 = d.z() * d.z();
  return static_cast<T>(0.105786) *
      (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
}

template <typename T>
T HardcodedSH4p1(const Vector3<T>& d) {
  // -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
  return static_cast<T>(-0.669047) *
      d.x() * d.z() * (7.0 * d.z() * d.z() - 3.0);
}

template <typename T>
T HardcodedSH4p2(const Vector3<T>& d) {
  // 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
  return static_cast<T>(0.473087) * (d.x() * d.x() - d.y() * d.y())
      * (7.0 * d.z() * d.z() - 1.0);
}

template <typename T>
T HardcodedSH4p3(const Vector3<T>& d) {
  // -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
  return static_cast<T>(-1.770131) *
      d.x() * d.z() * (d.x() * d.x() - 3.0 * d.y() * d.y());
}

template <typename T>
T HardcodedSH4p4(const Vector3<T>& d) {
  // 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
  T x2 = d.x() * d.x();
  T y2 = d.y() * d.y();
  return static_cast<T>(0.625836) *
      (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2));
}

// Compute the factorial for an integer @x. It is assumed x is at least 0.
// This implementation precomputes the results for low values of x, in which
// case this is a constant time lookup.
//
// The vast majority of SH evaluations will hit these precomputed values.
double Factorial(int x) {
  const double factorial_cache[kCacheSize] = {1, 1, 2, 6, 24, 120, 720, 5040,
                                              40320, 362880, 3628800, 39916800,
                                              479001600, 6227020800,
                                              87178291200, 1307674368000};

  if (x < kCacheSize) {
    return factorial_cache[x];
  } else {
    double s = 1.0;
    for (int n = 2; n <= x; n++) {
      s *= n;
    }
    return s;
  }
}

// Compute the double factorial for an integer @x. This assumes x is at least
// 0.  This implementation precomputes the results for low values of x, in
// which case this is a constant time lookup.
//
// The vast majority of SH evaluations will hit these precomputed values.
// See http://mathworld.wolfram.com/DoubleFactorial.html
double DoubleFactorial(int x) {
  const double dbl_factorial_cache[kCacheSize] = {1, 1, 2, 3, 8, 15, 48, 105,
                                                  384, 945, 3840, 10395, 46080,
                                                  135135, 645120, 2027025};

  if (x < kCacheSize) {
    return dbl_factorial_cache[x];
  } else {
    double s = 1.0;
    double n = x;
    while (n > 1.0) {
      s *= n;
      n -= 2.0;
    }
    return s;
  }
}

// Evaluate the associated Legendre polynomial of degree @l and order @m at
// coordinate @x. The inputs must satisfy:
// 1. l >= 0
// 2. 0 <= m <= l
// 3. -1 <= x <= 1
// See http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
//
// This implementation is based off the approach described in [1],
// instead of computing Pml(x) directly, Pmm(x) is computed. Pmm can be
// lifted to Pmm+1 recursively until Pml is found
double EvalLegendrePolynomial(int l, int m, double x) {
  // Compute Pmm(x) = (-1)^m(2m - 1)!!(1 - x^2)^(m/2), where !! is the double
  // factorial.
  double pmm = 1.0;
  // P00 is defined as 1.0, do don't evaluate Pmm unless we know m > 0
  if (m > 0) {
    double sign = (m % 2 == 0 ? 1 : -1);
    pmm = sign * DoubleFactorial(2 * m - 1) * pow(1 - x * x, m / 2.0);
  }

  if (l == m) {
    // Pml is the same as Pmm so there's no lifting to higher bands needed
    return pmm;
  }

  // Compute Pmm+1(x) = x(2m + 1)Pmm(x)
  double pmm1 = x * (2 * m + 1) * pmm;
  if (l == m + 1) {
    // Pml is the same as Pmm+1 so we are done as well
    return pmm1;
  }

  // Use the last two computed bands to lift up to the next band until l is
  // reached, using the recurrence relationship:
  // Pml(x) = (x(2l - 1)Pml-1 - (l + m - 1)Pml-2) / (l - m)
  for (int n = m + 2; n <= l; n++) {
    double pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m);
    pmm = pmm1;
    pmm1 = pmn;
  }
  // Pmm1 at the end of the above loop is equal to Pml
  return pmm1;
}

// ---- The following functions are used to implement SH rotation computations
//      based on the recursive approach described in [1, 4]. The names of the
//      functions correspond with the notation used in [1, 4].

// See http://en.wikipedia.org/wiki/Kronecker_delta
template <typename T>
T KroneckerDelta(int i, int j) {
  if (i == j) {
    return 1.0;
  } else {
    return 0.0;
  }
}

// [4] uses an odd convention of referring to the rows and columns using
// centered indices, so the middle row and column are (0, 0) and the upper
// left would have negative coordinates.
//
// This is a convenience function to allow us to access an Eigen::MatrixXd
// in the same manner, assuming r is a (2l+1)x(2l+1) matrix.
template <typename T>
T GetCenteredElement(const MatrixX<T>& r, int i, int j) {
  // The shift to go from [-l, l] to [0, 2l] is (rows - 1) / 2 = l,
  // (since the matrix is assumed to be square, rows == cols).
  int offset = (r.rows() - 1) / 2;
  return r(i + offset, j + offset);
}

// P is a helper function defined in [4] that is used by the functions U, V, W.
// This should not be called on its own, as U, V, and W (and their coefficients)
// select the appropriate matrix elements to access (arguments @a and @b).
template <typename T>
T P(int i, int a, int b, int l, const std::vector<MatrixX<T>>& r) {
  if (b == l) {
    return GetCenteredElement(r[1], i, 1) *
        GetCenteredElement(r[l - 1], a, l - 1) -
        GetCenteredElement(r[1], i, -1) *
        GetCenteredElement(r[l - 1], a, -l + 1);
  } else if (b == -l) {
    return GetCenteredElement(r[1], i, 1) *
        GetCenteredElement(r[l - 1], a, -l + 1) +
        GetCenteredElement(r[1], i, -1) *
        GetCenteredElement(r[l - 1], a, l - 1);
  } else {
    return GetCenteredElement(r[1], i, 0) * GetCenteredElement(r[l - 1], a, b);
  }
}

// The functions U, V, and W should only be called if the correspondingly
// named coefficient u, v, w from the function ComputeUVWCoeff() is non-zero.
// When the coefficient is 0, these would attempt to access matrix elements that
// are out of bounds. The list of rotations, @r, must have the @l - 1
// previously completed band rotations. These functions are valid for l >= 2.
template <typename T>
T U(int m, int n, int l, const std::vector<MatrixX<T>>& r) {
  // Although [1, 4] split U into three cases for m == 0, m < 0, m > 0
  // the actual values are the same for all three cases
  return P(0, m, n, l, r);
}

template <typename T>
T V(int m, int n, int l, const std::vector<MatrixX<T>>& r) {
  if (m == 0) {
    return P(1, 1, n, l, r) + P(-1, -1, n, l, r);
  } else if (m > 0) {
    return P(1, m - 1, n, l, r) * sqrt(1 + KroneckerDelta<T>(m, 1)) -
        P(-1, -m + 1, n, l, r) * (1 - KroneckerDelta<T>(m, 1));
  } else {
    // Note there is apparent errata in [1,4,4b] dealing with this particular
    // case. [4b] writes it should be P*(1-d)+P*(1-d)^0.5
    // [1] writes it as P*(1+d)+P*(1-d)^0.5, but going through the math by hand,
    // you must have it as P*(1-d)+P*(1+d)^0.5 to form a 2^.5 term, which
    // parallels the case where m > 0.
    return P(1, m + 1, n, l, r) * (1 - KroneckerDelta<T>(m, -1)) +
        P(-1, -m - 1, n, l, r) * sqrt(1 + KroneckerDelta<T>(m, -1));
  }
}

template <typename T>
T W(int m, int n, int l, const std::vector<MatrixX<T>>& r) {
  if (m == 0) {
    // whenever this happens, w is also 0 so W can be anything
    return 0.0;
  } else if (m > 0) {
    return P(1, m + 1, n, l, r) + P(-1, -m - 1, n, l, r);
  } else {
    return P(1, m - 1, n, l, r) - P(-1, -m + 1, n, l, r);
  }
}

// Calculate the coefficients applied to the U, V, and W functions. Because
// their equations share many common terms they are computed simultaneously.
template <typename T>
void ComputeUVWCoeff(int m, int n, int l, T* u, T* v, T* w) {
  T d = KroneckerDelta<T>(m, 0);
  T denom = (abs(n) == l ? 2.0 * l * (2.0 * l - 1) : (l + n) * (l - n));

  *u = sqrt((l + m) * (l - m) / denom);
  *v = 0.5 * sqrt((1 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom)
      * (1 - 2 * d);
  *w = -0.5 * sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d);
}

// Calculate the (2l+1)x(2l+1) rotation matrix for the band @l.
// This uses the matrices computed for band 1 and band l-1 to compute the
// matrix for band l. @rotations must contain the previously computed l-1
// rotation matrices, and the new matrix for band l will be appended to it.
//
// This implementation comes from p. 5 (6346), Table 1 and 2 in [4] taking
// into account the corrections from [4b].
template <typename T>
void ComputeBandRotation(int l, std::vector<MatrixX<T>>* rotations) {
  // The band's rotation matrix has rows and columns equal to the number of
  // coefficients within that band (-l <= m <= l implies 2l + 1 coefficients).
  MatrixX<T> rotation(2 * l + 1, 2 * l + 1);
  for (int m = -l; m <= l; m++) {
    for (int n = -l; n <= l; n++) {
      T u, v, w;
      ComputeUVWCoeff(m, n, l, &u, &v, &w);

      // The functions U, V, W are only safe to call if the coefficients
      // u, v, w are not zero
      if (!NearByMargin(u, 0.0))
          u *= U(m, n, l, *rotations);
      if (!NearByMargin(v, 0.0))
          v *= V(m, n, l, *rotations);
      if (!NearByMargin(w, 0.0))
          w *= W(m, n, l, *rotations);

      rotation(m + l, n + l) = (u + v + w);
    }
  }

  rotations->push_back(rotation);
}

}  // namespace
template <typename S>
Vector3<S> ToVector(S phi, S theta) {
  S r = sin(theta);
  return Vector3<S>(r * cos(phi), r * sin(phi), cos(theta));
}

template <typename S>
void ToSphericalCoords(const Vector3<S>& dir, S* phi, S* theta) {
  CHECK(NearByMargin(dir.squaredNorm(), static_cast<S>(1.0)),
        "dir is not unit");
  // Explicitly clamp the z coordinate so that numeric errors don't cause it
  // to fall just outside of acos' domain.
  *theta = acos(Clamp(dir.z(), static_cast<S>(-1.0), static_cast<S>(1.0)));
  // We don't need to divide dir.y() or dir.x() by sin(theta) since they are
  // both scaled by it and atan2 will handle it appropriately.
  *phi = atan2(dir.y(), dir.x());
}

template <typename S>
S ImageXToPhi(int x, int width) {
  // The directions are measured from the center of the pixel, so add 0.5
  // to convert from integer pixel indices to float pixel coordinates.
  return static_cast<S>(2.0 * M_PI) * (x + static_cast<S>(0.5)) / width;
}

template <typename S>
S ImageYToTheta(int y, int height) {
  return static_cast<S>(M_PI) * (y + static_cast<S>(0.5)) / height;
}

template <typename S>
Vector2<S> ToImageCoords(S phi, S theta, int width, int height) {
  // Allow theta to repeat and map to 0 to pi. However, to account for cases
  // where y goes beyond the normal 0 to pi range, phi may need to be adjusted.
  theta = Clamp(FastFMod(theta, static_cast<S>(2.0 * M_PI)),
                         static_cast<S>(0.0), static_cast<S>(2.0 * M_PI));
  if (theta > static_cast<S>(M_PI)) {
    // theta is out of bounds. Effectively, theta has rotated past the pole
    // so after adjusting theta to be in range, rotating phi by pi forms an
    // equivalent direction.
    theta = static_cast<S>(2.0 * M_PI) - theta;  // now theta is between 0 and pi
    phi += static_cast<S>(M_PI);
  }
  // Allow phi to repeat and map to the normal 0 to 2pi range.
  // Clamp and map after adjusting theta in case theta was forced to update phi.
  phi = Clamp(FastFMod(phi, static_cast<S>(2.0 * M_PI)),
                       static_cast<S>(0.0), static_cast<S>(2.0 * M_PI));

  // Now phi is in [0, 2pi] and theta is in [0, pi] so it's simple to inverse
  // the linear equations in ImageCoordsToSphericalCoords, although there's no
  // -0.5 because we're returning floating point coordinates and so don't need
  // to center the pixel.
  return Vector2<S>(width * phi / static_cast<S>(2.0 * M_PI),
                    height * theta / static_cast<S>(M_PI));
}

template <typename T, typename S>
T EvalSHSlow(int l, int m, S phi, S theta) {
  CHECK(l >= 0, "l must be at least 0.");
  CHECK(-l <= m && m <= l, "m must be between -l and l.");

  double kml = sqrt((2.0 * l + 1) * Factorial(l - abs(m)) /
              (4.0 * M_PI * Factorial(l + abs(m))));
  if (m > 0) {
    return sqrt(2.0) * kml * cos(m * phi) *
        EvalLegendrePolynomial(l, m, cos(theta));
  } else if (m < 0) {
    return sqrt(2.0) * kml * sin(-m * phi) *
        EvalLegendrePolynomial(l, -m, cos(theta));
  } else {
    return kml * EvalLegendrePolynomial(l, 0, cos(theta));
  }
}

template <typename T, typename S>
T EvalSHSlow(int l, int m, const Vector3<S>& dir) {
  S phi, theta;
  ToSphericalCoords(dir, &phi, &theta);
  return EvalSH<T,S>(l, m, phi, theta);
}

template <typename T, typename S>
T EvalSH(int l, int m, S phi, S theta) {
  // If using the hardcoded functions, switch to cartesian
  if (l <= kHardCodedOrderLimit) {
    return EvalSH<T,S>(l, m, ToVector(phi, theta));
  } else {
    // Stay in spherical coordinates since that's what the recurrence
    // version is implemented in
    return EvalSHSlow<T,S>(l, m, phi, theta);
  }
}

template <typename T, typename S>
T EvalSH(int l, int m, const Vector3<S>& dir) {
  if (l <= kHardCodedOrderLimit) {
    // Validate l and m here (don't do it generally since EvalSHSlow also
    // checks it if we delegate to that function).
    CHECK(l >= 0, "l must be at least 0.");
    CHECK(-l <= m && m <= l, "m must be between -l and l.");
    CHECK(NearByMargin(dir.squaredNorm(), static_cast<S>(1.0)),
          "dir is not unit.");

    switch (l) {
      case 0:
        return static_cast<T>(HardcodedSH00(dir));
      case 1:
        switch (m) {
          case -1:
            return static_cast<T>(HardcodedSH1n1(dir));
          case 0:
            return static_cast<T>(HardcodedSH10(dir));
          case 1:
            return static_cast<T>(HardcodedSH1p1(dir));
        }
      case 2:
        switch (m) {
          case -2:
            return static_cast<T>(HardcodedSH2n2(dir));
          case -1:
            return static_cast<T>(HardcodedSH2n1(dir));
          case 0:
            return static_cast<T>(HardcodedSH20(dir));
          case 1:
            return static_cast<T>(HardcodedSH2p1(dir));
          case 2:
            return static_cast<T>(HardcodedSH2p2(dir));
        }
      case 3:
        switch (m) {
          case -3:
            return static_cast<T>(HardcodedSH3n3(dir));
          case -2:
            return static_cast<T>(HardcodedSH3n2(dir));
          case -1:
            return static_cast<T>(HardcodedSH3n1(dir));
          case 0:
            return static_cast<T>(HardcodedSH30(dir));
          case 1:
            return static_cast<T>(HardcodedSH3p1(dir));
          case 2:
            return static_cast<T>(HardcodedSH3p2(dir));
          case 3:
            return static_cast<T>(HardcodedSH3p3(dir));
        }
      case 4:
        switch (m) {
          case -4:
            return static_cast<T>(HardcodedSH4n4(dir));
          case -3:
            return static_cast<T>(HardcodedSH4n3(dir));
          case -2:
            return static_cast<T>(HardcodedSH4n2(dir));
          case -1:
            return static_cast<T>(HardcodedSH4n1(dir));
          case 0:
            return static_cast<T>(HardcodedSH40(dir));
          case 1:
            return static_cast<T>(HardcodedSH4p1(dir));
          case 2:
            return static_cast<T>(HardcodedSH4p2(dir));
          case 3:
            return static_cast<T>(HardcodedSH4p3(dir));
          case 4:
            return static_cast<T>(HardcodedSH4p4(dir));
        }
    }

    // This is unreachable given the CHECK's above but the compiler can't tell.
    return static_cast<T>(0.0);
  } else {
    // Not hard-coded so use the recurrence relation (which will convert this
    // to spherical coordinates).
    return EvalSHSlow<T,S>(l, m, dir);
  }
}

template <typename T, typename S>
std::unique_ptr<std::vector<T>> ProjectFunction(
    int order, const SphericalFunction<T,S>& func, int sample_count) {
  CHECK(order >= 0, "Order must be at least zero.");
  CHECK(sample_count > 0, "Sample count must be at least one.");

  // This is the approach demonstrated in [1] and is useful for arbitrary
  // functions on the sphere that are represented analytically.
  const int sample_side = static_cast<int>(floor(sqrt(sample_count)));
  std::unique_ptr<std::vector<T>> coeffs(new std::vector<T>());
  coeffs->assign(GetCoefficientCount(order), static_cast<T>(0.0));

  // generate sample_side^2 uniformly and stratified samples over the sphere
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<S> rng(0.0, 1.0);
  for (int t = 0; t < sample_side; t++) {
    for (int p = 0; p < sample_side; p++) {
      S alpha = (t + rng(gen)) / sample_side;
      S beta = (p + rng(gen)) / sample_side;
      // See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
      S phi = 2.0 * M_PI * beta;
      S theta = acos(2.0 * alpha - 1.0);

      // evaluate the analytic function for the current spherical coords
      T func_value = func(phi, theta);

      // evaluate the SH basis functions up to band O, scale them by the
      // function's value and accumulate them over all generated samples
      for (int l = 0; l <= order; l++) {
        for (int m = -l; m <= l; m++) {
          T sh = EvalSH<T,S>(l, m, phi, theta);
          (*coeffs)[GetIndex(l, m)] += func_value * sh;
        }
      }
    }
  }

  // scale by the probability of a particular sample, which is
  // 4pi/sample_side^2. 4pi for the surface area of a unit sphere, and
  // 1/sample_side^2 for the number of samples drawn uniformly.
  T weight = 4.0 * M_PI / (sample_side * sample_side);
  for (unsigned int i = 0; i < coeffs->size(); i++) {
     (*coeffs)[i] *= weight;
  }

  return coeffs;
}

std::unique_ptr<std::vector<Eigen::Array3f>> ProjectEnvironment(
    int order, const Image& env) {
  CHECK(order >= 0, "Order must be at least zero.");

  // An environment map projection is three different spherical functions, one
  // for each color channel. The projection integrals are estimated by
  // iterating over every pixel within the image.
  double pixel_area = (2.0 * M_PI / env.width()) * (M_PI / env.height());

  std::unique_ptr<std::vector<Eigen::Array3f>> coeffs(
      new std::vector<Eigen::Array3f>());
  coeffs->assign(GetCoefficientCount(order), Eigen::Array3f(0.0, 0.0, 0.0));

  Eigen::Array3f color;
  for (int t = 0; t < env.height(); t++) {
    double theta = ImageYToTheta<double>(t, env.height());
    // The differential area of each pixel in the map is constant across a
    // row. Must scale the pixel_area by sin(theta) to account for the
    // stretching that occurs at the poles with this parameterization.
    double weight = pixel_area * sin(theta);

    for (int p = 0; p < env.width(); p++) {
      double phi = ImageXToPhi<double>(p, env.width());
      color = env.GetPixel(p, t);

      for (int l = 0; l <= order; l++) {
        for (int m = -l; m <= l; m++) {
          int i = GetIndex(l, m);
          double sh = EvalSH<double, double>(l, m, phi, theta);
          (*coeffs)[i] += sh * weight * color.array();
        }
      }
    }
  }

  return coeffs;
}

template <typename T>
std::unique_ptr<std::vector<T>> ProjectSparseSamples(
    int order, const std::vector<Vector3<T>>& dirs, 
    const std::vector<T>& values,
    SvdType svdType) {
  CHECK(order >= 0, "Order must be at least zero.");
  CHECK(dirs.size() == values.size(),
      "Directions and values must have the same size.");

  // Solve a linear least squares system Ax = b for the coefficients, x.
  // Each row in the matrix A are the values of the spherical harmonic basis
  // functions evaluated at that sample's direction (from @dirs). The
  // corresponding row in b is the value in @values.
  std::unique_ptr<std::vector<T>> coeffs(new std::vector<T>());
  coeffs->assign(GetCoefficientCount(order), 0.0);

  MatrixX<T> basis_values(dirs.size(), coeffs->size());
  VectorX<T> func_values(dirs.size());

  T phi, theta;
  for (unsigned int i = 0; i < dirs.size(); i++) {
    func_values(i) = values[i];
    ToSphericalCoords(dirs[i], &phi, &theta);

    for (int l = 0; l <= order; l++) {
      for (int m = -l; m <= l; m++) {
        basis_values(i, GetIndex(l, m)) = EvalSH<T,T>(l, m, phi, theta);
      }
    }
  }

  // Use SVD to find the least squares fit for the coefficients of the basis
  // functions that best match the data
  VectorX<T> soln;
  switch(svdType) {
    case SvdType::kJacobi:
      soln = basis_values.jacobiSvd(
          Eigen::ComputeThinU | Eigen::ComputeThinV).solve(func_values);
      break;
    case SvdType::kBdcs:
      soln = basis_values.bdcSvd(
          Eigen::ComputeThinU | Eigen::ComputeThinV).solve(func_values);
      break;
    default:
      CHECK(false, "Invalid svdType.");
  }

  // Copy everything over to our coeffs array
  for (unsigned int i = 0; i < coeffs->size(); i++) {
    (*coeffs)[i] = soln(i);
  }
  return coeffs;
}

template <typename T>
std::unique_ptr<std::vector<T>> ProjectWeightedSparseSamples(
    int order, const std::vector<Vector3<T>>& dirs, 
    const std::vector<T>& values, const std::vector<T>& weights,
    SvdType svdType) {
  CHECK(order >= 0, "Order must be at least zero.");
  CHECK(dirs.size() == values.size(),
      "Directions and values must have the same size.");
  CHECK(dirs.size() == weights.size(),
      "Directions and weights must have the same size.");

  // Solve a weighted linear least squares system W^(1/2)*Ax = W^(1/2)*b
  // for the coefficients, x. Each row in the matrix A are the values of the
  // spherical harmonic basis functions evaluated at that sample's direction
  // (from @dirs). The corresponding row in b is the value in @values.
  std::unique_ptr<std::vector<T>> coeffs(new std::vector<T>());
  coeffs->assign(GetCoefficientCount(order), 0.0);

  MatrixX<T> basis_values(dirs.size(), coeffs->size());
  VectorX<T> func_values(dirs.size());
  VectorX<T> weight_values(dirs.size());
  Eigen::DiagonalMatrix<T, Eigen::Dynamic> W(dirs.size());

  T phi, theta;
  for (unsigned int i = 0; i < dirs.size(); i++) {
    func_values(i) = values[i];
    weight_values(i)=sqrt(weights[i]);
    ToSphericalCoords(dirs[i], &phi, &theta);

    for (int l = 0; l <= order; l++) {
      for (int m = -l; m <= l; m++) {
        basis_values(i, GetIndex(l, m)) = EvalSH<T,T>(l, m, phi, theta);
      }
    }
  }

  W.diagonal() = weight_values;

  // Use SVD to find the least squares fit for the coefficients of the basis
  // functions that best match the data
  VectorX<T> soln;
  switch(svdType) {
    case SvdType::kJacobi:
      soln = (W * basis_values).jacobiSvd(
          Eigen::ComputeThinU | Eigen::ComputeThinV).solve(W * func_values);
      break;
    case SvdType::kBdcs:
      soln = (W * basis_values).bdcSvd(
          Eigen::ComputeThinU | Eigen::ComputeThinV).solve(W * func_values);
      break;
    default:
      CHECK(false, "Invalid svdType.");
  }

  // Copy everything over to our coeffs array
  for (unsigned int i = 0; i < coeffs->size(); i++) {
    (*coeffs)[i] = soln(i);
  }
  return coeffs;
}

template <typename T, typename S>
T EvalSHSum(int order, const std::vector<T>& coeffs, S phi, S theta) {
  using SHType = typename std::conditional<
      std::is_same<T, Eigen::Array3f>::value, float, T>::type;
  if (order <= kHardCodedOrderLimit) {
    // It is faster to compute the cartesian coordinates once
    return EvalSHSum<T,S>(order, coeffs, ToVector(phi, theta));
  }

  CHECK(GetCoefficientCount(order) == coeffs.size(),
      "Incorrect number of coefficients provided.");
  T sum = Zero<T>();
  for (int l = 0; l <= order; l++) {
    for (int m = -l; m <= l; m++) {
      sum += EvalSH<SHType,S>(l, m, phi, theta) * coeffs[GetIndex(l, m)];
    }
  }
  return sum;
}

template <typename T, typename S>
T EvalSHSum(int order, const std::vector<T>& coeffs, 
            const Vector3<S>& dir) {
  using SHType = typename std::conditional<
      std::is_same<T, Eigen::Array3f>::value, float, T>::type;
  if (order > kHardCodedOrderLimit) {
    // It is faster to switch to spherical coordinates
    S phi, theta;
    ToSphericalCoords(dir, &phi, &theta);
    return EvalSHSum<T,S>(order, coeffs, phi, theta);
  }

  CHECK(GetCoefficientCount(order) == coeffs.size(),
        "Incorrect number of coefficients provided.");
  CHECK(NearByMargin(dir.squaredNorm(), static_cast<S>(1.0)),
        "dir is not unit.");

  T sum = Zero<T>();
  for (int l = 0; l <= order; l++) {
    for (int m = -l; m <= l; m++) {
      sum += EvalSH<SHType,S>(l, m, dir) * coeffs[GetIndex(l, m)];
    }
  }
  return sum;
}

Rotation::Rotation(int order, const Eigen::Quaterniond& rotation)
    : order_(order), rotation_(rotation) {
  band_rotations_.reserve(GetCoefficientCount(order));
}

std::unique_ptr<Rotation> Rotation::Create(
    int order, const Eigen::Quaterniond& rotation) {
  CHECK(order >= 0, "Order must be at least 0.");
  CHECK(NearByMargin(rotation.squaredNorm(), 1.0),
        "Rotation must be normalized.");

  std::unique_ptr<Rotation> sh_rot(new Rotation(order, rotation));

  // Order 0 (first band) is simply the 1x1 identity since the SH basis
  // function is a simple sphere.
  Eigen::MatrixXd r(1, 1);
  r(0, 0) = 1.0;
  sh_rot->band_rotations_.push_back(r);

  r.resize(3, 3);
  // The second band's transformation is simply a permutation of the
  // rotation matrix's elements, provided in Appendix 1 of [1], updated to
  // include the Condon-Shortely phase. The recursive method in
  // ComputeBandRotation preserves the proper phases as high bands are computed.
  Eigen::Matrix3d rotation_mat = rotation.toRotationMatrix();
  r(0, 0) = rotation_mat(1, 1);
  r(0, 1) = -rotation_mat(1, 2);
  r(0, 2) = rotation_mat(1, 0);
  r(1, 0) = -rotation_mat(2, 1);
  r(1, 1) = rotation_mat(2, 2);
  r(1, 2) = -rotation_mat(2, 0);
  r(2, 0) = rotation_mat(0, 1);
  r(2, 1) = -rotation_mat(0, 2);
  r(2, 2) = rotation_mat(0, 0);
  sh_rot->band_rotations_.push_back(r);

  // Recursively build the remaining band rotations, using the equations
  // provided in [4, 4b].
  for (int l = 2; l <= order; l++) {
    ComputeBandRotation(l, &(sh_rot->band_rotations_));
  }

  return sh_rot;
}

std::unique_ptr<Rotation> Rotation::Create(int order,
                                           const Rotation& rotation) {
  CHECK(order >= 0, "Order must be at least 0.");

  std::unique_ptr<Rotation> sh_rot(new Rotation(order,
                                                rotation.rotation_));

  // Copy up to min(order, rotation.order_) band rotations into the new
  // SHRotation. For shared orders, they are the same. If the new order is
  // higher than already calculated then the remainder will be computed next.
  for (int l = 0; l <= std::min(order, rotation.order_); l++) {
    sh_rot->band_rotations_.push_back(rotation.band_rotations_[l]);
  }

  // Calculate remaining bands (automatically skipped if there are no more).
  for (int l = rotation.order_ + 1; l <= order; l++) {
    ComputeBandRotation(l, &(sh_rot->band_rotations_));
  }

  return sh_rot;
}

template <typename T>
void Rotation::Apply(const std::vector<T>& coeff,
                     std::vector<T>* result) const {
  CHECK(coeff.size() == GetCoefficientCount(order_),
        "Incorrect number of coefficients provided.");

  // Resize to the required number of coefficients.
  // If result is already the same size as coeff, there's no need to zero out
  // its values since each index will be written explicitly later.
  if (result->size() != coeff.size()) {
    result->assign(coeff.size(), T());
  }

  // Because of orthogonality, the coefficients outside of each band do not
  // interact with one another. By separating them into band-specific matrices,
  // we take advantage of that sparsity.

  for (int l = 0; l <= order_; l++) {
    VectorX<T> band_coeff(2 * l + 1);

    // Fill band_coeff from the subset of @coeff that's relevant.
    for (int m = -l; m <= l; m++) {
      // Offset by l to get the appropiate vector component (0-based instead
      // of starting at -l).
      band_coeff(m + l) = coeff[GetIndex(l, m)];
    }

    band_coeff = band_rotations_[l].cast<T>() * band_coeff;

    // Copy rotated coefficients back into the appropriate subset into @result.
    for (int m = -l; m <= l; m++) {
      (*result)[GetIndex(l, m)] = band_coeff(m + l);
    }
  }
}

void RenderDiffuseIrradianceMap(const Image& env_map, Image* diffuse_out) {
  std::unique_ptr<std::vector<Eigen::Array3f>> coeffs =
      ProjectEnvironment(kIrradianceOrder, env_map);
  RenderDiffuseIrradianceMap(*coeffs, diffuse_out);
}

void RenderDiffuseIrradianceMap(const std::vector<Eigen::Array3f>& sh_coeffs,
                                Image* diffuse_out) {
  for (int y = 0; y < diffuse_out->height(); y++) {
    double theta = ImageYToTheta<double>(y, diffuse_out->height());
    for (int x = 0; x < diffuse_out->width(); x++) {
      double phi = ImageXToPhi<double>(x, diffuse_out->width());
      Vector3<double> normal = ToVector(phi, theta);
      Eigen::Array3f irradiance = RenderDiffuseIrradiance(sh_coeffs, normal);
      diffuse_out->SetPixel(x, y, irradiance);
    }
  }
}

Eigen::Array3f RenderDiffuseIrradiance(
    const std::vector<Eigen::Array3f>& sh_coeffs,
    const Eigen::Vector3d& normal) {
  // Optimization for if sh_coeffs is empty, then there is no environmental
  // illumination so irradiance is 0.0 regardless of the normal.
  if (sh_coeffs.empty()) {
    return Eigen::Array3f(0.0, 0.0, 0.0);
  }

  // Compute diffuse irradiance
  Eigen::Quaterniond rotation;
  rotation.setFromTwoVectors(Eigen::Vector3d::UnitZ(), normal).normalize();

  std::vector<double> rotated_cos(kIrradianceCoeffCount);
  std::unique_ptr<sh::Rotation> sh_rot(Rotation::Create(
      kIrradianceOrder, rotation));
  sh_rot->Apply(cosine_lobe<double>, &rotated_cos);

  Eigen::Array3f sum(0.0, 0.0, 0.0);
  // The cosine lobe is 9 coefficients and after that all bands are assumed to
  // be 0. If sh_coeffs provides more than 9, they are irrelevant then. If it
  // provides fewer than 9, this assumes that the remaining coefficients would
  // have been 0 and can safely ignore the rest of the cosine lobe.
  unsigned int coeff_count = kIrradianceCoeffCount;
  if (coeff_count > sh_coeffs.size()) {
    coeff_count = sh_coeffs.size();
  }
  for (unsigned int i = 0; i < coeff_count; i++) {
    sum += rotated_cos[i] * sh_coeffs[i];
  }
  return sum;
}

// ---- Template specializations -----------------------------------------------
template Vector2<double> ToImageCoords<double>(
    double phi, double theta, int width, int height);
template Vector2<float> ToImageCoords<float>(
    float phi, float theta, int width, int height);

template std::unique_ptr<std::vector<double>> ProjectFunction<double, double>(
    int order, const SphericalFunction<double, double>& func, int sample_count);
template std::unique_ptr<std::vector<float>> ProjectFunction<float, float>(
    int order, const SphericalFunction<float, float>& func, int sample_count);
template std::unique_ptr<std::vector<double>> ProjectFunction<double, float>(
    int order, const SphericalFunction<double, float>& func, int sample_count);
template std::unique_ptr<std::vector<float>> ProjectFunction<float, double>(
    int order, const SphericalFunction<float, double>& func, int sample_count);

template std::unique_ptr<std::vector<double>> ProjectSparseSamples<double>(
    int order, const std::vector<Vector3<double>>& dirs,
    const std::vector<double>& values, SvdType svdType);
template std::unique_ptr<std::vector<float>> ProjectSparseSamples<float>(
    int order, const std::vector<Vector3<float>>& dirs,
    const std::vector<float>& values, SvdType svdType);

template std::unique_ptr<std::vector<double>>
    ProjectWeightedSparseSamples<double>(
      int order, const std::vector<Vector3<double>>& dirs,
      const std::vector<double>& values, const std::vector<double>& weights,
      SvdType svdType);
template std::unique_ptr<std::vector<float>>
    ProjectWeightedSparseSamples<float>(
      int order, const std::vector<Vector3<float>>& dirs,
      const std::vector<float>& values, const std::vector<float>& weights,
      SvdType svdType);

template double EvalSHSum<double, double>(
    int order, const std::vector<double>& coeffs, double phi, double theta);
template float EvalSHSum<float, float>(
    int order, const std::vector<float>& coeffs, float phi, float theta);
template double EvalSHSum<double, float>(
    int order, const std::vector<double>& coeffs, float phi, float theta);
template float EvalSHSum<float, double>(
    int order, const std::vector<float>& coeffs, double phi, double theta);

template Eigen::Array3f EvalSHSum<Eigen::Array3f, double>(
    int order,  const std::vector<Eigen::Array3f>& coeffs,
    double phi, double theta);
template Eigen::Array3f EvalSHSum<Eigen::Array3f, float>(
    int order,  const std::vector<Eigen::Array3f>& coeffs,
    float phi, float theta);

template double EvalSHSum<double, double>(
    int order, const std::vector<double>& coeffs, const Vector3<double>& dir);
template float EvalSHSum<float, float>(
    int order, const std::vector<float>& coeffs, const Vector3<float>& dir);
template double EvalSHSum<double, float>(
    int order, const std::vector<double>& coeffs, const Vector3<float>& dir);
template float EvalSHSum<float, double>(
    int order, const std::vector<float>& coeffs, const Vector3<double>& dir);

template Eigen::Array3f EvalSHSum<Eigen::Array3f, double>(
    int order,  const std::vector<Eigen::Array3f>& coeffs,
    const Vector3<double>& dir);
template Eigen::Array3f EvalSHSum<Eigen::Array3f, float>(
    int order,  const std::vector<Eigen::Array3f>& coeffs,
    const Vector3<float>& dir);

template void Rotation::Apply<double>(const std::vector<double>& coeff,
                                       std::vector<double>* result) const;
template void Rotation::Apply<float>(const std::vector<float>& coeff,
                                      std::vector<float>* result) const;

// The generic implementation for Rotate doesn't handle aggregate types
// like Array3f so split it apart, use the generic version and then recombine
// them into the final result.
template <> void Rotation::Apply<Eigen::Array3f>(
    const std::vector<Eigen::Array3f>& coeff,
    std::vector<Eigen::Array3f>* result) const {
  // Separate the Array3f coefficients into three vectors.
  std::vector<float> c1, c2, c3;
  for (unsigned int i = 0; i < coeff.size(); i++) {
    const Eigen::Array3f& c = coeff[i];
    c1.push_back(c(0));
    c2.push_back(c(1));
    c3.push_back(c(2));
  }

  // Compute the rotation in place
  Apply(c1, &c1);
  Apply(c2, &c2);
  Apply(c3, &c3);

  // Coellesce back into Array3f
  result->assign(GetCoefficientCount(order_), Eigen::Array3f::Zero());
  for (unsigned int i = 0; i < result->size(); i++) {
    (*result)[i] = Eigen::Array3f(c1[i], c2[i], c3[i]);
  }
}

}  // namespace sh
