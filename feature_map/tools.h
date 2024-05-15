#ifndef TOOLS_H
#define TOOLS_H

#include <cstdlib>
#include <cmath> 
#include <algorithm> 
#include <string>
#include "SETTINGS.h"

// helfpul constants
const Real EPS = 1e-8;
const Real INF = 1e12;

// camera parameters
const Real ASPECT = 4.0 / 3.0; 
const Real FOV = 65.0; 
const Real NEAR_PLANE = 1.0;
const Real PHONG = 10.0;
const VEC3 EYE = VEC3(0.0, 0.0, 0.0); 
const VEC3 LOOK_AT = VEC3(0.0, 0.0, 1.0); 
const VEC3 GAZE = LOOK_AT - EYE;
const VEC3 UP = VEC3(0.0, 1.0, 0.0); 
const int XRES = 800, YRES = 600; 

typedef unsigned long long ULL; 
typedef unsigned int UI; 

Real random_real () {
  return (rand() + 0.0) / (RAND_MAX + 0.0); 
}

VEC4 to_hom (VEC3 x) {
  return VEC4(x[0], x[1], x[2], 1.0);
}

VEC3 get_barycentric_coordinates (VEC3 a, VEC3 b, VEC3 c, VEC3 x) { 
  // equations copied from Shirley and Marschner ...
  auto n = (b - a).cross(c - a); 
  Real n_norm_sq = n.dot(n);
  Real alpha  = n.dot((c - b).cross(x - b)) / n_norm_sq;
  Real beta   = n.dot((a - c).cross(x - c)) / n_norm_sq;
  Real gamma  = n.dot((b - a).cross(x - a)) / n_norm_sq;
  VEC3 bary = VEC3(alpha, beta, gamma);
  return bary;
}

bool in_range(Real l, Real h, Real x) {
  return l <= x && x <= h;
}

bool check_point_inside_triangle (VEC3 a, VEC3 b, VEC3 c, VEC3 x) {
  VEC3 bary = get_barycentric_coordinates(a, b, c, x); 
  return in_range(0, 1, bary(0)) && in_range(0, 1, bary(1)) && in_range(0, 1, bary(2)); 
}

Real norm (VEC3 a) {
  return sqrt(a.dot(a));
}

VEC3 normalized (VEC3 a) {
  return a / norm(a);
}

VEC3 element_wise_mul (VEC3 a, VEC3 b) { 
  return VEC3(a(0) * b(0), a(1) * b(1), a(2) * b(2));
}

void print_vec2 (VEC2 x) {
  cout << x(0) << " " << x(1) << endl;
}

void print_vec3 (VEC3 x) {
  cout << x(0) << " " << x(1) << " " << x(2) << endl;
}

void print_vec4 (VEC4 x) {
  cout << x(0) << " " << x(1) << " " << x(2) << " " << x(3) << endl;
}

bool almost_equal (Real a, Real b, Real thresh=EPS) {
  return abs(a - b) < thresh;
  // return (abs(a - b) / (0.5 * (abs(a) + abs(b)))) < EPS;
}

Real degree_to_rad (Real degree) { 
  return 2.0 * M_PI * degree / 360.0;
}

VEC3 px_to_near_plane_pt (int i, int j) {
  Real h = abs(2.0 * NEAR_PLANE * tan(degree_to_rad(FOV / 2.0))); 
  Real w = abs(h * ASPECT); 

  Real l = -w / 2.0;
  Real r =  w / 2.0;

  Real t =  h / 2.0;
  Real b = -h / 2.0;

  Real x = (i * l + (XRES - i) * r) / (XRES + 0.0); 
  Real y = (j * t + (YRES - j) * b) / (YRES + 0.0); 
  
  return VEC3(x, y, NEAR_PLANE); 
}

VEC3 reflected_vector (VEC3 n, VEC3 l) { 
  return -l + 2.0 * (n.dot(l)) * n;
}

VEC3 refracted_vector(VEC3 n, VEC3 d, Real n1, Real n2, bool &tir_flag) {
  // ray d is in medium n1 and will hit surface with normal n and enter n2
  VEC3 n_inw = -n;
  VEC3 delta = d - (d.dot(n_inw) * n_inw);
  Real sin_theta_1 = norm(delta);
  Real sin_theta_2 = sin_theta_1 * n1 / n2;
  if ((1.0 - (sin_theta_2 * sin_theta_2)) < 0.0) {
    // your ray is tir'ed. 
    tir_flag = true;
    return VEC3(0.0, 0.0, 0.0);
  }
  Real cos_theta_2 = sqrt(1.0 - (sin_theta_2 * sin_theta_2)); 
  VEC3 new_d = cos_theta_2 * n_inw + sin_theta_2 * normalized(delta); 
  return new_d;
}

float clamp(float value)
{
  if (value < 0.0)      return 0.0;
  else if (value > 1.0) return 1.0;
  return value;
}

int argmin(vector<Real> &xs) {
  if (xs.size() == 0) {
    cout << "(argmin) Don't gimme empty xs!!" << endl;
    exit(1);
  }
  Real xMin = INF;
  int iMin = 0; 
  for (int i = 0; i < xs.size(); i++) {
    if (xs[i] < xMin) {
      iMin = i; 
      xMin = xs[i];
    }
  }
  return iMin;
}

int argmin_thresh(vector<Real> &xs, Real threshold) {
  Real xMin = INF;
  int iMin = -1; 
  for (int i = 0; i < xs.size(); i++) {
    if (xs[i] < xMin && xs[i] > threshold) {
      iMin = i; 
      xMin = xs[i];
    }
  }
  return iMin;
}

int argmax(vector<Real> &xs) {
  if (xs.size() == 0) {
    cout << "(argmax) Don't gimme empty xs!!" << endl;
    exit(1);
  }
  Real xMax = -INF;
  int iMax = 0; 
  for (int i = 0; i < xs.size(); i++) {
    if (xs[i] > xMax) {
      iMax = i; 
      xMax = xs[i];
    }
  }
  return iMax;
}

VEC3 element_wise_abs (VEC3 a) {
  return VEC3(abs(a(0)), abs(a(1)), abs(a(2)));
}

Real min3 (Real a, Real b, Real c) {
  return min(a, min(b, c)); 
}

Real max3 (Real a, Real b, Real c) {
  return max(a, max(b, c)); 
}

Real min4 (Real a, Real b, Real c, Real d) {
  return min(a, min(b, min(c, d))); 
}

Real max4 (Real a, Real b, Real c, Real d) {
  return max(a, max(b, max(c, d))); 
}

VEC3 random_perp (VEC3 d) { 
  // return some perpendicular vector to d
  while (true) {
    VEC3 r (random_real(), random_real(), random_real());     
    if (norm(r.cross(d)) != 0.0) {
      return normalized(r.cross(d));
    }
  }
}

MATRIX3 random_frame (VEC3 d) {
  // return a frame such that d is || to +ve z vector 
  auto z = normalized(d); 
  auto x = random_perp(z); 
  auto y = z.cross(x); 
  
  MATRIX3 m;
  m.col(0) = x; 
  m.col(1) = y; 
  m.col(2) = z;

  auto F = m.inverse();
  return F;
}

void solve_quadratic (Real A, Real B, Real C, vector<Real> &ts) {
  /** 
   * Solve A t ** 2 + B t + C = 0 
   * and put real solutions in ts
   */
   
  if (B * B > 4 * A * C) {
    if (almost_equal(B * B - 4 * A * C, 0.0)) {
      Real t = - B / (2 * A);
      ts.push_back(t); 
    } else {
      Real t1 = (- B - sqrt(B * B - 4 * A * C)) / (2 * A); 
      Real t2 = (- B + sqrt(B * B - 4 * A * C)) / (2 * A); 
      ts.push_back(t1);
      ts.push_back(t2);
    }
  } 
}

// helpful stuff for building LBVH
union MortonCodeType {
  UI byte_rep; 
  float num; 
};

string to_binary (UI num) { 
  string s = ""; 
  for (int i = 31; i >= 0; i--) 
    s += (num & (1 << i)) ? "1" : "0"; 
  return s;
}

string to_ieee754_vis (UI num) { 
  string s = ""; 
  for (int i = 31; i >= 0; i--) { 
    s += (num & (1 << i)) ? "1" : "0"; 
    if (i == 31 || i == 31 - 8) s += "|";
  }
  return s;
}

UI get_exponent (UI f)  {
  // positive floating point so don't worry about sign bit
  return f >> 23; 
}

UI get_mantissa (UI f) {
  return f & ((1 << 23) - 1);
}

UI int_as_float_to_int_as_fp (UI f) { 
  // guaranteed that the floating point representation of f is in [0, 1)
  int off = 126 - get_exponent(f);
  if (off < 0) {
    cout << "(int_as_float_to_int_as_fp): Are you sure you have a floating point in [0, 1)" << endl;
    exit(1);
  }
  UI num = get_mantissa(f); 
  num = (1 << 23) | num; 
  num = num << 8;
  return num >> off;
}

ULL int_as_fixed_point_to_morton_code (UI x, UI y, UI z) {
  ULL res = 0;
  int bit_id = 31;
  for (int i = 63; i >= 2; i -= 3) {
    UI x_bit = (x & (1 << bit_id)) >> bit_id;
    UI y_bit = (y & (1 << bit_id)) >> bit_id;
    UI z_bit = (z & (1 << bit_id)) >> bit_id;
    
    res = res | (x_bit << i); 
    res = res | (y_bit << (i - 1)); 
    res = res | (z_bit << (i - 2)); 

    bit_id -= 1;
  }
  return res;
}

ULL point_to_morton_code (VEC3 pt) { 
  float xf = (float) pt(0); 
  float yf = (float) pt(1); 
  float zf = (float) pt(2); 

  MortonCodeType xm; 
  xm.num = xf;

  MortonCodeType ym; 
  ym.num = yf;

  MortonCodeType zm; 
  zm.num = zf;

  UI int_fixed_x = int_as_float_to_int_as_fp(xm.byte_rep); 
  UI int_fixed_y = int_as_float_to_int_as_fp(ym.byte_rep); 
  UI int_fixed_z = int_as_float_to_int_as_fp(zm.byte_rep); 

  ULL morton = int_as_fixed_point_to_morton_code(int_fixed_x, int_fixed_y, int_fixed_z);

  return morton;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
UI expandBits(UI v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
UI morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    UI xx = expandBits((unsigned int)x);
    UI yy = expandBits((unsigned int)y);
    UI zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

ULL point_to_morton_code_karras (VEC3 pt) {
  // morton code construction from https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
  float xf = (float) pt(0); 
  float yf = (float) pt(1); 
  float zf = (float) pt(2); 
  return ((ULL) morton3D(xf, yf, zf)); 
}

void range (vector<int> &vals, int n) {
  vals.clear();
  for (int i = 0; i < n; i++)
    vals.push_back(i); 
}

struct arg_sort_comp {
  vector<ULL> *vals;
  arg_sort_comp (vector<ULL> *vals) : vals(vals) {}
  bool operator () (int i, int j) { return (*vals)[i] < (*vals)[j]; }
};

void sort_by_morton (vector<VEC3> &pts, vector<int> &sorted_ids) { 
  range(sorted_ids, pts.size()); 
  vector<ULL> morton_codes; 
  for (auto &pt: pts) 
    morton_codes.push_back(point_to_morton_code_karras(pt));
  arg_sort_comp comp(&morton_codes);
  sort(sorted_ids.begin(), sorted_ids.end(), comp); 
}

VEC3 projectile_motion (VEC3 x_init, VEC3 v, VEC3 a, Real t) {
  VEC3 x_fin(0.0, 0.0, 0.0); 
  for (int i = 0; i < 3; i++) 
    x_fin(i) = x_init(i) + v(i) * t + 0.5 * a(i) * t * t; 
  return x_fin;
}

VEC3 projectile_motion_vel(VEC3 v_init, VEC3 a, Real t) { 
  return v_init + a * t;
}

Real get_fractional_part (Real a){ 
  Real integral; 
  Real fractional = modf(a, &integral);
  return fractional;
}

int get_integral_part (Real a){ 
  Real integral; 
  modf(a, &integral);
  return (int) integral;
}

#endif

