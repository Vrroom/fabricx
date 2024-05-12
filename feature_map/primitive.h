#ifndef PRIMITIVE_H
#define PRIMITIVE_H
#include "tools.h"

struct AABB;

/** 
 * For surface type, the following is the enum: 
 * 0 - PHONG
 * 1 - REFLECTION
 * 2 - REFRACTION
 * 3 - TEXTURE
 * 4 - GLOSSY
 */

struct Ray { 
  VEC3 o, d;
  int depth = 0; 
  Ray (VEC3 d): o(0, 0, 0), d(normalized(d)) {} 
  Ray (VEC3 o, VEC3 d): o(o), d(normalized(d)) {}
  Ray (VEC3 o, VEC3 d, int depth, VEC3 n): o(o), d(normalized(d)), depth(depth) {
    this->o += 1e-4 * n; 
  }
  VEC3 point_at_time (Real t) { return o + t * d; } 
  void print () {
    cout << "Ray Origin "; 
    print_vec3(o);
    cout << "Ray Direction "; 
    print_vec3(d);
  }
}; 

struct Primitive {
  VEC3 color;
  int surface_type; // can be 0 - solid, 1 - reflecting, 2 - refracting, 3 - mandelbrot texture

  virtual VEC3 normal (VEC3 pt) {
    cout << "(Primitive::normal) You clearly shouldn't have come here" << endl;
    exit(1);
    return VEC3(0.0, 0.0, 0.0); 
  }

  virtual int get_surface_type () { 
    cout << "(Primitive::get_surface_type) What are you doing here ?!" << endl;
    exit(1);
  } 

  virtual void intersects (Ray &ray, vector<Real> &ts) {
    cout << "(Primitive::intersects) You clearly shouldn't have come here" << endl;
    exit(1);
  } 

  virtual VEC3 get_color () {
    cout << "(Primitive::get_color) You clearly shouldn't have come here" << endl;
    exit(1);
  }

  virtual VEC3 get_texture (VEC3 intersect_pt) {
    cout << "(Primitive::get_texture) You clearly shouldn't have come here" << endl;
    exit(1);
  }

  virtual AABB * to_aabb () {
    cout << "(Primitive::to_aabb) You clearly shouldn't have come here" << endl;
    exit(1);
  }

  virtual void print () {
    cout << "(Primitive::print) You clearly shouldn't have come here" << endl;
    exit(1);
  }
};

struct AABB : public Primitive {
  VEC3 v_lo, v_hi, color; // per coordinate minimum and maximum
  int surface_type; 

  AABB (VEC3 v_lo, VEC3 v_hi) : v_lo(v_lo), v_hi(v_hi), color(0, 0, 0), surface_type(0) {} 

  AABB (VEC3 v_lo, VEC3 v_hi, VEC3 color) : v_lo(v_lo), v_hi(v_hi), color(color), surface_type(0) {} 

  AABB (VEC3 v_lo, VEC3 v_hi, VEC3 color, int surface_type) : v_lo(v_lo), v_hi(v_hi), color(color), surface_type(surface_type) {} 

  int get_surface_type () override { return surface_type; } 

  VEC3 get_color () override { return color; }

  VEC3 normal (VEC3 pt) override { 
    for (int i = 0; i < 3; i++) {
      if (almost_equal(pt(i), v_lo(i), 1e-5)) {
        bool hit = true;
        for (int j = 0; j < 3; j++) if (j != i) 
          hit = hit && in_range(v_lo(j), v_hi(j), pt(j)); 

        if (hit) {
          VEC3 n(0.0, 0.0, 0.0); 
          n(i) = -1.0;
          return n;
        }
      }

      if (almost_equal(pt(i), v_hi(i), 1e-5)) {
        bool hit = true;
        for (int j = 0; j < 3; j++) if (j != i) 
          hit = hit && in_range(v_lo(j), v_hi(j), pt(j)); 

        if (hit) {
          VEC3 n(0.0, 0.0, 0.0); 
          n(i) = +1.0;
          return n;
        }
      }
    }
    cout << "(AABB::normal) Illegal point, are you sure it intersects?" << endl;
    cout << "(AABB::normal) Post-mortem Report -----------------------" << endl;
    print(); 
    cout << "pt "; 
    print_vec3(pt); 
    return VEC3(0.0, 0.0, 0.0); 
  }

  VEC3 get_texture (VEC3 intersect_pt) override {
    cout << "(AABB::get_texture) Not Implemented" << endl;
    exit(1);
  }

  void intersects (Ray &ray, vector<Real> &ts) override {
    for (int i = 0; i < 3; i++) 
      if (ray.d(i) != 0.0) {
        Real t = (v_lo(i) - ray.o(i)) / ray.d(i); 
        VEC3 pt = ray.point_at_time(t);
        bool hit = true;
        for (int j = 0; j < 3; j++) if (j != i) 
          hit = hit && in_range(v_lo(j), v_hi(j), pt(j)); 

        if (hit) ts.push_back(t); 
      }

    for (int i = 0; i < 3; i++) 
      if (ray.d(i) != 0.0) {
        Real t = (v_hi(i) - ray.o(i)) / ray.d(i); 
        VEC3 pt = ray.point_at_time(t);
        
        bool hit = true;
        for (int j = 0; j < 3; j++) if (j != i) 
          hit = hit && in_range(v_lo(j), v_hi(j), pt(j)); 

        if (hit) ts.push_back(t); 
      }
  }

  AABB * to_aabb () {
    return new AABB(v_lo, v_hi);
  }

  void print () {
    cout << "AABB Lower most point ";
    print_vec3(v_lo); 
    cout << "AABB Upper most point ";
    print_vec3(v_hi); 
  }

  void dilate (Real factor=EPS) {
    v_lo -= VEC3(factor, factor, factor); 
    v_hi += VEC3(factor, factor, factor); 
  }

  VEC3 centroid () {
    return 0.5 * (v_lo + v_hi); 
  }

};

struct Rectangle: public Primitive {
  VEC3 o, d1, d2, color, given_normal = VEC3(0.0, 0.0, 0.0); 
  Real s1, s2;
  int surface_type;
  Image texture;
  Real image_width_scale;

  MATRIX2 DtD; 
  Matrix<Real, 3, 2> D;

  Rectangle () {}

  Rectangle(VEC3 o, VEC3 d1, VEC3 d2, Real s1, Real s2) : o(o), d1(normalized(d1)), d2(normalized(d2)), s1(s1), s2(s2), color(0.0, 0.0, 0.0), surface_type(0) {
    D.col(0) = d1; 
    D.col(1) = d2;
    DtD = D.transpose() * D; 
  }

  Rectangle(VEC3 o, VEC3 d1, VEC3 d2, Real s1, Real s2, VEC3 color) : o(o), d1(normalized(d1)), d2(normalized(d2)), s1(s1), s2(s2), color(color), surface_type(0) {
    D.col(0) = d1; 
    D.col(1) = d2;
    DtD = D.transpose() * D; 
  }

  Rectangle(VEC3 o, VEC3 d1, VEC3 d2, Real s1, Real s2, VEC3 color, int surface_type) : o(o), d1(normalized(d1)), d2(normalized(d2)), s1(s1), s2(s2), color(color), surface_type(surface_type) {
    D.col(0) = d1; 
    D.col(1) = d2;
    DtD = D.transpose() * D; 
  }
  
  Rectangle(VEC3 o, VEC3 d1, VEC3 d2, Real s1, Real s2, VEC3 color, int surface_type, Image texture, Real image_width_scale) : o(o), d1(normalized(d1)), d2(normalized(d2)), s1(s1), s2(s2), color(color), surface_type(surface_type), texture(texture), image_width_scale(image_width_scale) {
    if (surface_type != 3) {
      cout << "(Rectangle::Rectangle) Wrong surface type!" << endl;
      exit(1);
    }
    D.col(0) = d1; 
    D.col(1) = d2;
    DtD = D.transpose() * D; 
  }
  
  Rectangle(VEC3 o, VEC3 d1, VEC3 d2, Real s1, Real s2, VEC3 color, int surface_type, Image texture, Real image_width_scale, VEC3 given_normal) : o(o), d1(normalized(d1)), d2(normalized(d2)), s1(s1), s2(s2), color(color), surface_type(surface_type), texture(texture), image_width_scale(image_width_scale), given_normal(given_normal) {
    if (surface_type != 3) {
      cout << "(Rectangle::Rectangle) Wrong surface type!" << endl;
      exit(1);
    }
    D.col(0) = d1; 
    D.col(1) = d2;
    DtD = D.transpose() * D; 
  }

  int get_surface_type () override { return surface_type; } 

  VEC3 get_color () override { return color; }

  VEC3 normal (VEC3 pt) { 
    if (norm(given_normal) < 0.5) return d1.cross(d2); 
    return given_normal;
  }

  VEC3 get_texture (VEC3 intersect_pt) override {
    if (surface_type != 3) {
      cout << "(Rectangle::get_surface_type) Function called with wrong surface type!" << endl;
      exit(1);
    }
    // image_width_scale decides how many units the image occupies in the x direction.
    Real image_aspect = (texture.yRes + 0.0) / (texture.xRes + 0.0); 
    Real image_height_scale = image_width_scale * image_aspect;
    Real alpha, beta;
    {
      // solve for coordinates of intersect_pt
      VEC2 b = D.transpose() * (intersect_pt - o);
      VEC2 x = DtD.inverse() * b; 
      alpha = x(0), beta = x(1); 
      // convert to image space
      alpha = (texture.xRes + 0.0) * get_fractional_part(alpha / image_width_scale); 
      beta  = (texture.yRes + 0.0) * get_fractional_part(beta  / image_height_scale); 
    }
    // do some fun bilinear interpolation
    int i  = get_integral_part(alpha) % texture.xRes, j = get_integral_part(beta) % texture.yRes;
    int i1 = (i + 1) % texture.xRes, j1 = (j + 1) % texture.yRes;
    // retrieve colors    
    VEC3 a = texture.get_color(i , j); 
    VEC3 b = texture.get_color(i1, j); 
    VEC3 c = texture.get_color(i , j1); 
    VEC3 d = texture.get_color(i1, j1); 
    // blend them carefully
    Real dx = get_fractional_part(alpha), dy = get_fractional_part(beta); 
    return a * (1.0 - dx) * (1.0 - dy) 
        +  b * (      dx) * (1.0 - dy)
        +  c * (1.0 - dx) * (      dy)
        +  d * (      dx) * (      dy);
  }

  void intersects (Ray &ray, vector<Real> &ts) override {
    // first find where ray intersects plane.
    // then decide whether the point on plane is inside rectangle
    // plane is Ax + By + Cz + D = 0. 
    // n = (A, B, C)
    VEC3 n = normal(VEC3(0.0, 0.0, 0.0)); 
    Real D = -n.dot(o); 
    // (ray.o + t*ray.d).dot(n) + D = 0 
    // ray.o.dot(n) + t*ray.d.dot(n) + D = 0
    Real t = (- D - ray.o.dot(n)) / ray.d.dot(n);
    VEC3 pt = ray.point_at_time(t);

    VEC3 a = o;
    VEC3 b = o + d1 * s1; 
    VEC3 c = o + d2 * s2; 
    VEC3 d = o + d1 * s1 + d2 * s2;
    if (check_point_inside_triangle(a, b, c, pt) || check_point_inside_triangle(b, c, d, pt)) 
      ts.push_back(t);
  }

  AABB * to_aabb () {
    VEC3 a = o;
    VEC3 b = o + d1 * s1; 
    VEC3 c = o + d2 * s2; 
    VEC3 d = o + d1 * s1 + d2 * s2;
    VEC3 v_lo(
      min4(a(0), b(0), c(0), d(0)) - 2e-5, 
      min4(a(1), b(1), c(1), d(1)) - 2e-5, 
      min4(a(2), b(2), c(2), d(2)) - 2e-5
    );
    VEC3 v_hi(
      max4(a(0), b(0), c(0), d(0)) + 2e-5, 
      max4(a(1), b(1), c(1), d(1)) + 2e-5, 
      max4(a(2), b(2), c(2), d(2)) + 2e-5
    );

    return new AABB(v_lo, v_hi);
  }

  void print () {
    cout << "Rectangle Origin ";
    print_vec3(o); 
    cout << "Rectangle Direction 1 ";
    print_vec3(d1); 
    cout << "Rectangle Direction 2 ";
    print_vec3(d2); 
    cout << "Rectangle Scale " << s1 << ", " << s2 << endl;
  }

  void rand_stratified_samples (vector<VEC3> &pts, int N) {
    // draw random stratified samples on this rectangle with N being the larger number of grid points
    Real S = max(s1, s2);
    Real dt = S / N; 
    for (Real x = dt / 2.0; x < s1; x += dt) 
      for (Real y = dt / 2.0; y < s2; y += dt) {
        Real rx = (random_real() - 0.5) * dt;
        Real ry = (random_real() - 0.5) * dt;
        pts.push_back(o + d1 * (x + rx) + d2 * (y + ry)); 
      }
  }

};


struct Sphere : public Primitive{
  VEC3 c, color;
  int surface_type;
  Real r;

  Image texture;

  Sphere (VEC3 c, Real r) : c(c), r(r), color(0, 0, 0), surface_type(0) {} 

  Sphere (VEC3 c, Real r, VEC3 color) : c(c), r(r), color(color), surface_type(0) {} 

  Sphere (VEC3 c, Real r, VEC3 color, int surface_type) : c(c), r(r), color(color), surface_type(surface_type) {} 

  Sphere (VEC3 c, Real r, VEC3 color, int surface_type, Image texture) : c(c), r(r), color(color), surface_type(surface_type), texture(texture) {
    if (surface_type != 3) {
      cout << "(Sphere::Sphere) Wrong surface type!" << endl;
      exit(1);
    }
  } 

  int get_surface_type () override { return surface_type; } 

  VEC3 get_color () override { return color; }

  VEC3 normal (VEC3 pt) override { return normalized(pt - c); }

  VEC3 get_texture (VEC3 intersect_pt) override {
    VEC3 n = normal(intersect_pt); 
    Real alpha = ((n(0) / 2.0) + 0.5) * texture.xRes;
    Real beta = ((n(1) / 2.0) + 0.5) * texture.yRes;
    // do some fun bilinear interpolation
    int i  = get_integral_part(alpha) % texture.xRes, j = get_integral_part(beta) % texture.yRes;
    int i1 = (i + 1) % texture.xRes, j1 = (j + 1) % texture.yRes;
    // retrieve colors    
    VEC3 a = texture.get_color(i , j); 
    VEC3 b = texture.get_color(i1, j); 
    VEC3 c = texture.get_color(i , j1); 
    VEC3 d = texture.get_color(i1, j1); 
    // blend them carefully
    Real dx = get_fractional_part(alpha), dy = get_fractional_part(beta); 
    return a * (1.0 - dx) * (1.0 - dy) 
        +  b * (      dx) * (1.0 - dy)
        +  c * (1.0 - dx) * (      dy)
        +  d * (      dx) * (      dy);
  }

  void intersects (Ray &ray, vector<Real> &ts) override {
    Real A = ray.d.dot(ray.d); 
    Real B = 2.0 * (ray.o.dot(ray.d) - this->c.dot(ray.d));
    Real C = - (this->r * this->r) - 2 * this->c.dot(ray.o) + this->c.dot(this->c) + ray.o.dot(ray.o); 
    solve_quadratic(A, B, C, ts);
  }

  AABB * to_aabb () {
    VEC3 v_lo = c - VEC3(r, r, r); 
    VEC3 v_hi = c + VEC3(r, r, r); 
    return new AABB(v_lo, v_hi);
  }

  void print () {
    cout << "Circle Center ";
    print_vec3(c); 
    cout << "Circle Radius ";
    cout << r << endl;
  }
};

struct Triangle : public Primitive {
  VEC3 a, b, c, color; 
  int surface_type;
  
  Triangle (VEC3 a, VEC3 b, VEC3 c) : a(a), b(b), c(c), color(0, 0, 0), surface_type(0) {} 

  Triangle (VEC3 a, VEC3 b, VEC3 c, VEC3 color) : a(a), b(b), c(c), color(color), surface_type(0) {} 

  Triangle (VEC3 a, VEC3 b, VEC3 c, VEC3 color, int surface_type) : a(a), b(b), c(c), color(color), surface_type(surface_type) {} 

  int get_surface_type () override { return surface_type; } 

  VEC3 get_color () override { return color; }
  
  VEC3 normal (VEC3 pt) {
    // independent of pt
    VEC3 n = (a - b).cross(c - b);
    n = -normalized(n);
    return n;
  }

  void intersects (Ray &ray, vector<Real> &ts) override {
    // first find where ray intersects plane.
    // then decide whether the point on plane is inside triangle
    // plane is Ax + By + Cz + D = 0. 
    // n = (A, B, C)
    VEC3 n = normal(VEC3(0.0, 0.0, 0.0)); 
    Real D = -n.dot(a); 
    // (ray.o + t*ray.d).dot(n) + D = 0 
    // ray.o.dot(n) + t*ray.d.dot(n) + D = 0
    Real t = (- D - ray.o.dot(n)) / ray.d.dot(n);
    VEC3 pt = ray.point_at_time(t);

    if (check_point_inside_triangle(a, b, c, pt)) 
      ts.push_back(t);
  }

  AABB * to_aabb () {
    VEC3 v_lo (min3(a(0), b(0), c(0)), min3(a(1), b(1), c(1)), min3(a(2), b(2), c(2))); 
    VEC3 v_hi (max3(a(0), b(0), c(0)), max3(a(1), b(1), c(1)), max3(a(2), b(2), c(2))); 
    return new AABB(v_lo, v_hi);
  }

  void print () {
    cout << "Triangle Point A ";
    print_vec3(a); 
    cout << "Triangle Point B ";
    print_vec3(b); 
    cout << "Triangle Point C ";
    print_vec3(c); 
  }
};

struct Cylinder : public Primitive{
  VEC3 c, d, color;
  Real r, l;
  int surface_type;
  MATRIX3 F, F_inv;

  Cylinder (VEC3 c, VEC3 d, Real r, Real l) : c(c), d(d), r(r), l(l), color(0, 0, 0), surface_type(0) {
    F = random_frame(d); 
    F_inv = F.inverse(); 
  }

  Cylinder (VEC3 c, VEC3 d, Real r, Real l, VEC3 color) : c(c), d(d), r(r), l(l), color(color), surface_type(0) {
    F = random_frame(d); 
    F_inv = F.inverse(); 
  }

  Cylinder (VEC3 c, VEC3 d, Real r, Real l, VEC3 color, int surface_type) : c(c), d(d), r(r), l(l), color(color), surface_type(surface_type) {
    F = random_frame(d);
    F_inv = F.inverse(); 
  }

  int get_surface_type () override { return surface_type; } 

  VEC3 get_color () override { return color; }

  VEC3 normal (VEC3 pt) override {
    VEC3 tr = F * (pt - c);
    Real sq_distance_from_axis = (tr(0) * tr(0)) + (tr(1) * tr(1)); 
    if (almost_equal(r * r, sq_distance_from_axis, 1e-5)
        && in_range(-l / 2, l / 2, tr(2))) {
      VEC3 n = VEC3(tr(0), tr(1), 0.0); 
      return normalized(F_inv * n);
    } else if (sq_distance_from_axis < (r * r)) {
      if (almost_equal(tr(2), -l / 2, 1e-5))
        return F_inv * VEC3(0.0, 0.0, -1.0); 
      else if (almost_equal(tr(2), l / 2, 1e-5)) 
        return F_inv * VEC3(0.0, 0.0,  1.0); 
    }
    cout << "(Cylinder::normal) Illegal point, are you sure it intersects?" << endl;
    cout << "(Cylinder::normal) Post-mortem Report -----------------------" << endl;
    print(); 
    cout << "pt "; 
    print_vec3(pt); 
    cout << "tr "; 
    print_vec3(tr); 
    exit(1);
    return VEC3(0.0, 0.0, 0.0); 
  }

  VEC3 get_texture (VEC3 intersect_pt) override {
    cout << "(Cylinder::get_texture) This method is required for 12.ppm and shouldn't be called here" << endl;
    exit(1);
  }

  void intersects (Ray &ray, vector<Real> &ts) override {
    Ray tr_ray(F * (ray.o - c), F * ray.d); // transformed ray
    // find intersection between ray and infinite tube
    Real A = (tr_ray.d(0) * tr_ray.d(0)) + (tr_ray.d(1) * tr_ray.d(1));
    Real B = 2.0 * ((tr_ray.d(0) * tr_ray.o(0)) + (tr_ray.d(1) * tr_ray.o(1)));
    Real C = (- r * r) + (tr_ray.o(0) * tr_ray.o(0)) + (tr_ray.o(1) * tr_ray.o(1));
    solve_quadratic(A, B, C, ts); 
    // remove intersection points which are not within length
    for (int i = ts.size() - 1; i >= 0; i--) 
      if (!in_range(-l / 2, l / 2, tr_ray.point_at_time(ts[i])(2)))
        ts.erase(ts.begin() + i); // at max 2 points so I won't cry about perf.

    // find intersection between ray and bottom cap
    if (tr_ray.d(2) != 0) { 
      Real t = ((-l / 2.0) - tr_ray.o(2)) / tr_ray.d(2);
      VEC3 pt = tr_ray.point_at_time(t); 
      if (((pt(0) * pt(0)) + (pt(1) * pt(1))) < (r * r))
        ts.push_back(t);
    }
    // find intersection between ray and top cap
    if (tr_ray.d(2) != 0) { 
      Real t = ((+l / 2.0) - tr_ray.o(2)) / tr_ray.d(2);
      VEC3 pt = tr_ray.point_at_time(t); 
      if (((pt(0) * pt(0)) + (pt(1) * pt(1))) < (r * r))
        ts.push_back(t);
    }
  }

  void print() { 
    cout << "Cylinder(VEC3(" << c(0) << "," << c(1) << "," << c(2) << "), VEC3(" << d(0) << "," << d(1) << "," << d(2) << "), " << r << "," << l << ")" << endl;
    // cout << "Cylinder Center "; 
    // print_vec3(c);
    // cout << "Cylinder Axis "; 
    // print_vec3(d);
    // cout << "Cylinder Radius " << r << endl;
    // cout << "Cylinder Length " << l << endl;
  }

  AABB * to_aabb () {
    VEC3 v_lo(0.0, 0.0, 0.0), v_hi(0.0, 0.0, 0.0); 
    VEC3 l1 = c + (0.5 * l * normalized(d)), l2 = c - (0.5 * l * normalized(d));
    VEC3 d1 = normalized(random_perp(d)); 
    VEC3 d2 = normalized(d.cross(d1));
    for (int i = 0; i < 3; i++) {
      Real theta1, theta2;

      if (almost_equal(d2(i), 0.0)) 
        theta1 = M_PI / 2;
      else {
        Real tan_theta = d1(i) / d2(i); 
        theta1 = atan(tan_theta); 
      }

      theta2 = theta1 + M_PI;

      Real p1 = l1(i) + r * sin(theta1) * d1(i) + r * cos(theta1) * d2(i); 
      Real p2 = l1(i) + r * sin(theta2) * d1(i) + r * cos(theta2) * d2(i); 

      Real p3 = l2(i) + r * sin(theta1) * d1(i) + r * cos(theta1) * d2(i); 
      Real p4 = l2(i) + r * sin(theta2) * d1(i) + r * cos(theta2) * d2(i); 
  
      v_lo(i) = min4(p1, p2, p3, p4);
      v_hi(i) = max4(p1, p2, p3, p4);
    }
    return new AABB(v_lo, v_hi);
  }

};

struct Light {

  virtual void sample_points (vector<VEC3> &light_points) {
    cout << "(Light::sample_points) You clearly shouldn't have come here" << endl;
    exit(1);
  }

  virtual VEC3 get_color () {
    cout << "(Light::get_color) You clearly shouldn't have come here" << endl;
    exit(1);
  }
  
};

struct PointLight: public Light {
  VEC3 p, c; 
  PointLight (VEC3 p, VEC3 c) : p(p), c(c) {}

  void sample_points (vector<VEC3> &light_points) override {
    light_points.push_back(p); 
  }
  
  VEC3 get_color() override { return c; }

};

struct AreaLight: public Light {
  Rectangle r;

  AreaLight (Rectangle r) : r(r) {} 

  void sample_points (vector<VEC3> &light_points) override {
    r.rand_stratified_samples(light_points, 3); 
  }
  
  VEC3 get_color() override { return r.get_color(); }

};

Ray px_to_cam_ray (int i, int j) {
  VEC3 d = px_to_near_plane_pt(i, j); 
  return Ray(EYE, d); 
}

AABB * aabb_union (AABB *a, AABB *b) {
  VEC3 v_lo(
    min(a->v_lo(0), b->v_lo(0)), 
    min(a->v_lo(1), b->v_lo(1)), 
    min(a->v_lo(2), b->v_lo(2))
  );

  VEC3 v_hi(
    max(a->v_hi(0), b->v_hi(0)), 
    max(a->v_hi(1), b->v_hi(1)), 
    max(a->v_hi(2), b->v_hi(2))
  );

  return new AABB(v_lo, v_hi); 
}

struct Camera {
  VEC3 eye; 
  VEC3 lookingAt; 
  VEC3 up; 
  float halfY, halfX;
  VEC3 cz, cx, cy; 

  Camera () {}

  Camera (VEC3 eye, VEC3 lookingAt, VEC3 up) : eye(eye), lookingAt(lookingAt), up(up) {
    this->cz = (lookingAt - eye).normalized();
    this->cx = up.cross(cz).normalized();
    this->cy = cz.cross(cx).normalized();
    this->halfY = (lookingAt - eye).norm() * tan(45.0f / 360.0f * M_PI);
    this->halfX = this->halfY * 4.0f / 3.0f;
  } 

  void get_primary_rays_for_pixel (int x, int y, int xRes, int yRes, vector<Ray> &prs) {
    float ratioX = 1.0f - x / float(xRes) * 2.0f;
    float ratioY = 1.0f - y / float(yRes) * 2.0f;

    float ratioXp1 = 1.0f - (x + 1) / float(xRes) * 2.0f;
    float ratioYp1 = 1.0f - (y + 1) / float(yRes) * 2.0f;

    VEC3 rayHitImage = lookingAt + ratioX * halfX * cx + ratioY * halfY * cy;

    const VEC3 rayDir = (rayHitImage - eye).normalized();
    Ray primary_ray(eye, rayDir); 
    prs.push_back(primary_ray);

    // Real dx = abs((ratioX * halfX) - (ratioXp1 * halfX)); 
    // Real dy = abs((ratioY * halfY) - (ratioYp1 * halfY)); 

    // VEC3 rect_center = rayHitImage - (dx / 2.0) * cx - (dy / 2.0) * cy;
    // Rectangle rect(rect_center, cx, cy, dx, dy); 

    // vector<VEC3> points; 
    // rect.rand_stratified_samples(points, 2); 
    // for (auto &pt: points) 
    //   prs.push_back(Ray(eye, normalized(pt - eye)));
  }

  virtual void get_secondary_rays_for_pixel (vector<Ray> &rays) {
    cout << "(Camera::get_secondary_rays_for_pixel) You clearly shouldn't have come here" << endl;
    exit(1);
  }

};

struct PinHoleCamera : public Camera {
  PinHoleCamera () : Camera() {} 

  PinHoleCamera (VEC3 eye, VEC3 lookingAt, VEC3 up) : Camera(eye, lookingAt, up) {} 

  void get_secondary_rays_for_pixel(vector<Ray> &rays) override {}
}; 

struct LensCamera : public Camera {
  Real aperture, focal_length;
  Rectangle aperture_rect;

  LensCamera (VEC3 eye, VEC3 lookingAt, VEC3 up, Real aperture, Real focal_length) : Camera(eye, lookingAt, up), aperture(aperture), focal_length(focal_length) {
    VEC3 pt = eye - (aperture / 2.0) * cx - (aperture / 2.0) * cy;
    this->aperture_rect = Rectangle(pt, cx, cy, aperture, aperture); 
  } 

  void set_aperture (Real new_aperture) {
    VEC3 pt = eye - (new_aperture / 2.0) * cx - (new_aperture / 2.0) * cy;
    this->aperture = new_aperture; 
    this->aperture_rect = Rectangle(pt, cx, cy, new_aperture, new_aperture); 
  }

  void get_secondary_rays_for_pixel (vector<Ray> &primary_rays) override {
    for (auto &pr : primary_rays) { 
      VEC3 c = pr.point_at_time(focal_length); 
      vector<VEC3> points; 
      aperture_rect.rand_stratified_samples(points, 4); 
      for (auto &pt : points) 
        primary_rays.push_back(Ray(pt, normalized(c - pt))); 
    }
  }
};

struct OrthographicCamera : public Camera {
  Rectangle rect; // the rectangle from which we are shooting rays

  OrthographicCamera(Rectangle rect) : rect(rect) {}

  void get_primary_rays_for_pixel (int x, int y, int xRes, int yRes, vector<Ray> &prs) {
    double dx = rect.s1 / xRes;
    double dy = rect.s2 / yRes;
    VEC3 pt = rect.o 
              + (rect.d1 * (((x + 0.0) / (xRes + 0.0) * rect.s1) + dx / 2.))
              + (rect.d2 * (((y + 0.0) / (yRes + 0.0) * rect.s2) + dy / 2.));
    Ray r(pt, rect.normal(VEC3(0,0,0))); 
    prs.push_back(r);
  }

}; 

Cylinder * make_cylinder_from_endpoints (VEC3 a, VEC3 b, Real radius, VEC3 color=VEC3(1.0, 0.0, 0.0)) { 
  VEC3 c = (a + b) * 0.5;
  VEC3 d = (b - a);
  Real l = d.norm();
  d *= 1.0 / l;
  return new Cylinder(c, d, radius, l, color); 
}

AABB * make_aabb_from_center_and_dims(VEC3 c, Real lx, Real ly, Real lz, VEC3 color=VEC3(1.0, 0.0, 0.0)) {
  VEC3 v_lo (c(0) - (lx / 2.0), c(1) - (ly / 2.0), c(2) - (lz / 2.0)); 
  VEC3 v_hi (c(0) + (lx / 2.0), c(1) + (ly / 2.0), c(2) + (lz / 2.0)); 
  return new AABB(v_lo, v_hi, color); 
}

#endif
