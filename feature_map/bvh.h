#ifndef BVH_H
#define BVH_H

#include "tools.h"
#include "primitive.h" 

struct LBVH {

  vector<Primitive *> scene; 
  vector<AABB *> T; 
  
  void add_primitive (Primitive * p) {
    scene.push_back(p); 
  }

  void build () {
    int N = scene.size(); 

    // get primitive centroids
    vector<VEC3> centroids; 
    for (int i = 0; i < N; i++) {
      AABB * aabb = scene[i]->to_aabb(); 
      centroids.push_back(aabb->centroid());
      delete aabb;
    }
    // make sure they are positive and in [0, 1) 
    Real mx = INF, Mx = -INF, my = INF, My = -INF, mz = INF, Mz = -INF;
    for (int i = 0; i < N; i++) {
      mx = min(mx, centroids[i](0)); 
      Mx = max(Mx, centroids[i](0)); 

      my = min(my, centroids[i](1)); 
      My = max(My, centroids[i](1)); 

      mz = min(mz, centroids[i](2)); 
      Mz = max(Mz, centroids[i](2)); 
    }

    Real trans = min3(mx, my, mz); 
    VEC3 translation(trans, trans, trans); 
    VEC3 delta_perturb(1e-6, 1e-6, 1e-6);
    Real scale = max3(Mx - trans, My - trans, Mz - trans); 

    for (int i = 0; i < N; i++) {
      centroids[i] = centroids[i] - translation + delta_perturb; 
      centroids[i] = centroids[i] / (scale + 1e-5); 
    }

    // sort by morton code
    vector<int> sorted_ids; 
    sort_by_morton(centroids, sorted_ids); 
    vector<Primitive *> sorted_scene; 
    for (int i = 0; i < N; i++) 
      sorted_scene.push_back(scene[sorted_ids[i]]); 
    scene = sorted_scene;

    // now build the tree
    T.resize(4 * N + 10); 
    _build(1, 0, N - 1); 
  }

  void _build (int node, int l, int h) {
    if (l > h) return;
    if (l == h) {
      T[node] = scene[l]->to_aabb(); 
      T[node]->dilate();
      return;
    }

    int m = (l + h) / 2;

    _build(2 * node    , l    , m); 
    _build(2 * node + 1, m + 1, h); 

    T[node] = aabb_union(T[2 * node], T[2 * node + 1]);
    T[node]->dilate();
  }

  Primitive * ray_scene_intersect (Ray &r, Real &min_t) { 
    int N = scene.size(); 
    min_t = INF;
    return _ray_scene_intersect(1, 0, N - 1, r, min_t); 
  }

  int ray_scene_intersect_idx (Ray &r, Real &min_t) { 
    int N = scene.size(); 
    min_t = INF;
    return _ray_scene_intersect_idx(1, 0, N - 1, r, min_t); 
  }

  int ray_scene_intersect_idx_slow (Ray &r, Real &min_t) { 
    int N = scene.size(); 
    min_t = INF;
    int id = -1;
    for (int i = 0; i < N; i++) {
      Real check_t;
      if (_intersect_wrapper(scene[i], r, check_t) != NULL)  {
        if (check_t < min_t) {
          min_t = check_t;
          id = i;
        }
      }
    }
    return id;
  }

  Primitive * _intersect_wrapper (Primitive *p, Ray &r, Real &min_t) {
    vector<Real> ts; 
    p->intersects(r, ts); 
    int tId = argmin_thresh(ts, 0.0); 
    if (tId >= 0) {
      min_t = ts[tId]; 
      return p;
    }
    return NULL;
  }

  Primitive * _ray_scene_intersect(int node, int l, int h, Ray &r, Real &min_t) {
    if (l > h) 
      return NULL;
    Real check_t;
    if (_intersect_wrapper(T[node], r, check_t) == NULL) 
      return NULL; 
    if (l == h) 
      return _intersect_wrapper(scene[l], r, min_t); 

    int m = (l + h) / 2;
    Real t1 = INF, t2 = INF; 
    Primitive *p1 = _ray_scene_intersect(2 * node    , l    , m, r, t1); 
    Primitive *p2 = _ray_scene_intersect(2 * node + 1, m + 1, h, r, t2); 
    
    if (p1 == NULL && p2 == NULL) 
      return NULL;
    else if (p1 == NULL) {
      min_t = t2;
      return p2;
    } else if (p2 == NULL) {
      min_t = t1;
      return p1;
    } else {
      min_t = min(t1, t2);
      if (t1 < t2) return p1;
      else return p2;
    }
  }

  int _ray_scene_intersect_idx(int node, int l, int h, Ray &r, Real &min_t) {
    if (l > h) 
      return -1;
    Real check_t;
    if (_intersect_wrapper(T[node], r, check_t) == NULL) 
      return -1; 
    if (l == h) 
      return (_intersect_wrapper(scene[l], r, min_t) == NULL) ? -1 : l;

    int m = (l + h) / 2;
    Real t1 = INF, t2 = INF; 
    int p1 = _ray_scene_intersect_idx(2 * node    , l    , m, r, t1); 
    int p2 = _ray_scene_intersect_idx(2 * node + 1, m + 1, h, r, t2); 
    
    if (p1 == -1 && p2 == -1) 
      return -1;
    else if (p1 == -1) {
      min_t = t2;
      return p2;
    } else if (p2 == -1) {
      min_t = t1;
      return p1;
    } else {
      min_t = min(t1, t2);
      if (t1 < t2) return p1;
      else return p2;
    }
  }

  void clear () { 
    for (int i = 0; i < scene.size(); i++)
      delete scene[i];
    for (int i = 0; i < T.size(); i++) 
      delete T[i];
    scene.clear(); 
    T.clear();
  }


}; 

#endif
