#include <cstdio>
#include <map>
#include <omp.h>
#include <set>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <float.h>
#include "cxxopts.hpp"
#include "SETTINGS.h"
#include "imageTools.h"
#include "primitive.h"
#include "bvh.h"

using namespace std;

int WINDOW_WIDTH = 128;
int WINDOW_HEIGHT = 128;
int NUM_PROFILE_POINTS=200; 
int NUM_SWEEP_POINTS=200;
int N_ANGLE = 10;

double R = 10; 
double PHI = M_PI / 3;
double EPSILON = 1e-3;

bool DELTA_TRANSMISSION = false;

typedef VEC3 Color;

vector<Color> COLORS;

typedef pair<int, int> P; 

enum FeatureMapType {
  ID_MAP,
  POSITION_MAP,
  NORMAL_MAP, 
  TANGENT_MAP, 
  BENT_NORMAL_MAP,
  INVALID_MAP
};

OrthographicCamera cam(
  Rectangle (
    VEC3(0.0, 0.0, 2.0), 
    VEC3(0.0, 1.0, 0.0), 
    VEC3(1.0, 0.0, 0.0), 
    1.0, 
    1.0
  )
); 

// scene geometry
LBVH scene;

struct SurfaceInteraction {
  Primitive *prim; 
  Real t;
  SurfaceInteraction (Primitive *prim, Real t) 
    : prim(prim), t(t) {}
};

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
SurfaceInteraction * rayColor(Ray &ray, VEC3& pixelColor, bool &intersected) {
  pixelColor = VEC3(0.0, 0.0, 0.0);
  // look for intersection with scene
  Real tMinFound = INF;
  int pId = scene.ray_scene_intersect_idx(ray, tMinFound); 
  if (pId >= 0) {
    Primitive *prim = scene.scene[pId];
    pixelColor = prim->get_color(); 
    return new SurfaceInteraction(prim, tMinFound);
  } else {
    intersected = false;
  }
  return NULL;
}

VEC3 getColor (int i, int j, int k, vector<VEC3> &normals, vector<VEC3> &tangents, int type, FeatureMapType map_type) { 
  if (map_type == ID_MAP) { 
    if (type >= COLORS.size()) {
      return (type % 2 == 0) ? VEC3(1, 0, 0) : VEC3(0, 0, 0); 
    }
    return COLORS[type];
  } else if (map_type == NORMAL_MAP) { 
    VEC3 pt = normalized((normals[i] + normals[j] + normals[k]) / 3.0);
    return (pt + VEC3(1.0, 1.0, 1.0)) / 2.0;
  } else if (map_type == TANGENT_MAP) {
    VEC3 pt = normalized((tangents[i] + tangents[j] + tangents[k]) / 3.0);
    return (pt + VEC3(1.0, 1.0, 1.0)) / 2.0;
  } else {
    return VEC3(0.0, 0.0, 1.0);
  }
}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
void renderImage(int& xRes, int& yRes, const string& filename, FeatureMapType &map_type) 
{
  int xRes_eq = xRes;
  int yRes_eq = yRes;
  if (map_type == POSITION_MAP)
  {
    xRes_eq += 1;
    yRes_eq += 1;
  }

  VEC4** colorArray = new VEC4*[xRes_eq];
  for (int i = 0; i < xRes_eq; i++)
  {
    colorArray[i] = new VEC4[yRes_eq];
  }

  Image im(xRes_eq, yRes_eq);
  // allocate the final image
  const int totalCells = xRes_eq * yRes_eq;

  ofstream out("data.txt");
  for (int y = 0; y < yRes_eq; y++) { 
    for (int x = 0; x < xRes_eq; x++) 
    {
      // get the color
      VEC4 color(0.0, 0.0, 0.0, 0.0);
      if (map_type == POSITION_MAP)
      {
        Ray r = cam.get_ray_for_pixel_corner(x, y, xRes_eq, yRes_eq);
        VEC3 rColor;
        bool intersected = true;
        SurfaceInteraction * si = rayColor(r, rColor, intersected);
        if (intersected || !DELTA_TRANSMISSION) color = to_hom(rColor);
      }
      else
      {
        vector<Ray> rays; 
        cam.get_primary_rays_for_pixel(x, y, xRes_eq, yRes_eq, rays); 
        for (auto r: rays) {
          VEC3 rColor;
          bool intersected = true;
          SurfaceInteraction * si = rayColor(r, rColor, intersected);
          if (intersected || !DELTA_TRANSMISSION) {
            if (map_type == BENT_NORMAL_MAP) {
              /**
               * Bent normal is computed by integrating wi V <n, wi> over the upper 
               * hemisphere. This is from: 
               *  
               *  "Practical Real-Time Strategies for Accurate Indirect Occlusion"
               */
              VEC3 pt = r.point_at_time(si->t);
              VEC3 n  = si->prim->normal(VEC3(0.0, 0.0, 0.0));
              VEC3 bent_normal(0.0, 0.0, 0.0); 
              int total_sampled = 0;
              for (int phi_i = 0; phi_i <= N_ANGLE; phi_i++) {
                for (int theta_i = 0; theta_i < N_ANGLE; theta_i++) {
                  double phi = phi_i * (M_PI / 2) * (1.0 / 10); // [0, pi/2] upper hemisphere
                  double theta = theta_i * 2.0 * M_PI * (1.0 / 10);  // [0, 2pi)
                  VEC3 d(
                    sin(phi) * cos(theta),
                    sin(phi) * sin(theta), 
                    cos(phi)
                  );
                  Ray sr(pt, d, 1, n); 
                  Real tMinFound = INF;
                  int pId = scene.ray_scene_intersect_idx(sr, tMinFound); 
                  double V = pId < 0 ? 1.0: 0.0; // if nothing intersecting then visible, else not
                  bent_normal += (d * V * max(d.dot(n), 0.0)); // \int wi V <n, wi> dwi
                  total_sampled++; 
                  out << y << " " << x << " " << phi << " " << theta << " " << V << endl;
                }
              } 
              color += to_hom((normalized(bent_normal) + VEC3(1.0, 1.0, 1.0)) / 2); 
            } else {
              color += to_hom(rColor); 
            }
          }
          if (si != NULL) 
            delete si;
        }
        color = color * (1.0 / ((Real) rays.size()));
      } 

      // set, in final image
      im.set_color_4(y, x, color);
      colorArray[x][y] = color;
    }
  }
  // detect whether there are some degenerate pixels 
  // Fix them.

  string filename_txt = filename + ".txt";
  ofstream txtFile;
  txtFile.open(filename_txt);
  for (int i = 0; i < xRes_eq; i++)
  {
    for (int j = 0; j < yRes_eq; j++)
    {
      txtFile << colorArray[i][j][0] << " "
              << colorArray[i][j][1] << " "
              << colorArray[i][j][2] << " "
              << colorArray[i][j][3] << ",";
    }
    txtFile << "\n";
  }
  txtFile.close();
  delete[] colorArray;

  string filename_png = filename + ".png";
  writePNG(filename_png, im);
}

void build (FeatureMapType map_type) {
  scene.clear();
  vector<VEC3> vertices, tangents, normals; 
  vector<VEC3> profile; 

  double x = 1.0 / NUM_PROFILE_POINTS; 
  double h = tan(M_PI / 3) * x * 0.5;

  for (int i = 0; i < NUM_PROFILE_POINTS / 2; i++) {
    profile.push_back(VEC3(2 * i * x    , 0, 0)); 
    profile.push_back(VEC3(2 * i * x + x, 0, 0)); 
    profile.push_back(VEC3(2 * i * x + x + x/2, 0, h)); 
  }
  profile.push_back(VEC3(2 *(NUM_PROFILE_POINTS / 2) * x, 0, 0)); 

  double y = 1.0 / NUM_SWEEP_POINTS; 

  for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
    for (int i = 0; i < profile.size() - 1; i++) { 
      double y1 = j * y, y2 = j * y + y; 
      Triangle * a = new Triangle(
        VEC3(profile[i][0], y2, profile[i][2]),
        VEC3(profile[i][0], y1, profile[i][2]),
        VEC3(profile[i + 1][0], y1, profile[i + 1][2]),
        VEC3(1, 0, 0)
      );
      Triangle * b = new Triangle(
        VEC3(profile[i][0], y2, profile[i][2]),
        VEC3(profile[i + 1][0], y1, profile[i + 1][2]),
        VEC3(profile[i + 1][0], y2, profile[i + 1][2]),
        VEC3(1, 1, 0)
      );
      if (map_type == POSITION_MAP) {
        // average z coordinate, works well if the triangle is sufficiently small
        // TODO: currently the z coordinates are too small and seem to round to 0
        a->color = VEC3(0.0, 0.0, (a->a[2] + a->b[2] + a->c[2]) / 3.0);
        b->color = VEC3(0.0, 0.0, (b->a[2] + b->b[2] + b->c[2]) / 3.0);
      } else if (map_type == NORMAL_MAP) { 
        a->color = a->normal(VEC3(0.0, 0.0, 0.0));
        a->color = (a->color + VEC3(1.0, 1.0, 1.0)) / 2.0;

        b->color = b->normal(VEC3(0.0, 0.0, 0.0));
        b->color = (b->color + VEC3(1.0, 1.0, 1.0)) / 2.0;
      } else if (map_type == TANGENT_MAP) { 
        a->color = b->color = VEC3(0.5, 1.0, 0.5);
      } else {
        if (abs(profile[i][2] - profile[i + 1][2]) < 1e-5) 
          a->color = b->color = VEC3(0, 1, 0); 
        else 
          a->color = b->color = VEC3(1, 1, 0); 
      }
      scene.add_primitive(a);
      scene.add_primitive(b);
    }
  }
  scene.build();
}

void renderMapType (FeatureMapType map_type, string fileName) {
  build(map_type);
  renderImage(WINDOW_WIDTH, WINDOW_HEIGHT, fileName, map_type);
}

int main(int argc, char* argv[]) {
  cxxopts::Options options(argv[0], " - Command line options");
  string file_path;
  options
    .add_options()
    ("w,window-width", "Image width", cxxopts::value<int>()->default_value(std::to_string(WINDOW_WIDTH)))
    ("h,window-height", "Image height", cxxopts::value<int>()->default_value(std::to_string(WINDOW_HEIGHT)))
    ("p,num-profile-points", "Number of points on the profile curve", cxxopts::value<int>()->default_value(std::to_string(NUM_PROFILE_POINTS)))
    ("s,num-sweep-points", "Number of points on sweep curve", cxxopts::value<int>()->default_value(std::to_string(NUM_SWEEP_POINTS)))
    ("r,radius", "Radius of curvature", cxxopts::value<double>()->default_value(std::to_string(R)))
    ("phi,twisting-angle", "Twisting angle in radians", cxxopts::value<double>()->default_value(std::to_string(PHI)))
    ("d,delta-transmission", "Delta transmission", cxxopts::value<bool>()->default_value(std::to_string(DELTA_TRANSMISSION)))
    ("help", "Print help");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
  }

  WINDOW_WIDTH = result["window-width"].as<int>();
  WINDOW_HEIGHT = result["window-height"].as<int>();
  NUM_PROFILE_POINTS = result["num-profile-points"].as<int>();
  NUM_SWEEP_POINTS = result["num-sweep-points"].as<int>();
  R = result["radius"].as<double>();
  PHI = result["twisting-angle"].as<double>();
  DELTA_TRANSMISSION = result["delta-transmission"].as<bool>();

  vector<FeatureMapType> types = { POSITION_MAP, NORMAL_MAP, TANGENT_MAP, ID_MAP, BENT_NORMAL_MAP }; 
  vector<string> names = { "position_map", "normal_map", "tangent_map", "id_map", "bent_normal_map" }; 
  for (int i = 0; i < types.size(); i++) {
    renderMapType(types[i], names[i]);
  }
  return 0;
}
