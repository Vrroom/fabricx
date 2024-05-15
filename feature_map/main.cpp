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

int WINDOW_WIDTH = 1024;
int WINDOW_HEIGHT = 1024;
int NUM_PROFILE_POINTS=200; 
int NUM_SWEEP_POINTS=200;
int N_TILE = 1;

double R = 10; 
double PHI = M_PI / 3;
double EPSILON = 1e-3;

typedef VEC3 Color;

vector<Color> COLORS;

typedef pair<int, int> P; 

struct Filament2D {
  Rectangle rect; 
  int type; 
  Filament2D(Rectangle &rect, int type) : rect(rect), type(type) {} 
}; 

typedef vector<Filament2D> FilamentMap; 

FilamentMap readFilamentMap(const string& filePath) {
  /** 
   * Filament map is a special .fil file. It specifies a series of
   * rectangles and a type. Together, a rectangle and a type specify a 
   * thread filament.
   *
   * The type is simply a natural number with even type represeting horizontal
   * weft threads and odd type specifying vertical warp threads.
   *
   * Here is a few lines from satin.fil
   *
   *   0.0 0.0 0.20 0.10 0
   *   0 0.1 0.20 0.90 1
   *   0.2 0.0 0.20 0.40 1
   *   0.2 0.4 0.20 0.1 0
   *
   * The rectangle coordinates are such that the filament map sits in a [0, 1] by [0, 1]
   * rectangle. 
   *
   * Finally, there may be a few lines at the end of the form
   *   
   *   R G B type
   *
   * These are used to specify the colors of the threads
   */ 
  FilamentMap fil_map;
  ifstream file(filePath);

  if (!file) {
    cerr << "Error opening file: " << filePath << endl;
    return fil_map;
  }

  string line;
  map<int, Color> type_to_color;
  while (getline(file, line)) { 
    if (line.empty() || line[0] == '#') {
      continue;
    }

    {
      stringstream ss(line);
      double x, y, width, height;
      int type;
      if (ss >> x >> y >> width >> height >> type) {
        Rectangle r(
          VEC3(x - EPSILON, y - EPSILON, 0.0), 
          VEC3(1.0, 0.0, 0.0), 
          VEC3(0.0, 1.0, 0.0),
          width + 2 * EPSILON,
          height + 2 * EPSILON
        );
        fil_map.push_back(Filament2D(r, type));
        continue;
      }
    }

    {
      stringstream ss(line);
      int R, G, B, type;
      if (ss >> R >> G >> B >> type) {
        type_to_color[type] = Color(R,G,B) / 255.;
      }
    }
  }

  for (int type = 0; type < type_to_color.size(); type++) {
    COLORS.push_back(type_to_color[type]); 
  }

  file.close();
  // now tile it
  FilamentMap final_map; 
  
  // naive tiling
  for (int i = 0; i < N_TILE; i++) 
    for (int j = 0; j < N_TILE; j++) {
      VEC3 new_o((i + 0.0), (j + 0.0), 0.0); 
      for (int k = 0; k < fil_map.size(); k++) {
        Filament2D fil = fil_map[k];
        fil.rect.o += new_o;
        final_map.push_back(fil);
      }
    }

  // scale everything down
  for (int i = 0; i < final_map.size(); i++) {
    final_map[i].rect.o /= N_TILE;
    final_map[i].rect.s1 /= N_TILE;
    final_map[i].rect.s2 /= N_TILE;
  }
  return final_map;
}

enum FeatureMapType {
  ID_MAP,
  NORMAL_MAP, 
  TANGENT_MAP, 
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

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
void rayColor(Ray &ray, VEC3& pixelColor) {
  pixelColor = VEC3(0.0, 0.0, 0.0);
  // look for intersection with scene
  Real tMinFound = INF;
  int pId = scene.ray_scene_intersect_idx(ray, tMinFound); 
  if (pId >= 0) {
    Primitive *prim = scene.scene[pId];
    pixelColor = prim->get_color(); 
  }
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
  Image im(xRes, yRes);
  // allocate the final image
  const int totalCells = xRes * yRes;

  // compute image plane
  #pragma omp parallel for collapse(2)
  for (int y = 0; y < yRes; y++) { 
    for (int x = 0; x < xRes; x++) 
    {
      // get the color
      VEC3 color(0.0, 0.0, 0.0);
      vector<Ray> rays; 
      cam.get_primary_rays_for_pixel(x, y, xRes, yRes, rays); 
      for (auto r: rays) {
        VEC3 rColor;
        rayColor(r, rColor);
        color += rColor; 
      }
      color = color * (1.0 / ((Real) rays.size())); 

      // set, in final image
      im.set_color(x, y, color); 
    }
  }
  // detect whether there are some degenerate pixels 
  if (map_type == NORMAL_MAP || map_type == TANGENT_MAP) { 
    set<P> degenerate; 
    for (int y = 0; y < yRes; y++) {
      for (int x = 0; x < xRes; x++) {
        VEC3 color = im.get_color(x, y); 
        VEC3 vec = color * 2.0 - VEC3(1.0, 1.0, 1.0);
        if (abs(norm(vec) - 1.0) > 1e-2) 
          degenerate.insert(make_pair(x, y));
      }
    }
    vector<P> nbrs = { {-1, 0}, {0, 1}, {1, 0}, {0, -1} };
    while(!degenerate.empty()) { 
      auto it = degenerate.begin(); 
      for (; it != degenerate.end(); ++it) {
        int x = it->first, y = it->second; 
        bool found = false;
        for (auto nbr: nbrs) { 
          int x_ = x + nbr.first, y_ = y + nbr.second; 
          if (x_ < xRes && x_ >= 0 && y_ < yRes && y_ >= 0) {
            VEC3 color = im.get_color(x_, y_); 
            VEC3 vec = color * 2.0 - VEC3(1.0, 1.0, 1.0);
            if (abs(norm(vec) - 1.0) <= 1e-2) {
              im.set_color(x, y, color);
              found = true;
            }
          }
        }
        if (found) {
          degenerate.erase(it); 
          break;
        }
      }
    }
  }
  // Fix them.
  writePPM(filename, im);

}

void placeFilament (Filament2D &fil, FeatureMapType map_type) { 
  vector<VEC3> vertices, tangents, normals; 
  if (fil.type == 0) {
    // horizontal segment such as the one we are building now
    double w = fil.rect.s1;
    double a = fil.rect.s2 / 2;
    double tan_theta = w / (2.0 * R); 
    double u_max = atan(tan_theta); 
    for (int i = 0; i < NUM_PROFILE_POINTS; i++) {
      double u = -u_max + (2.0 * u_max) * (i + 0.0) / (NUM_PROFILE_POINTS + 0.0); 
      VEC3 x_0(fil.rect.o[1] + a, fil.rect.o[0] + w/2 + R * sin(u), R * cos(u) - R);
      for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
        double v = -M_PI + (2.0 * M_PI) * (j + 0.0) / (NUM_SWEEP_POINTS + 0.0);
        VEC3 n(
          sin(v), 
          sin(u) * cos(v), 
          cos(u) * cos(v)
        );
        VEC3 t(
          -cos(v) * sin(PHI), 
          cos(u) * cos(PHI) + sin(u) * sin(v) * sin(PHI), 
          -sin(u) * cos(PHI) + cos(u) * sin(v) * sin(PHI)
        );
        vertices.push_back(x_0 + a * n); 
        tangents.push_back(VEC3(t[1], t[0], -t[2])); 
        normals.push_back(VEC3(n[1], -n[0], n[2])); // orient it like Jin et al.
      }
    }
  } else {
    double w = fil.rect.s2;
    double a = fil.rect.s1 / 2;
    double tan_theta = w / (2.0 * R); 
    double u_max = atan(tan_theta); 
    for (int i = 0; i < NUM_PROFILE_POINTS; i++) {
      double u = -u_max + (2.0 * u_max) * (i + 0.0) / (NUM_PROFILE_POINTS + 0.0); 
      VEC3 x_0(R * sin(u) + w/2 + fil.rect.o[1], a + fil.rect.o[0], R * cos(u) - R);
      for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
        double v = -M_PI + (2.0 * M_PI) * (j + 0.0) / (NUM_SWEEP_POINTS + 0.0);
        VEC3 n(
          sin(u) * cos(v), 
          sin(v), 
          cos(u) * cos(v));
        VEC3 t(
          cos(u) * cos(PHI) + sin(u) * sin(v) * sin(PHI), 
          -cos(v) * sin(PHI), 
          -sin(u) * cos(PHI) + cos(u) * sin(v) * sin(PHI)
        );
        vertices.push_back(x_0 + a * n); 
        tangents.push_back(VEC3(t[1], t[0], -t[2])); 
        normals.push_back(VEC3(n[1], -n[0], n[2])); // orient it like Jin et al.
      }
    }
  }
  for (int i = 0; i < NUM_PROFILE_POINTS - 1; i++) {
    for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
      int l = i * NUM_SWEEP_POINTS + j; 
      int m = i * NUM_SWEEP_POINTS + (j + 1) % NUM_SWEEP_POINTS; 
      int n = (i + 1) * NUM_SWEEP_POINTS + (j + 1) % NUM_SWEEP_POINTS; 
      int p = (i + 1) * NUM_SWEEP_POINTS + j;
      auto c1 = getColor(l, m, n, normals, tangents, fil.type, map_type); 
      auto c2 = getColor(n, p, l, normals, tangents, fil.type, map_type); 
      scene.add_primitive(
        new Triangle(
          vertices[l],
          vertices[m],
          vertices[n],
          c1
        )
      );
      scene.add_primitive(
        new Triangle(
          vertices[n],
          vertices[p],
          vertices[l],
          c2
        )
      );
    }
  }
}

void build (FilamentMap &fil_map, FeatureMapType map_type) {
  scene.clear();
  for (auto &f: fil_map) 
    placeFilament(f, map_type); 
  scene.build();
}

void renderMapType (FilamentMap &fil_map, FeatureMapType map_type, string fileName) {
  build(fil_map, map_type);
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
    ("n,num-tiles", "Number of tiles to make using the Filament map", cxxopts::value<int>()->default_value(std::to_string(N_TILE)))
    ("f,file-path", "File path for input/output", cxxopts::value<string>(file_path))
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
  N_TILE = result["num-tiles"].as<int>();
  file_path = result["file-path"].as<string>(); 

  FilamentMap fil_map = readFilamentMap(file_path); 
  vector<FeatureMapType> types = { NORMAL_MAP, TANGENT_MAP, ID_MAP }; 
  vector<string> names = { "normal_map.ppm", "tangent_map.ppm", "id_map.ppm" }; 
  for (int i = 0; i < types.size(); i++) {
    renderMapType(fil_map, types[i], names[i]);
  }
  return 0;
}
