#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <float.h>
#include "SETTINGS.h"
#include "imageTools.h"
#include "primitive.h"
#include "bvh.h"

using namespace std;

int windowWidth = 1024;
int windowHeight = 1024;
int profileNumPoints=200; 
int sweepNumPoints=200;

double R = 10; 
double PHI = M_PI / 3;
double EPSILON = 1e-3;

struct Filament2D {
  Rectangle rect; 
  int type; 
  Filament2D(Rectangle &rect, int type) : rect(rect), type(type) {} 
}; 

typedef vector<Filament2D> FilamentMap; 

FilamentMap readFilamentMap(const string& filePath) {
  FilamentMap fil_map;
  ifstream file(filePath);

  if (!file) {
    cerr << "Error opening file: " << filePath << endl;
    return fil_map;
  }

  string line;
  while (getline(file, line)) { 
    if (line.empty() || line[0] == '#') {
      continue;
    }

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
    } else {
      std::cerr << "Error parsing line: " << line << std::endl;
    }
  }

  file.close();
  return fil_map;
}

enum FeatureMapType {
  ID_MAP,
  NORMAL_MAP, 
  ORIENTATION_MAP, 
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
    if (type == 0) 
      return VEC3(1.0, 0.0, 0.0); 
    else 
      return VEC3(0.0, 0.0, 1.0); 
  } else if (map_type == NORMAL_MAP) { 
    VEC3 pt = normalized((normals[i] + normals[j] + normals[k]) / 3.0);
    return (pt + VEC3(1.0, 1.0, 1.0)) / 2.0;
  } else if (map_type == ORIENTATION_MAP) {
    VEC3 pt = normalized((tangents[i] + tangents[j] + tangents[k]) / 3.0);
    return (pt + VEC3(1.0, 1.0, 1.0)) / 2.0;
  } else {
    return VEC3(0.0, 0.0, 1.0);
  }
}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
void renderImage(int& xRes, int& yRes, const string& filename) 
{
  Image im(xRes, yRes);
  // allocate the final image
  const int totalCells = xRes * yRes;

  // compute image plane
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
    for (int i = 0; i < profileNumPoints; i++) {
      double u = -u_max + (2.0 * u_max) * (i + 0.0) / (profileNumPoints + 0.0); 
      VEC3 x_0(fil.rect.o[1] + w/2, fil.rect.o[0] + a + R * sin(u), R * cos(u) - R);
      for (int j = 0; j < sweepNumPoints; j++) {
        double v = -M_PI + (2.0 * M_PI) * (j + 0.0) / (sweepNumPoints + 0.0);
        VEC3 n(
          sin(v), 
          sin(u) * cos(v), 
          cos(u) * cos(v));
        VEC3 t(
          -cos(v) * sin(PHI), 
          cos(u) * cos(PHI) + sin(u) * sin(v) * sin(PHI), 
          -sin(u) * cos(PHI) + cos(u) * sin(v) * sin(PHI)
        );
        vertices.push_back(x_0 + a * n); 
        tangents.push_back(t); 
        normals.push_back(n);
      }
    }
  } else {
    double w = fil.rect.s2;
    double a = fil.rect.s1 / 2;
    double tan_theta = w / (2.0 * R); 
    double u_max = atan(tan_theta); 
    for (int i = 0; i < profileNumPoints; i++) {
      double u = -u_max + (2.0 * u_max) * (i + 0.0) / (profileNumPoints + 0.0); 
      VEC3 x_0(R * sin(u) + w/2 + fil.rect.o[1], a + fil.rect.o[0], R * cos(u) - R);
      for (int j = 0; j < sweepNumPoints; j++) {
        double v = -M_PI + (2.0 * M_PI) * (j + 0.0) / (sweepNumPoints + 0.0);
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
        tangents.push_back(t); 
        normals.push_back(n);
      }
    }
  }
  for (int i = 0; i < profileNumPoints - 1; i++) {
    for (int j = 0; j < sweepNumPoints; j++) {
      int l = i * sweepNumPoints + j; 
      int m = i * sweepNumPoints + (j + 1) % sweepNumPoints; 
      int n = (i + 1) * sweepNumPoints + (j + 1) % sweepNumPoints; 
      int p = (i + 1) * sweepNumPoints + j;
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
  // buildIrawanCircular(M_PI / 4.0, 0.2, map_type);
  build(fil_map, map_type);
  renderImage(windowWidth, windowHeight, fileName);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <Filament Map file>" << std::endl;
    return 1;
  }
  std::string input(argv[1]);
  FilamentMap fil_map = readFilamentMap(input); 
  vector<FeatureMapType> types = { NORMAL_MAP, ORIENTATION_MAP, ID_MAP }; 
  vector<string> names = { "normal_map.ppm", "orientation_map.ppm", "id_map.ppm" }; 
  for (int i = 0; i < types.size(); i++) {
    renderMapType(fil_map, types[i], names[i]);
  }
  return 0;
}
