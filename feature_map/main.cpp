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
#include "PerlinNoise.hpp"

using namespace std;

int WINDOW_WIDTH = 128;
int WINDOW_HEIGHT = 128;
int NUM_PROFILE_POINTS=200; 
int NUM_SWEEP_POINTS=200;
int N_TILE = 1;
int N_ANGLE = 10;

double R = 10; 
double PHI = M_PI / 3;
double EPSILON = 1e-3;

bool DELTA_TRANSMISSION = false;
bool ADD_NOISE = false;

const siv::PerlinNoise::seed_type seed = 123456u;

const siv::PerlinNoise perlin{ seed };

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
  } 
  intersected = false;
  return NULL;
}

VEC3 getColor (int i, int j, int k, Triangle *t, VEC3 tangent_dir, int type, FeatureMapType map_type) { 
  if (map_type == ID_MAP) { 
    if (type >= COLORS.size()) {
      return (type % 2 == 0) ? VEC3(1, 0, 0) : VEC3(0, 0, 0); 
    }
    return COLORS[type];
  } else if (map_type == POSITION_MAP) {
    return VEC3(0.0, 0.0, (t->a[2] + t->b[2] + t->c[2]) / 3.0); // average z coordinate, works well if the triangle is sufficiently small
  } else if (map_type == NORMAL_MAP) { 
    VEC3 pt = t->normal(VEC3(0,0,0));
    pt = pt * sign(pt.dot(VEC3(0, 0, 1))); 
    return (pt + VEC3(1.0, 1.0, 1.0)) / 2.0;
  } else if (map_type == TANGENT_MAP) {
    VEC3 pt = tangent_dir;
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
  // compute image plane
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
        if (intersected || !DELTA_TRANSMISSION)
          color = to_hom(rColor);
      }
      else
      {
        vector<Ray> rays; 
        cam.get_primary_rays_for_pixel(x, y, xRes_eq, yRes_eq, rays); 
        for (auto r: rays) {
          VEC3 rColor;
          bool intersected = true;
          SurfaceInteraction * si = rayColor(r, rColor, intersected);
          if (intersected || !DELTA_TRANSMISSION)
          {
            if ((map_type == BENT_NORMAL_MAP) && intersected)
            {
              /**
               * Bent normal is computed by integrating wi V <n, wi> over the upper 
               * hemisphere. This is from: 
               *  
               *  "Practical Real-Time Strategies for Accurate Indirect Occlusion"
               */
              VEC3 pt = r.point_at_time(si->t);
              VEC3 n  = si->prim->normal(VEC3(0.0, 0.0, 0.0)); 
              n = n * sign(n(2)); // ensure normal faces +z
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
            }
            else color += to_hom(rColor); 
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
    if (map_type == BENT_NORMAL_MAP) cout << "bent normal map processed " << y+1 << " rows" << endl;
  }
  // detect whether there are some degenerate pixels 
  // TODO: skipped for now since it seems to produce artifacts
  /*
  if ((map_type == NORMAL_MAP || map_type == TANGENT_MAP) && !DELTA_TRANSMISSION) { 
    set<P> degenerate; 
    for (int y = 0; y < yRes_eq; y++) {
      for (int x = 0; x < xRes_eq; x++) {
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
          if (x_ < xRes_eq && x_ >= 0 && y_ < yRes_eq && y_ >= 0) {
            VEC3 color = im.get_color(x_, y_); 
            VEC3 vec = color * 2.0 - VEC3(1.0, 1.0, 1.0);
            if (abs(norm(vec) - 1.0) <= 1e-2) {
              im.set_color(y, x, color);
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
  */
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

void placeFilament (Filament2D &fil, FeatureMapType map_type) { 
  vector<VEC3> vertices; 
  if ((fil.type % 2) == 1) {
    double w = fil.rect.s1;
    double a = fil.rect.s2 / 2;
    double tan_theta = w / (2.0 * R); 
    double u_max = atan(tan_theta); 
    for (int i = 0; i <= NUM_PROFILE_POINTS; i++) {
      double u = -u_max + (2.0 * u_max) * (i + 0.0) / (NUM_PROFILE_POINTS + 0.0); 
      VEC3 x_0(fil.rect.o[1] + a, fil.rect.o[0] + w/2 + R * sin(u), R * cos(u) - R);
      for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
        double v = -M_PI + (2.0 * M_PI) * (j + 0.0) / (NUM_SWEEP_POINTS + 0.0);
        VEC3 n(sin(v), sin(u) * cos(v), cos(u) * cos(v));
        VEC3 pt = x_0 + a * n; 
        VEC3 noise_term = VEC3(0,0,0); 
        if (ADD_NOISE) 
          noise_term = perlin.noise2D(pt[0], pt[1]) * VEC3(0, 0, 1);
        vertices.push_back(pt + noise_term); 
      }
    }
  } else {
    double w = fil.rect.s2;
    double a = fil.rect.s1 / 2;
    double tan_theta = w / (2.0 * R); 
    double u_max = atan(tan_theta); 
    for (int i = 0; i <= NUM_PROFILE_POINTS; i++) {
      double u = -u_max + (2.0 * u_max) * (i + 0.0) / (NUM_PROFILE_POINTS + 0.0); 
      VEC3 x_0(R * sin(u) + w/2 + fil.rect.o[1], a + fil.rect.o[0], R * cos(u) - R);
      for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
        double v = -M_PI + (2.0 * M_PI) * (j + 0.0) / (NUM_SWEEP_POINTS + 0.0);
        VEC3 n(sin(u) * cos(v), sin(v), cos(u) * cos(v));
        VEC3 pt = x_0 + a * n;
        VEC3 noise_term = VEC3(0,0,0); 
        if (ADD_NOISE) 
          noise_term = perlin.noise2D(pt[0], pt[1]) * VEC3(0, 0, 1);
        vertices.push_back(pt + noise_term); 
      }
    }
  }
  for (int i = 0; i <= NUM_PROFILE_POINTS - 1; i++) {
    for (int j = 0; j < NUM_SWEEP_POINTS; j++) {
      int l = i * NUM_SWEEP_POINTS + j; 
      int m = i * NUM_SWEEP_POINTS + (j + 1) % NUM_SWEEP_POINTS; 
      int n = (i + 1) * NUM_SWEEP_POINTS + (j + 1) % NUM_SWEEP_POINTS; 
      int p = (i + 1) * NUM_SWEEP_POINTS + j;
      Triangle *t1 = new Triangle(vertices[l], vertices[m], vertices[n]); 
      Triangle *t2 = new Triangle(vertices[n], vertices[p], vertices[l]); 
      VEC3 tangent_dir = rodriguez_formula(normalized(vertices[p] - vertices[l]), t1->normal(VEC3(0,0,0)), PHI);
      auto c1 = getColor(l, m, n, t1, tangent_dir, fil.type, map_type); 
      auto c2 = getColor(n, p, l, t2, tangent_dir, fil.type, map_type); 
      t1->color = c1;
      t2->color = c2;
      scene.add_primitive(t1);
      scene.add_primitive(t2);
    }
  }
}

void build (FilamentMap &fil_map, FeatureMapType map_type) {
  R = R / N_TILE;
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
    ("d,delta-transmission", "Delta transmission", cxxopts::value<bool>()->default_value(std::to_string(DELTA_TRANSMISSION)))
    ("a,add-noise", "Add Perlin Noise", cxxopts::value<bool>()->default_value(std::to_string(ADD_NOISE)))
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
  DELTA_TRANSMISSION = result["delta-transmission"].as<bool>();
  ADD_NOISE = result["add-noise"].as<bool>();
  file_path = result["file-path"].as<string>(); 

  FilamentMap fil_map = readFilamentMap(file_path); 
  vector<FeatureMapType> types = { NORMAL_MAP, TANGENT_MAP, ID_MAP, POSITION_MAP, BENT_NORMAL_MAP }; 
  vector<string> names = { "normal_map", "tangent_map", "id_map", "position_map", "bent_normal_map" }; 
  for (int i = 0; i < types.size(); i++) {
    renderMapType(fil_map, types[i], names[i]);
  }
  return 0;
}
