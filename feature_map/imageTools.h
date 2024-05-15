#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>

using namespace std;

struct Image { 
  vector<float> values;
  int xRes, yRes;

  Image () {} 

  Image (int xRes, int yRes) : xRes(xRes), yRes(yRes) {
    values.resize(4 * xRes * yRes);
  }

  void set_color (int i, int j, VEC3 c) {
    int x = i + j * xRes;
    values[4 * x]     = 255.0 * max(0.0, min(c(0), 1.0));
    values[4 * x + 1] = 255.0 * max(0.0, min(c(1), 1.0));
    values[4 * x + 2] = 255.0 * max(0.0, min(c(2), 1.0));
    values[4 * x + 3] = 255.0;
  }

  void set_color_4 (int i, int j, VEC4 c) {
    int x = i + j * xRes;
    values[4 * x]     = 255.0 * max(0.0, min(c(0), 1.0));
    values[4 * x + 1] = 255.0 * max(0.0, min(c(1), 1.0));
    values[4 * x + 2] = 255.0 * max(0.0, min(c(2), 1.0));
    values[4 * x + 3] = 255.0 * max(0.0, min(c(3), 1.0));
  }

  VEC3 get_color (int i, int j) { 
    if (i < 0 || i >= xRes || j < 0 || j >= yRes) {
      cout << "(get_color) You messed up somewhere!" << endl;
      exit(0);
    }
    int index = i + j * xRes;
    return VEC3(values[4 * index], values[4 * index + 1], values[4 * index + 2]) / 255.0;
  }

  VEC4 get_color_4 (int i, int j) { 
    if (i < 0 || i >= xRes || j < 0 || j >= yRes) {
      cout << "(get_color) You messed up somewhere!" << endl;
      exit(0);
    }
    int index = i + j * xRes;
    return VEC4(values[4 * index], values[4 * index + 1], values[4 * index + 2], values[4 * index + 3]) / 255.0;
  }

};

void writePNG(const string& filename, Image &im)
{
  vector<float> &values = im.values;
  int &xRes = im.xRes;
  int &yRes = im.yRes;
  int totalCells = xRes * yRes;
  unsigned char* pixels = new unsigned char[4 * totalCells];
  for (int i = 0; i < 4 * totalCells; i++)
    pixels[i] = static_cast<unsigned char>(values[i]); 

  stbi_write_png(filename.c_str(), im.xRes, im.yRes, 4, pixels, im.xRes * 4);
}

#endif
