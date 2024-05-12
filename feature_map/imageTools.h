#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H
#include <vector>

using namespace std;

struct Image { 
  vector<float> values;
  int xRes, yRes;

  Image () {} 

  Image (int xRes, int yRes) : xRes(xRes), yRes(yRes) {
    values.resize(3 * xRes * yRes);
  }

  void set_color (int i, int j, VEC3 c) {
    int x = i + j * xRes;
    values[3 * x]     = 255.0 * max(0.0, min(c(0), 1.0));
    values[3 * x + 1] = 255.0 * max(0.0, min(c(1), 1.0));
    values[3 * x + 2] = 255.0 * max(0.0, min(c(2), 1.0));
  }

  VEC3 get_color (int i, int j) { 
    if (i < 0 || i >= xRes || j < 0 || j >= yRes) {
      cout << "(get_color) You messed up somewhere!" << endl;
      exit(0);
    }
    int index = i + j * xRes;
    return VEC3(values[3 * index], values[3 * index + 1], values[3 * index + 2]) / 255.0;
  }

};

void readPPM(const string& filename, Image &im)
{
  vector<float> &values = im.values;
  int &xRes = im.xRes;
  int &yRes = im.yRes;
  // try to open the file
  FILE *fp;
  fp = fopen(filename.c_str(), "rb");
  if (fp == NULL)
  {
    cout << " Could not open file \"" << filename.c_str() << "\" for reading." << endl;
    cout << " Make sure you're not trying to read from a weird location or with a " << endl;
    cout << " strange filename. Bailing ... " << endl;
    exit(0);
  }

  // get the dimensions
  unsigned char newline;
  int res = fscanf(fp, "P6\n%d %d\n255%c", &xRes, &yRes, &newline);
  if (newline != '\n') {
    cout << " The header of " << filename.c_str() << " may be improperly formatted." << endl;
    cout << " The program will continue, but you may want to check your input. " << endl;
  }
  int totalCells = xRes * yRes;

  // grab the pixel values
  unsigned char* pixels = new unsigned char[3 * totalCells];
  res = fread(pixels, 1, totalCells * 3, fp);

  // copy to a nicer data type
  for (int i = 0; i < 3 * totalCells; i++)
    values.push_back(pixels[i]);

  // clean up
  delete[] pixels;
  fclose(fp);
  cout << " Read in file " << filename.c_str() << endl;
}

void writePPM(const string& filename, Image &im)
{
  vector<float> &values = im.values;
  int &xRes = im.xRes;
  int &yRes = im.yRes;
  int totalCells = xRes * yRes;
  unsigned char* pixels = new unsigned char[3 * totalCells];
  for (int i = 0; i < 3 * totalCells; i++)
    pixels[i] = values[i];

  FILE *fp;
  fp = fopen(filename.c_str(), "wb");
  if (fp == NULL)
  {
    cout << " Could not open file \"" << filename.c_str() << "\" for writing." << endl;
    cout << " Make sure you're not trying to write from a weird location or with a " << endl;
    cout << " strange filename. Bailing ... " << endl;
    exit(0);
  }

  fprintf(fp, "P6\n%d %d\n255\n", xRes, yRes);
  fwrite(pixels, 1, totalCells * 3, fp);
  fclose(fp);
  delete[] pixels;
}

#endif
