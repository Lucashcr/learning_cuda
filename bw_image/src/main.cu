#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char *loadImage(const char *filePath, int *width, int *height,
                         int *channels) {
  std::cout << filePath << std::endl;
  unsigned char *img = stbi_load(filePath, width, height, channels, 0);
  if (img == NULL) {
    std::cout << "Error in loading the image" << std::endl;
    exit(1);
  }

  std::cout << "Loaded image with a width of " << *width << "px, a height of "
            << *height << "px and " << *channels << " channels\n"
            << std::endl;

  return img;
}

extern "C" void applyBlackAndWhiteFilter(unsigned char *h_img, const int width,
                                         const int height, const int channels);

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "File path not informed" << std::endl;
    exit(1);
  }

  if (argc > 3) {
    std::cout << "Too many arguments" << std::endl;
    exit(1);
  }

  std::string inputFilePath = argv[1];
  std::string outputFilePath;
  if (argc != 3) {
    outputFilePath = argv[1];
    auto dotPos = outputFilePath.rfind('.');
    outputFilePath = outputFilePath.replace(dotPos, 0, "-bw");
  } else {
    outputFilePath.append(argv[2]);
  }

  int width, height, channels;
  unsigned char *img =
      loadImage(inputFilePath.c_str(), &width, &height, &channels);

  applyBlackAndWhiteFilter(img, width, height, channels);

  stbi_write_png(outputFilePath.c_str(), width, height, channels, img,
                 width * channels);
  stbi_image_free(img);

  return 0;
}
