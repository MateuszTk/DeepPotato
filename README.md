# DeepPotato
Simple neural network

## Build
```
mkdir build
cd build
cmake .. -DBUILD_XOR=ON -DBUILD_IMAGE_COMPRESSION=OFF
make
```
### Windows
```
mkdir build
cd build
cmake .. -G "NMake Makefiles" -DBUILD_XOR=ON -DBUILD_IMAGE_COMPRESSION=OFF
nmake
```
## Demo description
#### XOR
XOR demo is a simple neural network that learns XOR function.
#### Image compression
Image compression demo is a simple neural network that reproduces input image.
### Note
Additional dependencies are required for image compression demo.
```
cd demo/image_compression/external
git clone https://github.com/nothings/stb
git clone --depth 1 --branch SDL2 https://github.com/libsdl-org/SDL.git
```
