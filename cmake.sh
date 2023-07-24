#~!/bin/bash
sudo rm -rf build
sudo rm bin/*
mkdir build
cd build
cmake ..
make -j4
sleep 1s
cd ..

