# Installattion

## Linux

1.  create environment
 
    conda create --name tkw1 python=3.5
    source activate tkw1

2. compile dlib without CUDA
https://www.learnopencv.com/install-dlib-on-ubuntu/
https://github.com/ageitgey/face_recognition/issues/236

    wget http://dlib.net/files/dlib-19.6.tar.bz2
    tar xvf dlib-19.6.tar.bz2
    cd dlib-19.6/
    mkdir build
    cd build
    cmake .. -DLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=0
    cmake --build .
    cd ..
    python3 setup.py install --no DLIB_USE_CUDA

3. install other requirements 

    pip install -r requirements_pip.txt



# Starting it

    cd FaceFinder
    python facefinder.py