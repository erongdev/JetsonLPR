FROM nvcr.io/nvidia/deepstream-l4t:6.1.1-triton

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Bangkok
#For chinese comments
ENV LANG C.UTF-8

RUN apt-get update && \
    apt install libavformat58 --reinstall && \
    apt install libavcodec58 --reinstall && \
    apt install libavresample4 --reinstall && \
    apt install libavutil56 --reinstall

RUN apt-get update && \
    find / -name " *opencv* " -exec rm -i {} \; && \
    apt-get autoremove -y && \
    apt-get install -y \
      pkg-config cmake build-essential \
      git autoconf libtool unzip \
      python3-dev python3-pip \
      libatlas-base-dev gfortran ffmpeg libgtk2.0-dev \
      libavcodec-dev libavformat-dev libswscale-dev \
      libtbb2 libtbb-dev libjpeg-dev libtiff5-dev \
      libdc1394-22-dev qt5-default libopenblas-dev \
      libgstreamer1.0-0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
      gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools \
      protobuf-compiler libprotoc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir numpy==1.19.4 && \
    pip3 install --no-cache-dir tqdm==4.48.2 easydict six && \
    pip3 install --no-cache-dir pycuda==2019.1.2 && \
    pip3 install --no-cache-dir pydantic && \
    pip3 install --no-cache-dir imagezmq simplejpeg && \
    pip3 install --no-cache-dir shapely

WORKDIR /git
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protobuf-python-3.8.0.zip && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.8.0/protoc-3.8.0-linux-aarch_64.zip && \
    unzip protobuf-python-3.8.0.zip && \
    unzip protoc-3.8.0-linux-aarch_64.zip -d protoc-3.8.0 && \
    cp protoc-3.8.0/bin/protoc /usr/local/bin/protoc && \
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp && \
    cd /git/protobuf-3.8.0/ && \
    bash autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && make check && make install && \
    ldconfig && \
    cd python/ && \
    python3 setup.py build --cpp_implementation && \
    python3 setup.py test --cpp_implementation && \
    python3 setup.py install --cpp_implementation && \
    cd /git/ && rm -rf /git/* && \
    pip3 install --no-cache-dir onnx==1.9.0 && \
    pip3 install --no-cache-dir pyinstaller-hooks-contrib==2021.3 && \
    pip3 install --no-cache-dir pyinstaller==4.7


RUN wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.5.3.tar.gz && \
    wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.3.tar.gz && \
    tar zxvf opencv.tar.gz && \
    tar zxvf opencv_contrib.tar.gz && \
    cd opencv-4.5.3 && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_GTK=ON -D WITH_GTK_2_X=ON -D OPENCV_GENERATE_PKGCONFIG=YES -D WITH_FREETYPE=YES -D OPENCV_EXTRA_MODULES_PATH=/git/opencv_contrib-4.5.3/modules -D OPENCV_ENABLE_NONFREE=True .. && \
    make -j8 && make install && \
    rm -rf /git/*


