# Use backup image as the base
FROM 7d8007fb133ba195aab037c64832e0d2ebdb6e94c4ef579191aae7ebe2e1ee02

USER root

RUN apt-get update && apt-get install -yq --no-install-recommends software-properties-common && \
    apt-add-repository universe && apt-get update

# Update repositories and install necessary dependencies (system-level)
RUN apt-get update && apt-get install -yq --allow-unauthenticated --no-install-recommends \
    bc \
    libnetcdf-dev \
    libpng-dev \
    libssl-dev \
    libswitch-perl \
    libxml2-dev \
    locales \
    m4 \
    nodejs \
    pkg-config \
    python3-geopandas \
    python3-h5py \
    python3-mpi4py \
    python3-matplotlib \
    python3-netcdf4 \
    python3-numpy \
    python3-pyproj \
    python3-rasterio \
    r-base-core \
    valgrind \
    libunistring0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# R packages installation
RUN Rscript -e "install.packages(c('ncdf4','optparse','stringr','plyr'), repos='https://cran.rstudio.com')"

# Miniconda is already installed on the backup
ENV PATH="/home/docker/miniconda3/bin:${PATH}"

RUN conda install -y -c conda-forge gdal

USER docker

WORKDIR /home/docker
