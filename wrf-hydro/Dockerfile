# Use your backup image as the base (ensure that your backup image is correctly referenced)
FROM wrf-hydro-training:latest

USER root

# Ensure software-properties-common is installed and add the universe repository
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
    libunistring2 \
    libgeos-dev \
    libproj-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install R packages including sf
RUN Rscript -e "install.packages(c('ncdf4','optparse','stringr','plyr'), repos='https://cran.rstudio.com', dependencies=TRUE)"

RUN conda install -y -c conda-forge gdal

# Verify installation
RUN R -e "library(ncdf4); library(optparse); library(stringr); library(plyr)" && python -c "from osgeo import ogr; print('ogr installed successfully')"

# Ensure Miniconda is on PATH (assumes Miniconda is installed at /home/docker/miniconda3)
ENV PATH="/home/docker/miniconda3/bin:${PATH}"

USER docker
WORKDIR /home/docker
