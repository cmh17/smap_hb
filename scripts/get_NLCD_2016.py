"""
Source: https://github.com/reirby/LandCoverDownloader.git
Modified minimally to get 2016 instead of 2021 data.

Download NLCD_2016_Land_Cover_L48 (US) landcover from www.mrlc.gov WCS 
for the extent of the input SHP file. The output is a tif image in 4326 coordinate system.
The target spatial resolution can be specified in degrees (4326) as an optional parameter.
Images for large extents are downloaded in tiles and then merged together
"""
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo_utils import gdal_calc
import os
import math
import sys


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("Usage: python get_NLCD_2016.py shp-file  cell-size-in-degrees(optional)")
        sys.exit(1)  # Exit with a non-zero status to indicate an error

    # Extract command-line arguments
    shapefile_path = sys.argv[1]
    if len(sys.argv) == 3:
        cell_size = sys.argv[2]
        try:
            cell_size = float(cell_size)
        except ValueError:
            print("Error: The second parameter must be a valid float.")
            sys.exit(1)
    else:
        cell_size = None

    get_land_cover(shapefile_path, cell_size)


def get_land_cover(shapefile_path, spatial_resolution=None):
    ext = get_bbox_4326(shapefile_path)
    min_x, max_x, min_y, max_y = ext
    span_x = abs(max_x - min_x)
    span_y = abs(max_y - min_y)
    area = span_x * span_y
    folder_path = os.path.dirname(shapefile_path)
    file_name = os.path.basename(shapefile_path)

    if area <= 9:  # 9 square degrees
        # just get the whole image
        out_img = os.path.join(folder_path, file_name[:-4] + ".tif")
        wcs_url = get_url(ext)
        print("Getting the image...")
        get_IMG(wcs_url, out_img, spatial_resolution)
    else:
        # split extent to tiles of ~ 1sq deg
        tile_extents = split_extent(ext, spatial_resolution)
        failed_img = []
        tile_paths = []
        for i, ext in enumerate(tile_extents):
            print(f"Getting image {i+1} out of {len(tile_extents)}")
            wcs_url = get_url(ext)
            out_folder = os.path.join(folder_path, file_name[:-4])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            out_img = os.path.join(out_folder, file_name[:-4] + f"_{i}.tif")
            result = get_IMG(wcs_url, out_img, spatial_resolution)
            if result == "fail":
                failed_img.append(i)
            else:
                tile_paths.append(out_img)
        # try downloading missed images one more time
        if len(failed_img) > 0:
            for i in failed_img:
                ext = tile_extents[i]
                min_x, max_x, min_y, max_y = ext
                out_img = os.path.join(out_folder, file_name[:-4] + f"_{i}.tif")
                wcs_url = get_url(ext)
                result = get_IMG(wcs_url, out_img, spatial_resolution)
                if result == "success":
                    tile_paths.append(out_img)
                else:
                    print(f"Something is wrong. Can't get the file with {wcs_url}.")

        # mosaic tiles back together
        output_mosaic = os.path.join(out_folder, file_name[:-4] + "_mosaic.tif")
        build_mosaic(tile_paths, output_mosaic)


def split_extent(extent, spatial_resolution):
    # Size of each subextent (approximately 1 square degree)
    min_x, max_x, min_y, max_y = extent
    subextent_size = 1.0  # in degrees
    # add some tile overlap to prevent possible gaps
    if spatial_resolution:
        tile_margin = spatial_resolution
    else:
        tile_margin = 0.0002

    subextents = []

    # Calculate the number of rows and columns
    num_rows = math.ceil((max_y - min_y) / subextent_size)
    num_cols = math.ceil((max_x - min_x) / subextent_size)

    step_x = (max_x - min_x) / num_cols
    step_y = round((max_y - min_y) / num_rows, 8)

    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the bounds of the subextent
            subextent_min_x = round((min_x + col * step_x) - tile_margin, 8)
            subextent_max_x = round((min_x + (col + 1) * step_x) + tile_margin, 8)
            subextent_min_y = round((min_y + row * step_y) - tile_margin, 8)
            subextent_max_y = round((min_y + (row + 1) * step_y) + tile_margin, 8)

            # Add the subextent to the list
            subextents.append(
                (subextent_min_x, subextent_max_x, subextent_min_y, subextent_max_y)
            )

    return subextents


def get_IMG(wcs_url, fname, spatial_resolution=None):
    ds = gdal.Open(wcs_url)
    # Check if the dataset is successfully opened
    if ds is None:
        print("Failed to open WCS dataset")
        return "fail"
    else:
        output_format = "GTiff"
        # Create output dataset
        image_driver = gdal.GetDriverByName(output_format)

        if spatial_resolution:
            # Use specified spatial resolution if provided
            output_ds = gdal.Warp(
                fname,
                ds,
                xRes=spatial_resolution,
                yRes=-spatial_resolution,
                resampleAlg="near",
            )
        else:
            # Use default spatial resolution
            output_ds = image_driver.CreateCopy(fname, ds)

        # Close datasets
        ds = None
        output_ds = None

        print(f"Image saved to {fname}")
        return "success"


def build_mosaic(input_IMGs, output_mosaic):
    # List of input raster files
    input_files = input_IMGs
    # Output mosaic file
    output_file = output_mosaic
    # Warp options - here we set the format to GeoTIFF
    warp_options = gdal.WarpOptions(
        format="GTiff", resampleAlg="near", creationOptions=["COMPRESS=LZW"]
    )

    # Mosaic the input files
    gdal.Warp(output_file, input_files, options=warp_options)
    print(f"Mosaic created: {output_mosaic}")


def get_url(extent):
    min_x, max_x, min_y, max_y = extent
    url = f"https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2016_Land_Cover_L48/wcs?service=WCS&version=2.0.1&request=getcoverage&coverageid=NLCD_2016_Land_Cover_L48&subset=Lat({min_y},{max_y})&subset=Long({min_x},{max_x})&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    return url


def get_4326_extent(layer, layer_srs, target_srs):
    in_memory_driver = ogr.GetDriverByName("Memory")
    # Create the target shapefile
    temp_ds = in_memory_driver.CreateDataSource("")
    # Create the target layer
    target_layer = temp_ds.CreateLayer(
        "reprojected_layer", geom_type=layer.GetGeomType()
    )
    # Create a transformation from the layer's spatial reference to EPSG:4326
    transform = osr.CoordinateTransformation(layer_srs, target_srs)

    # Apply the coordinate transformation to each feature and add to the target layer
    for feature in layer:
        geometry = feature.GetGeometryRef()
        geometry.Transform(transform)

        target_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_feature.SetGeometry(geometry)
        target_layer.CreateFeature(target_feature)
        target_feature = None  # Release the target feature

    extent = target_layer.GetExtent()
    target_ds = None
    return extent


def get_bbox_4326(shapefile_path):
    # set the CRS for the bounding box
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srid = target_srs.GetAuthorityCode(None)

    # Open the shapefile
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    dataset = shp_driver.Open(shapefile_path)

    if dataset is None:
        print(f"Failed to open shapefile: {shapefile_path}")
        return
    else:
        # Get the layer
        layer = dataset.GetLayer()
        # Get the spatial reference
        layer_srs = layer.GetSpatialRef()
        layer_srid = layer_srs.GetAuthorityCode(None)
        # reproject to 4326 and get the extent if shp is in different CRS
        if layer_srid != target_srid:
            extent = get_4326_extent(layer, layer_srs, target_srs)
        else:
            extent = layer.GetExtent()

        # Close the dataset
        dataset = None

        return extent

if __name__ == "__main__":
    main()
