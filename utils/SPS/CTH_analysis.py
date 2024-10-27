#!/bin/env python

"""

"""

import argparse
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cf_units
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.cartography
from iris.analysis.geometry import geometry_area_weights
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
from iris.util import equalise_attributes, mask_cube
import matplotlib.pyplot as plt
import numpy as np

india_lonmin, india_lonmax, india_latmin, india_latmax = 66, 90, 4, 37


def update_geo_coords(cube_geo):
    """
    The x and y coordinates for the geostationary projection are
    defined in radians. Use the satellite height to change these to
    metres, which Iris needs.
    """
    y_coord = cube_geo.coord("projection_y_coordinate")
    x_coord = cube_geo.coord("projection_x_coordinate")
    y_angles = y_coord.points
    x_angles = x_coord.points

    sat_height = y_coord.coord_system.perspective_point_height
    y_pts = sat_height * y_angles
    x_pts = sat_height * x_angles

    cube_geo_copy = cube_geo.copy()
    cube_geo_copy.coord("projection_y_coordinate").points = y_pts[:]
    cube_geo_copy.coord("projection_x_coordinate").points = x_pts[:]
    cube_geo_copy.coord("projection_y_coordinate").units = cf_units.Unit("m")
    cube_geo_copy.coord("projection_x_coordinate").units = cf_units.Unit("m")

    return cube_geo_copy


def plot_india(cube):
    """
    Plot the CTH on a PlateCarree projection,
    setting extent to India
    """
    plt.figure()
    plt.axes(
        projection=ccrs.PlateCarree(),
        extent=[india_lonmax, india_lonmin, india_latmax, india_latmin],
    )
    qplt.pcolormesh(cube)
    plt.gca().coastlines()
    iplt.show()


def empty_equi_cube(spacing, coord_bounds=None):
    """
    Return an empty equirectangular Iris cube
    spacing is degrees in both lat and lon
    coord_bounds should be [lonmin, lonmax, latmin, latmax]
    """
    if not coord_bounds:
        coord_bounds = [-180 + spacing, 180, -90, 90]
    lon0, lon1, lat0, lat1 = coord_bounds
    # Define equirectangular CRS
    semimajor_axis = 6378137
    semiminor_axis = 6356752.3
    CRS = GeogCS(semi_major_axis=semimajor_axis, semi_minor_axis=semiminor_axis)
    # Define lat/lon grid
    lats = np.arange(lat0, lat1, spacing)
    lons = np.arange(lon0, lon1, spacing)
    # Set up cube
    x = DimCoord(
        lons, standard_name="longitude", units="degrees_east", coord_system=CRS
    )
    y = DimCoord(
        lats, standard_name="latitude", units="degrees_north", coord_system=CRS
    )
    # Build the Iris cube
    empty_data = np.zeros((lats.size, lons.size))
    cube = iris.cube.Cube(empty_data)
    cube.add_dim_coord(y, 0)
    cube.add_dim_coord(x, 1)
    cube.coord("latitude").guess_bounds()
    cube.coord("longitude").guess_bounds()
    return cube


def regrid_regular_grid_india(source_cube):
    """
    Define an equirectangular lat/lon grid covering India
    and regrid input cube to this grid, using nearest
    neighbour interpolation
    """
    target_cube = empty_equi_cube(
        0.02, [india_lonmin, india_lonmax, india_latmin, india_latmax]
    )
    source_cube = update_geo_coords(source_cube)
    cube_ll = source_cube.regrid(target_cube, iris.analysis.Nearest())
    return cube_ll


def extract_india_cube(cube):
    """
    Given a cube that has a equirectangular lat/lon grid
    definition, return a similar cube with all data outside
    of India masked
    """
    filename = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(filename)

    india_geometry = [
        record.geometry
        for record in reader.records()
        if record.attributes["ISO_A3"] == "IND"
    ][0]

    india_weights = geometry_area_weights(cube, india_geometry, normalize=True)
    india_mask = np.where(india_weights > 0, False, True)
    india_cube = mask_cube(cube, india_mask)

    return india_cube


def plot_india_mean_cth(cube):
    """
    For a cube with multiple times, calculate the area mean
    for each time and plot a time series
    """
    grid_areas = iris.analysis.cartography.area_weights(cube)
    cube_mean = cube.collapsed(
        ["longitude", "latitude"], iris.analysis.MEAN, weights=grid_areas
    )
    qplt.plot(cube_mean, "x")
    iplt.show()


if __name__ == "__main__":

    msg = (
        "Plots CTH files. For a single file, create map plot over India. "
        "For multiple files create a timeseries of mean CTH."
    )
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("cth_files", nargs="+", help="CTH netcdf file(s)")
    args = parser.parse_args()

    if len(args.cth_files) == 1:
        # One file
        cube_geo = iris.load_cube(args.cth_files)
        plot_india(cube_geo)
    else:
        # Multiple files
        cubes_geo = iris.load(args.cth_files)
        # Merge the cubes in the time dimension
        # First the attributes need to be equalised to avoid merge failure
        # (https://scitools-iris.readthedocs.io/en/latest/userguide/merge_and_concat.html#merge-concat-common-issues)
        removed_attributes = equalise_attributes(cubes_geo)
        cube_geo = cubes_geo.merge_cube()

        # Regrid to lat/lon grid for just India region
        cube_ll = regrid_regular_grid_india(cube_geo)

        # Plot time series of mean values
        plot_india_mean_cth(cube_ll)
