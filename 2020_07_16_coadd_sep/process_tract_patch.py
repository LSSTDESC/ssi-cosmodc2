import os
import subprocess

import click
import numpy as np

import fitsio
import esutil.numpy_util
import sep

from lsst.daf.persistence import Butler
from sxdes import run_sep
from ssi_tools.layout_utils import make_hexgrid_for_tract
from fsi_tools.matching import do_balrogesque_matching
from desc_dc2_dm_data import REPOS

sep.set_extract_pixstack(1_000_000)

# this list is hard coded - the gen 2 butler doesn't have a method for introspection
DC2_TRACTS = set(
    [
        2723, 2730, 2897, 2904, 3076, 3083, 3259, 3266, 3445, 3452, 3635, 3642, 3830,
        3837, 4028, 4035, 4230, 4428, 4435, 4636, 4643, 4851, 4858, 5069, 2724, 2731,
        2898, 2905, 3077, 3084, 3260, 3267, 3446, 3453, 3636, 3643, 3831, 4022, 4029,
        4224, 4231, 4429, 4436, 4637, 4644, 4852, 4859, 5070, 2725, 2732, 2899, 2906,
        3078, 3085, 3261, 3268, 3447, 3454, 3637, 3825, 3832, 4023, 4030, 4225, 4232,
        4430, 4437, 4638, 4645, 4853, 4860, 5071, 2726, 2733, 2900, 2907, 3079, 3086,
        3262, 3441, 3448, 3631, 3638, 3826, 3833, 4024, 4031, 4226, 4233, 4431, 4438,
        4639, 4646, 4854, 5065, 5072, 3451, 2727, 2734, 2901, 2908, 3080, 3256, 3263,
        3442, 3449, 3632, 3639, 3827, 3834, 4025, 4032, 4227, 4234, 4432, 4439, 4640,
        4647, 4855, 5066, 5073, 2728, 2735, 2902, 3074, 3081, 3257, 3264, 3443, 3450,
        3633, 3640, 3828, 3835, 4026, 4033, 4228, 4235, 4433, 4440, 4641, 4648, 4856,
        5067, 5074, 2729, 2896, 2903, 3075, 3082, 3258, 3265, 3444, 3634, 3641, 3829,
        3836, 4027, 4034, 4229, 4236, 4434, 4441, 4642, 4850, 4857, 5068,
    ]
)

# DC2 truth catalog to use as injected sources
DC2_TRUTH_CAT = (
    "/global/cfs/cdirs/lsst/groups/fake-source-injection/DC2/catalogs/"
    "cosmoDC2_v1.1.4_small_fsi_catalog.fits"
)

OUTPUT_BUTLER = os.path.expandvars(os.path.join("$SCRATCH", "butler_coadd_sep"))
OUTPUT_DIR = "ssi_cats"


def _run_sep_and_add_radec(ti, img, zp, err=None, minerr=None):
    if err is None:
        err = np.sqrt(img.variance.array.copy())
        img = img.image.array.copy()

    if minerr is not None:
        msk = err < minerr
        err[msk] = minerr

    cat, seg = run_sep(
        img,
        err,
    )
    cat = esutil.numpy_util.add_fields(
        cat,
        [("ra", "f8"), ("dec", "f8"), ("mag_auto", "f8")]
    )
    wcs = ti.getWcs()
    cat["ra"], cat["dec"] = wcs.pixelToSkyArray(cat["x"], cat["y"], degrees=True)
    cat["mag_auto"] = zp - 2.5*np.log10(cat["flux_auto"])
    return cat, seg


@click.command()
@click.option(
    '--tract', type=int, default=None, help='the tract to process', required=True
)
@click.option(
    '--patch', type=int, default=None, help='the patch to process', required=True
)
@click.option('--seed', type=int, default=None, help='seed for the RNG', required=True)
def main(tract, patch, seed):
    """Run SSI on a DC2 tract and patch"""
    # first we need to extract the tract and patch from the butler in order to
    # setup the source catalog
    butler = Butler(REPOS["2.2i_dr6_wfd"])
    skymap = butler.get("deepCoadd_skyMap")

    if tract not in DC2_TRACTS:
        raise RuntimeError("Tract %d is not valid for DC2!" % tract)

    ti = skymap[tract]

    if patch < 0 or patch >= len(ti):
        raise RuntimeError(
            "patch %d is not valid for tract %d (has only %d patches)!" % (
                patch, tract, len(tract)
            )
        )

    # now we are making the truth catalog
    # - we cut to things brighter than mag 25 to avoid injecting gobs of faint things
    # - we cut the injection catalog to the patch bounaries in order to avoid drawing
    #   extra stuff
    # - we have to write the tract sources to disk for the stack task
    grid = make_hexgrid_for_tract(ti, rng=seed)
    srcs = fitsio.read(DC2_TRUTH_CAT)

    msk = srcs["rmagVar"] <= 25
    srcs = srcs[msk]

    rng = np.random.RandomState(seed=seed)
    inds = rng.choice(len(srcs), size=len(grid), replace=True)
    tract_sources = srcs[inds].copy()
    tract_sources["raJ2000"] = np.deg2rad(grid["ra"])
    tract_sources["decJ2000"] = np.deg2rad(grid["dec"])

    pi = ti[patch]
    msk = pi.getOuterBBox().contains(grid["x"], grid["y"])
    tract_sources = tract_sources[msk]

    subprocess.run("mkdir -p " + OUTPUT_DIR, shell=True, check=True)

    ssi_src_file = os.path.join(
        OUTPUT_DIR, "ssi_input_tract%d_patch%d.fits" % (tract, patch)
    )
    fitsio.write(
        ssi_src_file,
        tract_sources,
        clobber=True,
    )

    # now we need to run the SSI
    # for this we need to define an output butler area
    subprocess.run("mkdir -p " + OUTPUT_BUTLER, shell=True, check=True)
    cmd = """\
insertFakes.py \
/global/cfs/cdirs/lsst/production/DC2_ImSim/Run2.2i/desc_dm_drp/v19.0.0-v1\
/rerun/run2.2i-coadd-wfd-dr6-v1 \
--output %s/ \
--id tract=%d patch=%s \
filter=r -c fakeType=%s \
--clobber-config --no-versions
""" % (OUTPUT_BUTLER, tract, "%d,%d" % pi.getIndex(), ssi_src_file)
    subprocess.run(cmd, shell=True, check=True)

    # from here we have images with the sources on disk
    # we are going to read them back in, make a few catalogs, and output the data
    output_butler = Butler(OUTPUT_BUTLER)
    bbox = pi.getOuterBBox()
    coaddId = {
        'tract': ti.getId(),
        'patch':  "%d,%d" % pi.getIndex(),
        'filter': 'r'
    }

    image = output_butler.get(
        "deepCoadd_sub", bbox=bbox, immediate=True, dataId=coaddId
    )
    fake_image = output_butler.get(
        "fakes_deepCoadd_sub", bbox=bbox, immediate=True, dataId=coaddId
    )
    zp = 2.5*np.log10(image.getPhotoCalib().getInstFluxAtZeroMagnitude())

    orig_det_cat, orig_det_seg = _run_sep_and_add_radec(ti, image, zp)
    ssi_det_cat, ssi_det_seg = _run_sep_and_add_radec(ti, fake_image, zp)
    ssi_truth_cat, ssi_truth_seg = _run_sep_and_add_radec(
        ti,
        (fake_image.image.array - image.image.array).copy(),
        zp,
        np.zeros_like(np.sqrt(fake_image.variance.array.copy())),
        minerr=np.mean(np.sqrt(fake_image.variance.array.copy())),
    )

    match_flag, match_index = do_balrogesque_matching(
        ssi_det_cat, orig_det_cat, ssi_truth_cat, "flux_auto",
    )

    ssi_det_cat = esutil.numpy_util.add_fields(
        ssi_det_cat,
        [("match_flag", "i4"), ("match_index", "i8")]
    )
    ssi_det_cat["match_flag"] = match_flag
    ssi_det_cat["match_index"] = match_index

    output_fname = "ssi_data_tract%d_patch%d.fits" % (tract, patch)
    with fitsio.FITS(
        os.path.join(OUTPUT_DIR, output_fname), "rw", clobber=True
    ) as fits:
        fits.write(orig_det_cat, extname="orig_cat")
        fits.write(ssi_det_cat, extname="ssi_cat")
        fits.write(ssi_truth_cat, extname="truth_cat")


if __name__ == "__main__":
    main()
