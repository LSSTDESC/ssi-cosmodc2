import click
import subprocess
import numpy as np
import joblib


def _run_func(tract, patch, seed):
    logfile = "log_tract%d_patch%d_seed%d.oe" % (
        tract, patch, seed
    )
    subprocess.run(
        "python process_tract_patch.py --tract=%d --patch=%d --seed=%d >& %s" % (
            tract, patch, seed, logfile
        ),
        shell=True,
        check=False,
    )


@click.command()
@click.option(
    '--tract', type=int, default=None, help='the tract to process', required=True
)
@click.option('--seed', type=int, default=None, help='seed for the RNG')
def main(tract, seed):
    """Run SSI on all patches from a DC2 tract."""
    if seed is None:
        seed = tract
    seeds = np.random.RandomState(seed=tract).randint(low=1, high=2**29, size=49)

    jobs = [
        joblib.delayed(_run_func)(tract, i, seed)
        for i, seed in enumerate(seeds)
    ]

    with joblib.Parallel(backend="loky", n_jobs=8, verbose=100) as par:
        par(jobs)


if __name__ == "__main__":
    main()
