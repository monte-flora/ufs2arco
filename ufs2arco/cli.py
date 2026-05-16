import argparse
import yaml
from ufs2arco.driver import Driver
from ufs2arco.multidriver import MultiDriver

def main():
    parser = argparse.ArgumentParser(
        description="Run the ufs2arco workflow with a given YAML recipe.",
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML recipe file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass this flag to overwrite an existing zarr store in the specified location",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configs without saving data",
    )

    parser.add_argument(
        "--tranche",
        type=int,
        default=None,
        help=(
            "Run a specific tranche (0-indexed) from the tranche manifest at "
            "<recipe-stem>.tranches.yaml. Bounds the batch loop to that tranche's "
            "[start, stop) range and skips finalize. Generate the manifest with "
            "`python -m ufs2arco.tranches init`."
        ),
    )

    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Skip ingest entirely; only run target.finalize() + finalize_attributes() "
            "on the existing zarr. Errors if any tranche is still pending (override with --force)."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass safety checks (currently: --finalize on incomplete tranches).",
    )

    args = parser.parse_args()

    # Mutually-exclusive run modes.
    if args.tranche is not None and args.finalize:
        parser.error("--tranche and --finalize are mutually exclusive")

    with open(args.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    if "multisource" in config.keys():
        driver = MultiDriver(args.yaml_file)
        driver.run(
            overwrite=args.overwrite,
            validate=args.validate,
            tranche_id=args.tranche,
            finalize_only=args.finalize,
            force=args.force,
        )
    else:
        driver = Driver(args.yaml_file)
        driver.run(
            overwrite=args.overwrite,
            tranche_id=args.tranche,
            finalize_only=args.finalize,
            force=args.force,
        )

if __name__ == "__main__":
    main()
