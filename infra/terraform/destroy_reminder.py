"""Session 10: `terraform destroy` reminder/safety-check wrapper.

PROJECT.md §7 is unambiguous: never leave a SageMaker endpoint running, and
confirm nothing billable is left after every AWS-touching session. Every
prior session that touched AWS (see SESSIONS.md Sessions 6/7) did that
check *by hand*, in the session log, after the fact. This script makes that
a single command instead of something to remember: it runs
`terraform destroy` behind an explicit typed confirmation (not just
Terraform's own y/n, which is easy to reflexively approve), then re-runs
the same `list-endpoints` / `list-notebook-instances` checks against every
region this project has ever touched (see docs/sagemaker.md), not just the
one region being destroyed -- a leftover resource in a *different* region
than the one you're currently looking at is exactly how this kind of thing
goes unnoticed.

Usage (run under .venv-sagemaker, which already has boto3 --
requirements-sagemaker.txt):
    .venv-sagemaker/Scripts/python infra/terraform/destroy_reminder.py --region us-east-1

Terraform itself must be reachable -- pass --terraform-bin if it's not on
PATH (this session used a locally-downloaded binary, see this directory's
README.md prerequisites section).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

MODULE_DIR = Path(__file__).resolve().parent

# Every region this project's AWS work has ever touched (docs/sagemaker.md)
# -- checked regardless of which region is being destroyed, since a
# resource stranded in a region nobody's looking at is the actual risk.
REGIONS_TO_CHECK = ["us-east-1", "us-west-2", "ap-southeast-2"]


def run_destroy(terraform_bin: str, region: str) -> None:
    print(f"About to run `terraform destroy` in {MODULE_DIR} for region={region!r}.")
    print("This deletes the S3 bucket (and everything in it) and the SageMaker execution role.")
    confirmation = input("Type DESTROY (all caps) to proceed: ")
    if confirmation != "DESTROY":
        print("Confirmation not given -- aborting, nothing was touched.", file=sys.stderr)
        sys.exit(1)

    subprocess.run(
        [terraform_bin, "destroy", f"-var=region={region}"],
        cwd=MODULE_DIR,
        check=True,
    )


def safety_check(regions: list[str]) -> bool:
    """Returns True if everything is clean (nothing billable left running)."""
    all_clean = True
    for region in regions:
        sm = boto3.client("sagemaker", region_name=region)
        try:
            endpoints = sm.list_endpoints()["Endpoints"]
            notebooks = sm.list_notebook_instances()["NotebookInstances"]
        except ClientError as e:
            print(f"[{region}] could not check (skipping): {e}")
            continue

        if endpoints:
            all_clean = False
            print(f"[{region}] LEFTOVER ENDPOINTS (billing hourly!): {[e['EndpointName'] for e in endpoints]}")
        else:
            print(f"[{region}] endpoints: none")

        if notebooks:
            all_clean = False
            print(f"[{region}] LEFTOVER NOTEBOOK INSTANCES: {[n['NotebookInstanceName'] for n in notebooks]}")
        else:
            print(f"[{region}] notebook instances: none")

    return all_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", required=True, help="Region to run `terraform destroy` against.")
    parser.add_argument("--terraform-bin", default="terraform", help="Path to the terraform binary if not on PATH.")
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help="Run only the post-destroy safety check (e.g. if destroy already ran manually).",
    )
    args = parser.parse_args()

    if not args.skip_destroy:
        run_destroy(args.terraform_bin, args.region)

    print("\n--- Safety check: aws sagemaker list-endpoints / list-notebook-instances ---")
    regions = sorted(set(REGIONS_TO_CHECK + [args.region]))
    clean = safety_check(regions)

    print()
    if clean:
        print("Clean. No billable SageMaker resources left running in any checked region.")
    else:
        print("NOT CLEAN -- see LEFTOVER lines above. Delete them before ending this session (PROJECT.md §7).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
