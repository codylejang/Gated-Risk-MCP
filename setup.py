from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SROIE_DIRS = [
    "0325updated.task1train(626p)",
    "0325updated.task2train(626p)",
    "task1&2_test(361p)",
    "task3-test(347p)",
    "text.task1&2-test(361p)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time project setup for dependencies and dataset folders."
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip installation of requirements.",
    )
    parser.add_argument(
        "--download-cord",
        action="store_true",
        help="Download the CORD dataset into Data/CORD.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_data_layout(repo_root: Path) -> None:
    data_dir = repo_root / "Data"
    cord_dir = data_dir / "CORD"
    sroie_dir = data_dir / "SROIE"

    data_dir.mkdir(exist_ok=True)
    cord_dir.mkdir(exist_ok=True)
    sroie_dir.mkdir(exist_ok=True)

    for folder_name in SROIE_DIRS:
        (sroie_dir / folder_name).mkdir(exist_ok=True)

    readme_path = sroie_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "\n".join(
                [
                    "Place the manually downloaded SROIE files in this folder structure:",
                    "",
                    *[f"- {folder}" for folder in SROIE_DIRS],
                    "",
                    "CORD can be downloaded automatically with:",
                    "python setup_project.py --download-cord",
                ]
            ),
            encoding="utf-8",
        )


def install_requirements(repo_root: Path) -> None:
    run_command([sys.executable, "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")])


def download_cord_dataset(repo_root: Path) -> None:
    python_code = (
        "from datasets import load_dataset; "
        "ds = load_dataset('naver-clova-ix/cord-v2'); "
        "ds.save_to_disk(r'Data\\CORD')"
    )
    run_command([sys.executable, "-c", python_code])


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    ensure_data_layout(repo_root)

    if not args.skip_install:
        install_requirements(repo_root)

    if args.download_cord:
        download_cord_dataset(repo_root)

    print("Project setup complete.")
    print("Folders prepared under Data/.")
    print("Install command run:", "no" if args.skip_install else "yes")
    print("CORD download run:", "yes" if args.download_cord else "no")


if __name__ == "__main__":
    main()
