#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re
import sys
import typing
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)


def move_files(directory: str) -> None:
    base_path: Path = Path(directory).absolute()
    done_base_path: Path = Path(str(base_path) + "/done")
    done_base_path.mkdir(exist_ok=True, parents=True)

    files: typing.List[Path] = [file for file in base_path.iterdir() if file.is_file()]

    pattern = re.compile(r'INV-(20[0-9]{2})-([0-9]{3,6})_')

    for file in files:
        match = pattern.search(str(file))

        if match:
            year: str
            number: str
            year, number = match.groups()

            doc_grouping: int = int(number) // 100

            done_path = done_base_path.joinpath(f"INV/{year}/{doc_grouping:02}00")

            # Make the directory if it doesn't exist, including parent directories
            done_path.mkdir(exist_ok=True, parents=True)

            file.rename(done_path.joinpath(file.name))

            logging.info(f"moved {file}")

        else:
            logging.warning(f"SKIPPING {file}")


def main() -> None:
    target: str = sys.argv[1]

    if target:
        move_files(target)


if __name__ == "__main__":
    main()
