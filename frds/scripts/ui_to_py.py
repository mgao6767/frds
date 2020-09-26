"""This script convert all .ui files to Python files"""

import os
import pathlib
import subprocess


def pyuic_all_ui_files(
    ui_files_dir="gui/ui_components/ui_files",
    output_files_dir="gui/ui_components/generated_py_files",
    prefix="Ui",
):
    scripts_dir = pathlib.Path(__file__).expanduser().parent
    frds_dir = scripts_dir.parent
    ui_files_dir = frds_dir.joinpath(ui_files_dir)
    py_files_dir = frds_dir.joinpath(output_files_dir)
    for _, _, filenames in os.walk(ui_files_dir):
        for filename in filenames:
            if not filename.endswith(".ui"):
                continue
            ui_filepath = ui_files_dir.joinpath(filename).as_posix()
            py_filepath = py_files_dir.joinpath(
                f"{prefix}_{filename.replace('.ui', '')}.py"
            ).as_posix()
            subprocess.run(
                ["pyuic5", ui_filepath, "-o", py_filepath, "--from-imports"],
                check=True,
            )


def pyrcc_all_resource_files(
    files_dir="gui/ui_components/resources_files",
    output_files_dir="gui/ui_components/generated_py_files",
    suffix="rc",
):
    scripts_dir = pathlib.Path(__file__).expanduser().parent
    frds_dir = scripts_dir.parent
    files_dir = frds_dir.joinpath(files_dir)
    py_files_dir = frds_dir.joinpath(output_files_dir)
    for _, _, filenames in os.walk(files_dir):
        for filename in filenames:
            if not filename.endswith(".qrc"):
                continue
            ui_filepath = files_dir.joinpath(filename).as_posix()
            py_filepath = py_files_dir.joinpath(
                f"{filename.replace('.qrc', '')}_{suffix}.py"
            ).as_posix()
            subprocess.run(
                ["pyrcc5", ui_filepath, "-o", py_filepath], check=True,
            )


if __name__ == "__main__":
    pyrcc_all_resource_files()
    pyuic_all_ui_files()
