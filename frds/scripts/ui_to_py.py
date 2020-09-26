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
            ui_filepath = ui_files_dir.joinpath(filename).as_posix()
            py_filepath = py_files_dir.joinpath(
                f"{prefix}_{filename.replace('.ui', '')}.py"
            ).as_posix()
            subprocess.run(
                ["pyuic5", ui_filepath, "-o", py_filepath], check=True
            )


if __name__ == "__main__":
    pyuic_all_ui_files()
