from __future__ import annotations

import re
import json
import shutil
import subprocess

import itertools as it

from pathlib import Path

import logging

log = logging.getLogger(__name__)


def package_manuscript(
    submission_src_dir: str | Path,
    tex_name: str | Path = "manuscript.tex",
    figures_src_dir_name="figures",
    tables_src_dir_name="tables",
    figures_dest_dir_name="figures",
    delete_existing: bool = True,
):
    submission_src_dir = Path(submission_src_dir).expanduser().resolve()
    tex_name = Path(tex_name)

    target_dir = (
        submission_src_dir.expanduser().parent / f"{submission_src_dir.name}-packaged"
    )

    zip_ofpath = submission_src_dir.resolve().parent / f"{target_dir.name}.zip"

    if delete_existing:
        zip_ofpath.unlink(missing_ok=True)
        shutil.rmtree(target_dir, ignore_errors=True)

    target_dir.mkdir()

    tex_src_path = submission_src_dir / tex_name
    assert tex_src_path.exists()
    target_tex_path = target_dir / tex_name

    tex = tex_src_path.read_text()

    used_old_path_strs = re.findall(r"\\includegraphics\{(.+)\}", tex)
    used_str_path_map = {
        old_path_str: (
            lambda i, old_path: str(
                f"{figures_dest_dir_name}/figure-{i}{old_path.suffix}"
            )
        )(i, Path(old_path_str))
        for i, old_path_str in enumerate(used_old_path_strs, 1)
    }

    rep = {re.escape(k): v for k, v in used_str_path_map.items()}
    pattern = re.compile("|".join(rep))
    tex = pattern.sub(lambda m: rep[re.escape(m.group(0))], tex)

    tex = re.sub(
        rf"(\\input\{{({tables_src_dir_name}/.+)\}})",  # FIXME correct escaping?
        lambda x: (submission_src_dir / f"{x.group(2)}.tex").read_text(),
        tex,
    )

    if used_str_path_map:
        (target_dir / figures_dest_dir_name).mkdir(exist_ok=True)
    for old_path_str, new_path_str in used_str_path_map.items():
        old_path = submission_src_dir / Path(old_path_str)
        new_path = target_dir / Path(new_path_str)
        shutil.copy(old_path, new_path)

    old_bibliography_paths_wo_suffix = re.findall(r"\\bibliography\{([^}]+)\}", tex)
    assert len(old_bibliography_paths_wo_suffix) <= 1
    bib_map = {
        old_path_str: (lambda i, old_path: str(f"bibliography-{i}"))(
            i, Path(old_path_str)
        )
        for i, old_path_str in enumerate(old_bibliography_paths_wo_suffix, 1)
    }

    rep = {re.escape(k): v for k, v in bib_map.items()}
    pattern = re.compile(r"(?<=\\bibliography\{)" + "|".join(rep) + r"(?=\})")
    tex = pattern.sub(lambda m: rep[re.escape(m.group(0))], tex)

    for old_bib_local_path, new_bib_local_path in bib_map.items():
        old_path = submission_src_dir / f"{Path(old_bib_local_path)}.bib"
        new_path = target_dir / f"{Path(new_bib_local_path)}.bib"
        shutil.copy(old_path, new_path)

    for bibtexstyle in submission_src_dir.glob("*.bst"):
        shutil.copy(bibtexstyle, target_dir)

    pattern = re.compile(r"(^[^%]*(?<!\\))(%.*)$", flags=re.MULTILINE)
    # pattern.findall(tex)

    tex = pattern.sub(
        lambda m: m.group(1) if not "ORCID" in m.group(2) else m.group(0), tex
    )

    target_tex_path.write_text(tex)

    subprocess.run(["latexmk", "-pdf"], cwd=target_dir)

    if old_bibliography_paths_wo_suffix:
        bbl_path = target_dir / f"{tex_name.stem}.bbl"
        assert bbl_path.exists()
    assert (target_dir / f"{tex_name.stem}.pdf").exists()

    if old_bibliography_paths_wo_suffix:
        bbl = bbl_path.read_text()

    # breakpoint()

    tex = re.compile(r"\\bibliography\{[^}]+\}").sub(lambda _: bbl, tex)
    target_tex_path.write_text(tex)

    subprocess.run(["latexmk", "-C"], cwd=target_dir)
    assert not (target_dir / f"{tex_name.stem}.pdf").exists()

    subprocess.run(["latexmk", "-pdf"], cwd=target_dir)
    subprocess.run(["latexmk", "-c"], cwd=target_dir)
    assert (target_dir / f"{tex_name.stem}.pdf").exists()

    bbl_files = [*target_dir.glob("*.bbl")]
    bib_files = [*target_dir.glob("*.bib")]
    bst_files = [*target_dir.glob("*.bst")]

    for path in it.chain(bbl_files, bib_files, bst_files):
        path.unlink()

    subprocess.run(["zip", "-r", zip_ofpath, target_dir.resolve()])
    print(f"\nExported to {zip_ofpath}")
