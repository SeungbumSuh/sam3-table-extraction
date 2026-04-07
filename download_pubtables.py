"""Parallel download of PubTables-v2 Single Pages from HuggingFace into a Modal volume.

Downloads the original tar.gz files using **chunked byte-range requests**
spread across many containers (up to ``max-workers``), reassembles the chunks
on the volume, then extracts.  A second wave of containers converts the
PASCAL VOC XML annotations into COCO JSON.

Phases
------
1a. **Probe** — HEAD requests to learn file sizes.
1b. **Download** — each large file is split into ~1–2 GB byte-range chunks;
    every chunk is downloaded by its own container (up to ``max-workers``
    total containers across all files).
1c. **Reassemble + extract** — per file, stream the ordered chunks through
    gzip / tar decompression straight to the volume, then delete chunks.
2.  **Convert** — up to ``max-workers`` containers convert PASCAL VOC XML
    batches into partial COCO JSON shards.
3.  **Merge** — one container per split combines shards into a final
    ``_annotations.coco.json``.

Usage
-----
    modal run download_pubtables.py
    modal run download_pubtables.py --splits train
    modal run download_pubtables.py --collection Cropped-Tables --splits train,val,test
    modal run download_pubtables.py --max-workers 50
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]", "requests")
)

app = modal.App(name="pubtables-download", image=download_image)

dataset_vol = modal.Volume.from_name("pubtables-vol", create_if_missing=True, version=2)
MODAL_DATA_DIR = "/data"

DATASET_ID = "kensho/PubTables-v2"

# Files smaller than this are downloaded in a single request (no chunking).
_CHUNK_THRESHOLD = 500 * 1024 * 1024  # 500 MB

# Modal Volume v2 supports at most 260 000 files per directory.
# We shard extracted files into 256 subdirectories (2-char hex prefix)
# to stay well under that limit (~1 800 files/bucket at 460k total).
_SHARD_HEX_CHARS = 2


def _shard_prefix(filename: str) -> str:
    """Return a deterministic 2-hex-char subdirectory name for *filename*."""
    return hashlib.md5(filename.encode()).hexdigest()[:_SHARD_HEX_CHARS]

# ---------------------------------------------------------------------------
# Phase 1a — probe file sizes via HEAD
# ---------------------------------------------------------------------------


@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=120,
)
def probe_files(filenames: list[str]) -> dict[str, int]:
    """Return ``{filename: size_in_bytes}`` for every file."""
    import requests
    from huggingface_hub import hf_hub_url

    token = os.environ.get("HF_TOKEN", "")
    sizes: dict[str, int] = {}
    for fn in filenames:
        url = hf_hub_url(repo_id=DATASET_ID, filename=fn, repo_type="dataset")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        sizes[fn] = int(resp.headers.get("Content-Length", 0))
        print(f"  {fn}: {sizes[fn] / 1e9:.2f} GB")
    return sizes


# ---------------------------------------------------------------------------
# Phase 1b — download one byte-range chunk
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    memory=4096,
    retries=2,
)
def download_chunk(
    filename: str,
    chunk_idx: int,
    start_byte: int,
    end_byte: int,
    chunk_dir: str,
) -> dict[str, Any]:
    """Download bytes ``[start_byte, end_byte]`` to a numbered chunk file."""
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=DATASET_ID, filename=filename, repo_type="dataset")
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    headers["Range"] = f"bytes={start_byte}-{end_byte}"

    Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    chunk_path = Path(chunk_dir) / f"chunk_{chunk_idx:04d}.bin"

    written = 0
    with requests.get(url, headers=headers, stream=True, timeout=(60, 600)) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(
                f"HTTP {resp.status_code} for {filename} "
                f"range {start_byte}-{end_byte}"
            )
        with open(chunk_path, "wb") as fh:
            for data in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(data)
                written += len(data)

    dataset_vol.commit()
    expected = end_byte - start_byte + 1
    print(f"[{filename}] chunk {chunk_idx}: {written:,} / {expected:,} bytes")
    return {"chunk_idx": chunk_idx, "written": written}


# ---------------------------------------------------------------------------
# Phase 1c — reassemble ordered chunks → tar extraction → cleanup
# ---------------------------------------------------------------------------


class _ChunkedStream:
    """Reads numbered chunk files in order as a single byte stream."""

    def __init__(self, paths: list[Path]):
        self._paths = paths
        self._idx = 0
        self._fh = open(paths[0], "rb") if paths else None

    def read(self, size: int = -1) -> bytes:
        if self._fh is None:
            return b""
        if size == 0:
            return b""
        if size < 0:
            parts: list[bytes] = []
            while self._fh is not None:
                parts.append(self._fh.read())
                self._advance()
            return b"".join(parts)

        buf = bytearray()
        while len(buf) < size and self._fh is not None:
            data = self._fh.read(size - len(buf))
            if data:
                buf.extend(data)
            else:
                self._advance()
        return bytes(buf)

    def _advance(self) -> None:
        if self._fh:
            self._fh.close()
        self._idx += 1
        if self._idx < len(self._paths):
            self._fh = open(self._paths[self._idx], "rb")
        else:
            self._fh = None

    # tarfile's type stubs require these even though streaming mode never
    # calls write / seek / tell.
    def write(self, b: bytes) -> int:
        raise NotImplementedError

    def tell(self) -> int:
        raise NotImplementedError

    def seek(self, pos: int, whence: int = 0) -> int:
        raise NotImplementedError

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=14400,
    memory=4096,
    retries=1,
)
def reassemble_and_extract(
    chunk_dir: str,
    num_chunks: int,
    extract_dir: str,
    return_file_list: bool = False,
) -> dict[str, Any]:
    """Stream ordered chunk files through gzip → tar → volume."""
    import tarfile

    dataset_vol.reload()

    chunk_paths = [
        Path(chunk_dir) / f"chunk_{i:04d}.bin" for i in range(num_chunks)
    ]
    missing = [p for p in chunk_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing chunks: {missing}")

    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    reader = _ChunkedStream(chunk_paths)
    file_names: list[str] = []
    count = 0

    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name.startswith("/") or ".." in member.name:
                    continue
                basename = os.path.basename(member.name)
                shard = _shard_prefix(basename)
                member.name = f"{shard}/{basename}"
                tar.extract(member, path=extract_dir)
                if return_file_list:
                    file_names.append(member.name)
                count += 1
                if count % 10_000 == 0:
                    dataset_vol.commit()
                    print(f"  … {count:,} files extracted")
    finally:
        reader.close()

    for p in chunk_paths:
        p.unlink(missing_ok=True)

    dataset_vol.commit()
    print(f"Done — {count:,} files extracted, chunks cleaned up")

    return {
        "extract_dir": extract_dir,
        "num_files": count,
        "files": file_names,
    }


# ---------------------------------------------------------------------------
# Phase 2 — convert a batch of PASCAL VOC XMLs to partial COCO JSON
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=1800,
    memory=2048,
    retries=1,
)
def convert_xml_batch(
    xml_paths: list[str],
    split: str,
    batch_idx: int,
    output_subdir: str,
    image_id_offset: int,
) -> dict[str, Any]:
    """Parse PASCAL VOC XMLs and write a partial COCO JSON shard to the volume."""
    dataset_vol.reload()

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    category_map: dict[str, int] = {}
    ann_id = 0

    for local_idx, xml_path in enumerate(xml_paths):
        p = Path(xml_path)
        if not p.exists():
            continue

        tree = ET.parse(str(p))
        root = tree.getroot()

        image_id = image_id_offset + local_idx

        filename_el = root.find("filename")
        basename = (
            filename_el.text.strip()
            if filename_el is not None and filename_el.text
            else p.stem + ".jpg"
        )
        file_name = f"{_shard_prefix(basename)}/{basename}"

        size_el = root.find("size")
        width = int(size_el.findtext("width", "0")) if size_el is not None else 0
        height = int(size_el.findtext("height", "0")) if size_el is not None else 0

        coco_images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for obj in root.findall("object"):
            name = obj.findtext("name", "unknown").strip()
            if name not in category_map:
                category_map[name] = len(category_map) + 1

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            w, h = xmax - xmin, ymax - ymin

            ann_id += 1
            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_map[name],
                "bbox": [round(xmin, 2), round(ymin, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })

        if (local_idx + 1) % 2000 == 0:
            print(f"[batch {batch_idx}] {local_idx + 1}/{len(xml_paths)}")

    shards_dir = Path(MODAL_DATA_DIR) / output_subdir / split / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_path = shards_dir / f"shard_{batch_idx:04d}.json"
    with open(shard_path, "w") as f:
        json.dump({
            "batch_idx": batch_idx,
            "images": coco_images,
            "annotations": coco_annotations,
            "category_map": category_map,
        }, f)

    dataset_vol.commit()
    print(
        f"[batch {batch_idx}] done — "
        f"{len(coco_images)} images, {len(coco_annotations)} annotations"
    )
    return {
        "batch_idx": batch_idx,
        "num_images": len(coco_images),
        "num_annotations": len(coco_annotations),
    }


# ---------------------------------------------------------------------------
# Phase 3 — merge per-batch shards into one COCO annotation file
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=3600,
    memory=16384,
)
def merge_annotations(
    output_subdir: str,
    split: str,
    num_batches: int,
) -> dict[str, Any]:
    """Combine all shard JSONs into ``_annotations.coco.json``."""
    dataset_vol.reload()

    shards_dir = Path(MODAL_DATA_DIR) / output_subdir / split / "shards"

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    shard_meta: list[tuple[int, int, dict[str, int]]] = []

    for idx in range(num_batches):
        shard_path = shards_dir / f"shard_{idx:04d}.json"
        if not shard_path.exists():
            print(f"WARNING: missing {shard_path}")
            continue
        with open(shard_path) as f:
            partial = json.load(f)

        ann_start = len(all_annotations)
        all_images.extend(partial["images"])
        all_annotations.extend(partial["annotations"])
        ann_end = len(all_annotations)

        shard_meta.append((ann_start, ann_end, partial["category_map"]))
        print(f"Loaded shard {idx}: {len(partial['images'])} images")

    all_names: set[str] = set()
    for _, _, cat_map in shard_meta:
        all_names.update(cat_map.keys())
    global_name_to_id = {
        name: idx + 1 for idx, name in enumerate(sorted(all_names))
    }

    for ann_start, ann_end, cat_map in shard_meta:
        local_to_global = {
            local_id: global_name_to_id[name]
            for name, local_id in cat_map.items()
        }
        for ann in all_annotations[ann_start:ann_end]:
            ann["category_id"] = local_to_global[ann["category_id"]]

    # Reassign image and annotation IDs sequentially to guarantee uniqueness
    # (shard-local IDs can collide across batches).
    old_to_new_img: dict[int, int] = {}
    for new_id, img in enumerate(all_images, start=1):
        old_to_new_img[img["id"]] = new_id
        img["id"] = new_id

    for new_id, ann in enumerate(all_annotations, start=1):
        ann["id"] = new_id
        ann["image_id"] = old_to_new_img[ann["image_id"]]

    coco_categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(global_name_to_id.items(), key=lambda x: x[1])
    ]

    coco_dataset = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": coco_categories,
    }

    out_dir = Path(MODAL_DATA_DIR) / output_subdir / split
    out_path = out_dir / "_annotations.coco.json"
    with open(out_path, "w") as f:
        json.dump(coco_dataset, f)

    dataset_vol.commit()

    summary = {
        "annotation_path": str(out_path),
        "total_images": len(all_images),
        "total_annotations": len(all_annotations),
        "categories": coco_categories,
    }
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Phase 4 — validate the final COCO JSON against files on disk
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=1800,
    memory=4096,
)
def validate_split(output_subdir: str, split: str) -> dict[str, Any]:
    """Check that the COCO annotation file is consistent with files on disk."""
    dataset_vol.reload()

    base = Path(MODAL_DATA_DIR) / output_subdir / split
    ann_path = base / "_annotations.coco.json"
    if not ann_path.exists():
        return {"ok": False, "error": f"{ann_path} not found"}

    with open(ann_path) as f:
        coco = json.load(f)

    images: list[dict] = coco.get("images", [])
    annotations: list[dict] = coco.get("annotations", [])
    categories: list[dict] = coco.get("categories", [])

    errors: list[str] = []

    # Unique IDs
    img_ids = [img["id"] for img in images]
    ann_ids = [ann["id"] for ann in annotations]
    if len(img_ids) != len(set(img_ids)):
        errors.append(
            f"Duplicate image IDs: {len(img_ids)} total, "
            f"{len(set(img_ids))} unique"
        )
    if len(ann_ids) != len(set(ann_ids)):
        errors.append(
            f"Duplicate annotation IDs: {len(ann_ids)} total, "
            f"{len(set(ann_ids))} unique"
        )

    # Referential integrity: every annotation points to a valid image
    img_id_set = set(img_ids)
    orphan_anns = [a for a in annotations if a["image_id"] not in img_id_set]
    if orphan_anns:
        errors.append(
            f"{len(orphan_anns)} annotations reference non-existent image IDs"
        )

    # Category integrity: every annotation uses a valid category
    cat_id_set = {c["id"] for c in categories}
    bad_cats = [a for a in annotations if a["category_id"] not in cat_id_set]
    if bad_cats:
        errors.append(
            f"{len(bad_cats)} annotations reference non-existent category IDs"
        )

    # Image files on disk vs COCO image entries (file_name is a shard-relative
    # path like "a3/PMC1234.jpg", so compare against paths relative to images/).
    img_dir = base / "images"
    if img_dir.is_dir():
        disk_files = {
            str(f.relative_to(img_dir))
            for f in img_dir.rglob("*") if f.is_file()
        }
        coco_files = {img["file_name"] for img in images}
        missing_on_disk = coco_files - disk_files
        if missing_on_disk:
            errors.append(
                f"{len(missing_on_disk)} images listed in COCO JSON "
                f"but missing on disk (e.g. {list(missing_on_disk)[:3]})"
            )
        extra_on_disk = disk_files - coco_files
        if extra_on_disk:
            errors.append(
                f"{len(extra_on_disk)} image files on disk "
                f"not listed in COCO JSON"
            )

    ok = len(errors) == 0
    result = {
        "ok": ok,
        "split": split,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "errors": errors,
    }

    if ok:
        print(f"[{split}] VALID — {len(images):,} images, "
              f"{len(annotations):,} annotations, "
              f"{len(categories)} categories")
    else:
        for e in errors:
            print(f"[{split}] ERROR: {e}")

    return result


# ---------------------------------------------------------------------------
# Helpers for the local entrypoint
# ---------------------------------------------------------------------------


def _build_download_plan(
    file_sizes: dict[str, int],
    file_extract_dirs: dict[str, str],
    file_needs_list: dict[str, bool],
    max_workers: int,
) -> tuple[
    list[tuple[str, int, int, int, str]],
    list[tuple[str, int, str, bool]],
]:
    """Compute chunk assignments that fit within *max_workers* containers.

    Returns ``(chunk_args, extract_args)`` where:
    - ``chunk_args`` feeds :func:`download_chunk` via ``starmap``
    - ``extract_args`` feeds :func:`reassemble_and_extract` via ``starmap``
    """
    total_bytes = sum(file_sizes.values())
    if total_bytes == 0:
        return [], []

    target_chunk_bytes = max(total_bytes // max(max_workers, 1), 50 * 1024 * 1024)

    chunk_args: list[tuple[str, int, int, int, str]] = []
    extract_args: list[tuple[str, int, str, bool]] = []

    for filename, size in file_sizes.items():
        chunk_dir = f"{MODAL_DATA_DIR}/_chunks/{filename}"
        extract_dir = file_extract_dirs[filename]
        needs_list = file_needs_list[filename]

        if size < _CHUNK_THRESHOLD:
            num_chunks = 1
        else:
            num_chunks = max(1, math.ceil(size / target_chunk_bytes))

        actual_chunk = math.ceil(size / num_chunks) if num_chunks > 0 else size

        for i in range(num_chunks):
            start = i * actual_chunk
            end = min(start + actual_chunk - 1, size - 1)
            if start > size - 1:
                break
            chunk_args.append((filename, i, start, end, chunk_dir))

        extract_args.append((chunk_dir, num_chunks, extract_dir, needs_list))

    return chunk_args, extract_args


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    collection: str = "Single-Pages",
    splits: str = "train,val,test",
    output_subdir: str = "pubtables_v2_single_pages",
    max_workers: int = 100,
    file_types: str = "images,xml-annotations",
):
    """Download a PubTables-v2 collection into the pubtables-vol Modal volume.

    Parameters
    ----------
    collection : Which PubTables-v2 collection to download.
                 One of Single-Pages, Cropped-Tables, Full-Documents.
    splits : Comma-separated list of splits (train, val, test).
    output_subdir : Target directory under /data on the volume.
    max_workers : Maximum parallel containers for both the download phase
                  and the XML → COCO conversion phase.
    file_types : Comma-separated tar types to download
                 (images, xml-annotations, tables, words).
    """
    split_list = [s.strip() for s in splits.split(",")]
    type_list = [t.strip() for t in file_types.split(",")]

    print(f"{'Collection':>14}: {collection}")
    print(f"{'Splits':>14}: {', '.join(split_list)}")
    print(f"{'File types':>14}: {', '.join(type_list)}")
    print(f"{'Max workers':>14}: {max_workers}")
    print(f"{'Volume path':>14}: /data/{output_subdir}")
    print()

    # -- build list of tar.gz files ----------------------------------------
    tar_filenames: list[str] = []
    extract_dirs: dict[str, str] = {}
    needs_list: dict[str, bool] = {}

    for split in split_list:
        for ftype in type_list:
            fn = f"PubTables-v2_{collection}_{split}_{ftype}.tar.gz"
            tar_filenames.append(fn)
            extract_dirs[fn] = (
                f"{MODAL_DATA_DIR}/{output_subdir}/{split}/"
                f"{ftype.replace('-', '_')}"
            )
            needs_list[fn] = ftype == "xml-annotations"

    # -- Phase 1a: probe sizes ---------------------------------------------
    print("Phase 1a: probing file sizes …")
    file_sizes = probe_files.remote(tar_filenames)
    total_gb = sum(file_sizes.values()) / 1e9
    print(f"  Total: {total_gb:.1f} GB across {len(file_sizes)} files\n")

    # -- Phase 1b: chunked parallel download --------------------------------
    chunk_args, extract_args = _build_download_plan(
        file_sizes, extract_dirs, needs_list, max_workers,
    )

    print(
        f"Phase 1b: downloading via {len(chunk_args)} parallel chunk "
        f"containers …"
    )
    list(download_chunk.starmap(chunk_args))
    print("  All chunks downloaded.\n")

    # -- Phase 1c: reassemble + extract ------------------------------------
    print(
        f"Phase 1c: reassembling & extracting {len(extract_args)} tar.gz "
        f"files in parallel …"
    )
    extraction_results = list(reassemble_and_extract.starmap(extract_args))

    extraction_by_dir: dict[str, dict[str, Any]] = {}
    for r in extraction_results:
        print(f"  {r['extract_dir']}: {r['num_files']:,} files")
        extraction_by_dir[r["extract_dir"]] = r

    # -- Phase 2 + 3: XML → COCO conversion, then merge --------------------
    for split in split_list:
        xml_dir = (
            f"{MODAL_DATA_DIR}/{output_subdir}/{split}/xml_annotations"
        )
        xml_result = extraction_by_dir.get(xml_dir)
        if xml_result is None or xml_result["num_files"] == 0:
            print(f"\nSkipping {split} — no XML annotations extracted")
            continue

        xml_files = [
            f"{xml_dir}/{name}"
            for name in xml_result["files"]
            if name.endswith(".xml")
        ]
        if not xml_files:
            print(f"\nSkipping {split} — no .xml files found")
            continue

        num_workers = min(max_workers, len(xml_files))
        batch_size = math.ceil(len(xml_files) / num_workers)

        batch_args: list[tuple[list[str], str, int, str, int]] = []
        for i in range(num_workers):
            start = i * batch_size
            end = min(start + batch_size, len(xml_files))
            if start >= len(xml_files):
                break
            batch_args.append((
                xml_files[start:end],
                split,
                i,
                output_subdir,
                i * batch_size,
            ))

        num_batches = len(batch_args)
        print(
            f"\nPhase 2 [{split}]: converting {len(xml_files):,} XMLs "
            f"across {num_batches} containers …"
        )
        convert_results = list(convert_xml_batch.starmap(batch_args))

        total_imgs = sum(r["num_images"] for r in convert_results)
        total_anns = sum(r["num_annotations"] for r in convert_results)
        print(f"  Converted {total_imgs:,} images, {total_anns:,} annotations")

        print(f"Phase 3 [{split}]: merging annotations …")
        summary = merge_annotations.remote(output_subdir, split, num_batches)

        print(f"  → {summary['annotation_path']}")
        print(f"    {summary['total_images']:,} images")
        print(f"    {summary['total_annotations']:,} annotations")
        for cat in summary["categories"]:
            print(f"    [{cat['id']}] {cat['name']}")

        print(f"Phase 4 [{split}]: validating …")
        vresult = validate_split.remote(output_subdir, split)
        if not vresult["ok"]:
            print(f"  VALIDATION FAILED for {split}:")
            for e in vresult["errors"]:
                print(f"    • {e}")
        else:
            print(f"  Validation passed.")

    print(
        f"\nDone! Data is in Modal volume 'pubtables-vol' "
        f"at /data/{output_subdir}"
    )
