"""Scorer: evaluates a PairPlex output against SimPlex ground truth.

Final stage of the cells -> molecules -> routing -> reads -> truth -> scoring pipeline.
Reads the `truth_components`/`truth_barcodes` tables written by `truth.py` (via
`io.write_truth`) and one or more PairPlex paired-output parquets, and produces
`pair_scores` (one row per PairPlex-returned pair) and `key_scores` (one row per truth
`(well, barcode)` key, including keys PairPlex returned nothing for) per the frozen
scoring contract (design doc S6).
"""
import re
from pathlib import Path
import polars as pl
from .matching import candidates, resolve
from ._log import configure_logging, logger, pbar
_LIGHT=("IGK","IGL")

def _files(x):
    """Normalize `x` (a single path, a directory, or a list of paths) into a list of
    paired-output parquet paths. A directory is globbed for `**/*_paired.parquet`.
    """
    if isinstance(x,(list,tuple)): return [Path(p) for p in x]
    x=Path(x)
    return sorted(x.glob("**/*_paired.parquet")) if x.is_dir() else [x]

def _bc(sid): return re.split(r"_contig",sid)[0] if sid else sid

def _well_val(v):
    """Best-effort int-cast of a raw `well` cell value; returns None if missing/blank/unparseable."""
    if v is None: return None
    try:
        s=str(v).strip()
        if s=="" or s.lower()=="null": return None
        return int(float(s))
    except (ValueError,TypeError): return None

def _well_for(row, fname_well, f):
    """Resolve the well for one PairPlex output row: prefer an explicit `well` column
    value; otherwise fall back to the well number parsed from the output filename
    (real merged PairPlex output has no `well` column, only a `well<digits>` filename
    token). Raises `ValueError` if neither is available.
    """
    w=_well_val(row.get("well"))
    if w is not None: return w
    if fname_well is not None: return fname_well
    raise ValueError(f"cannot derive well for {f}: no usable 'well' value on the row and filename has no 'well<digits>' token")

def _index(comp):
    """Build a `{(final_well, barcode): {locus: {sequence: {source_pair_id, ...}}}}` lookup
    from `truth_components`, used by `candidates()` to restrict matching to sources
    actually observed at a given key.
    """
    idx={}
    for r in comp.iter_rows(named=True):
        e=idx.setdefault((int(r["final_well"]),r["barcode"]),{}).setdefault(r["locus"],{})
        e.setdefault(r["sequence"],set()).add(r["source_pair_id"])
    return idx

def _lights(seq, entry):
    """Union of `candidates()` matches for `seq` across both light loci (IGK, IGL)."""
    out=set()
    for L in _LIGHT: out|=candidates(seq,L,entry)
    return out

def _classify_origin(valid_assignments, resident):
    """Classify `origin_status` from a set of valid `(heavy_source, light_source)`
    assignments against the set of resident source ids at this key: `resident` if every
    assignment has both sources resident, `ambient` if every assignment has both
    non-resident, `resident_plus_ambient` if every assignment mixes, `ambiguous` if
    assignments disagree, `unknown` if there are no assignments.
    """
    cats=set()
    for h,l in valid_assignments:
        hr,lr=h in resident,l in resident
        cats.add("resident" if hr and lr else "ambient" if not hr and not lr else "resident_plus_ambient")
    return cats.pop() if len(cats)==1 else ("ambiguous" if cats else "unknown")

def score(pairplex_output, truth_dir, *, pairplex_metadata=None, quiet=False, verbose=False):
    """Score one or more PairPlex paired-output parquets against SimPlex truth.

    `pairplex_output` may be a single file, a list of files, or a directory (globbed for
    `**/*_paired.parquet`); all files are scored jointly against the whole truth in one
    pass (scoring per-file would wrongly mark every other well's truth keys as `missing`).
    `truth_dir` must contain `truth_components.parquet` and `truth_barcodes.parquet` (as
    written by `io.write_truth`). `pairplex_metadata` is accepted but currently unused
    (reserved for future `no_output_reason` refinement). `quiet` suppresses the progress bar
    and drops logging to WARNING; `verbose` enables DEBUG-level logging.

    For each returned pair, tries both (chain0, chain1) orientations against truth loci
    (IGH vs IGK/IGL) — never trusting PairPlex's own chain assignment — and resolves
    pairing status via `matching.resolve`. If both orientations yield results with the
    same `valid_assignments`, that shared result is used; if they disagree, the pair is
    `pairing_status="ambiguous"`/`origin_status="ambiguous"` (genuinely orientation-
    ambiguous). `origin_status` is derived from whether the resolved source(s) were
    physically resident at this `(well, barcode)` key. `output_status` is `duplicate` if
    more than one returned pair maps to the same key, else `unique`.

    Returns `(pair_scores, key_scores)`: `pair_scores` has one row per PairPlex-returned
    pair (empty-safe: a typed empty frame if no pairs were read); `key_scores` has one row
    per truth `(well, barcode)` key from `truth_barcodes` — including keys PairPlex
    returned nothing for (`output_status="missing"`) — carrying `key_status`
    (singleton/collision/ambient_only), observability flags (`captured_both`,
    `survived_both`, `sequenced_both`, `reference_pairable_both`), and `output_count`.
    """
    truth_dir=Path(truth_dir)
    comp=pl.read_parquet(truth_dir/"truth_components.parquet")
    tbar=pl.read_parquet(truth_dir/"truth_barcodes.parquet")
    idx=_index(comp)
    resident_at={}
    for r in comp.filter(pl.col("is_resident_source")).iter_rows(named=True):
        resident_at.setdefault((int(r["final_well"]),r["barcode"]),set()).add(r["source_pair_id"])
    kstat={(int(r["well"]),r["barcode"]):("collision" if r["is_collision"] else "ambient_only" if r["is_ambient_only"] else "singleton") for r in tbar.iter_rows(named=True)}

    configure_logging(quiet=quiet, verbose=verbose)
    files=_files(pairplex_output)
    logger.info("scoring %d PairPlex paired-output file(s) against truth %s (%d truth keys)",
                len(files), truth_dir, tbar.height)
    rows,seen=[],{}
    for f in pbar(files, desc="score", quiet=quiet):
        df=pl.read_parquet(f)
        logger.debug("scoring %s (%d pairs)", Path(f).name, df.height)
        m=re.search(r"well(\d+)",Path(f).name); fname_well=int(m.group(1)) if m else None
        for r in df.to_dicts():
            well=_well_for(r,fname_well,f); bc=_bc(r.get("sequence_id:0") or r.get("name","")); key=(well,bc); entry=idx.get(key)
            s0,s1=r.get("sequence:0"),r.get("sequence:1")
            res_here=resident_at.get(key,set())
            # try BOTH orientations against TRUTH loci (never trust PairPlex's annotation)
            results=[]
            for hseq,lseq in ((s0,s1),(s1,s0)):                  # (heavy,light) candidate orientation
                h=candidates(hseq,"IGH",entry); l=_lights(lseq,entry)
                if h and l: results.append(resolve(h,l))
            if not results:
                pstat,sres,resolved,origin=("unmatchable","none",None,"unknown")
            elif len(results)==1 or results[0].valid_assignments==results[1].valid_assignments:
                r0=results[0]; pstat,sres,resolved=r0.pairing_status,r0.source_resolution,r0.resolved_source
                origin=_classify_origin(r0.valid_assignments,res_here)
            else:                                                # two orientations, incompatible interpretations
                pstat,sres,resolved,origin=("ambiguous","none",None,"ambiguous")
            seen[key]=seen.get(key,0)+1
            rows.append({"pair_id":f"{f.stem}:{r.get('sequence_id:0')}","source_file":str(f),
                "well":well,"barcode":bc,"sequence_id:0":r.get("sequence_id:0"),"sequence_id:1":r.get("sequence_id:1"),
                "pairing_status":pstat,"source_resolution":sres,"origin_status":origin,
                "key_status":kstat.get(key,"unknown"),"output_status":"unique","resolved_source":resolved})
    for pr in rows:
        if seen[(pr["well"],pr["barcode"])]>1: pr["output_status"]="duplicate"
    pair_scores=pl.DataFrame(rows) if rows else pl.DataFrame(schema={c:pl.Utf8 for c in
        ["pair_id","source_file","barcode","sequence_id:0","sequence_id:1","pairing_status","source_resolution","origin_status","key_status","output_status","resolved_source"]}|{"well":pl.Int64})

    key_rows=[]
    for r in tbar.iter_rows(named=True):
        well,bc=int(r["well"]),r["barcode"]; oc=seen.get((well,bc),0)
        key_rows.append({"well":well,"barcode":bc,
            "key_status":("collision" if r["is_collision"] else "ambient_only" if r["is_ambient_only"] else "singleton"),
            "output_count":oc,"output_status":("missing" if oc==0 else "unique" if oc==1 else "duplicate"),
            "n_resident_cells":r.get("n_resident_cells",0),
            "captured_both":r.get("n_captured_both_resident_cells",0)>0,
            "survived_both":r.get("n_survived_both_resident_cells",0)>0,
            "sequenced_both":r.get("n_sequenced_both_resident_cells",0)>0,
            "reference_pairable_both":r.get("n_reference_pairable_resident_cells",0)>0,
            "no_output_reason":None if oc>0 else "unknown"})
    key_scores=pl.DataFrame(key_rows)
    if pair_scores.height:
        vc={s: int((pair_scores["pairing_status"]==s).sum()) for s in ("correct","mispaired","unmatchable","ambiguous")}
        missing=int((key_scores["output_status"]=="missing").sum()) if key_scores.height else 0
        logger.info("scored %d pairs: correct=%d mispaired=%d unmatchable=%d ambiguous=%d | %d truth key(s) with no output",
                    pair_scores.height, vc["correct"], vc["mispaired"], vc["unmatchable"], vc["ambiguous"], missing)
    else:
        logger.info("scored 0 pairs (no PairPlex output matched)")
    return pair_scores, key_scores
