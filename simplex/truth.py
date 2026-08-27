"""Ground-truth table construction: the mechanism-faithful truth consumed by `scoring.score`.

Aggregates the `cells`/`chain_status`/`molecules`/`reads` frames produced earlier in the
cells -> molecules -> routing -> reads -> truth -> scoring pipeline into the three
ground-truth tables written by `io.write_truth`: `truth_components` (per
`(final_well, barcode, origin_cell_id, chain)` observed support), `truth_cells` (per-cell
capture/survival/read summary), and `truth_barcodes` (per `(well, barcode)` occupancy,
collision, and dominance summary).
"""
import polars as pl
from ._contract import REF_MIN_READS, REF_MIN_UMIS

# full aggregate schema so a completely read-less run still yields a valid (empty) component table
_COMP_AGG_SCHEMA={"final_well":pl.Int64,"barcode":pl.Utf8,"origin_cell_id":pl.Int64,"chain":pl.Int8,
    "source_pair_id":pl.Utf8,"locus":pl.Utf8,"n_reads":pl.Int64,"n_reads_resident":pl.Int64,
    "n_reads_free":pl.Int64,"n_reads_index_hopped":pl.Int64,"n_umis":pl.Int64,"n_source_molecules":pl.Int64}

def _cc_seq(cells):
    """Reshape `cells` (one row per cell, two chain columns) into long form: one row per
    `(origin_cell_id, chain)` with that chain's sequence/locus and the cell's home
    `resident_well`/`barcode`. Used to attach ground-truth sequence/residency onto the
    read-derived aggregate in `build_truth_components`.
    """
    parts=[]
    for ch in (0,1):
        parts.append(cells.select([pl.col("cell_id").alias("origin_cell_id"),pl.lit(ch).cast(pl.Int8).alias("chain"),
            pl.col(f"chain{ch}_seq").alias("sequence"),pl.col(f"chain{ch}_locus").alias("locus"),
            pl.col("resident_well"),pl.col("barcode").alias("home_barcode")]))
    return pl.concat(parts)

def build_truth_components(cells,reads):
    """Build `truth_components`: one row per `(final_well, barcode, origin_cell_id, chain)`
    observed in `reads`, with read/UMI/molecule support split by resident/free/index-hopped,
    joined against the cell's ground-truth sequence/locus.

    `is_resident_source` is True only when both the destination `(final_well, barcode)`
    equals the cell's home `(resident_well, barcode)` — i.e. this row is genuinely the
    cell's resident contribution, not an ambient or hopped one that happened to land
    elsewhere. Empty-safe: if `reads` is empty, aggregates from a typed empty schema
    (`_COMP_AGG_SCHEMA`) so the result is still a valid (empty) component table.
    """
    cs=_cc_seq(cells)
    if reads.height==0:
        agg=pl.DataFrame(schema=_COMP_AGG_SCHEMA)
    else:
        agg=reads.group_by(["final_well","barcode","origin_cell_id","chain"]).agg([
            pl.col("source_pair_id").first(),pl.col("locus").first(),pl.len().alias("n_reads"),
            (~pl.col("is_free")&~pl.col("is_index_hopped")).sum().alias("n_reads_resident"),
            pl.col("is_free").sum().alias("n_reads_free"),pl.col("is_index_hopped").sum().alias("n_reads_index_hopped"),
            pl.col("umi").n_unique().alias("n_umis"),
            (pl.col("molecule_id").n_unique() if "molecule_id" in reads.columns else pl.col("umi").n_unique()).alias("n_source_molecules")])
    comp=agg.join(cs.select(["origin_cell_id","chain","sequence","resident_well","home_barcode"]),on=["origin_cell_id","chain"],how="left")
    return comp.with_columns(((pl.col("resident_well")==pl.col("final_well"))&(pl.col("home_barcode")==pl.col("barcode"))).alias("is_resident_source")).drop(["resident_well","home_barcode"])

def build_truth_cells(cells,chain_status,molecules,reads):
    """Build `truth_cells`: one row per cell with per-chain capture/survival/read-support
    columns pivoted wide (suffixed `_0`/`_1`).

    Joins `chain_status` (captured, n_molecules from `molecules.generate_molecules`) with
    per-chain survival (any surviving molecule) and per-chain read/UMI counts from `reads`
    (empty-safe: uses a typed empty frame if there are no reads at all), then pivots on
    `chain` so each cell is one row with both chains' stats side by side.
    """
    surv=(molecules.filter(pl.col("survived")).group_by(["origin_cell_id","chain"]).len().rename({"origin_cell_id":"cell_id","len":"sn"}))
    if reads.height:
        rc=reads.group_by(["origin_cell_id","chain"]).agg([pl.len().alias("n_reads_generated"),
            (~pl.col("is_free")).sum().alias("n_reads_resident"),pl.col("is_free").sum().alias("n_reads_free_out"),
            pl.col("is_index_hopped").sum().alias("n_reads_index_hopped_out"),pl.col("umi").n_unique().alias("n_umis")]).rename({"origin_cell_id":"cell_id"})
    else:
        rc=pl.DataFrame(schema={"cell_id":pl.Int64,"chain":pl.Int8,"n_reads_generated":pl.Int64,"n_reads_resident":pl.Int64,"n_reads_free_out":pl.Int64,"n_reads_index_hopped_out":pl.Int64,"n_umis":pl.Int64})
    st=(chain_status.join(surv,on=["cell_id","chain"],how="left").with_columns((pl.col("sn").fill_null(0)>0).alias("survived"))
        .join(rc,on=["cell_id","chain"],how="left").fill_null(0))
    wide=st.pivot(index="cell_id",on="chain",values=["captured","survived","n_molecules","n_umis",
        "n_reads_generated","n_reads_resident","n_reads_free_out","n_reads_index_hopped_out"])
    return cells.join(wide,on="cell_id",how="left")

def build_truth_barcodes(cells,truth_cells,components):
    """Build `truth_barcodes`: one row per `(well, barcode)` key, the scorer's key-level
    ground truth.

    The key set is the **union** of physical resident keys `(resident_well, barcode)`
    from `cells` and observed keys `(final_well, barcode)` from `components` — physical
    occupancy must come from `cells`, never only from observed components, otherwise a
    resident cell that produced no read would silently vanish (undercounting collisions,
    mis-labeling a key `ambient_only`). Computes `n_resident_cells`, `is_collision`
    (>=2 resident cells), `is_ambient_only` (observed key with 0 resident cells), and
    per-resident-cell-pair collision counts (`n_captured_both_resident_cells`,
    `n_survived_both_resident_cells`, `n_sequenced_both_resident_cells`,
    `n_reference_pairable_resident_cells`, the last using the frozen `REF_MIN_READS`/
    `REF_MIN_UMIS` thresholds) plus per-locus dominant-source columns (by reads and by
    UMIs, for heavy and light separately) with tie detection. Support is aggregated by
    `source_pair_id` first so clonal copies across cells sum before dominance is chosen.
    """
    physical=cells.select([pl.col("resident_well").alias("well"),pl.col("barcode"),pl.col("cell_id"),pl.col("source_pair_id")])
    occ=physical.group_by(["well","barcode"]).agg([pl.col("source_pair_id").unique().alias("resident_source_ids"),
        pl.col("cell_id").n_unique().alias("n_resident_cells")])
    # capture/survival per resident cell at home key (join truth_cells onto physical)
    tc=truth_cells.select(["cell_id","captured_0","captured_1","survived_0","survived_1"])
    cap=physical.join(tc,on="cell_id",how="left").with_columns([
        (pl.col("captured_0")&pl.col("captured_1")).alias("cap_both"),
        (pl.col("captured_0")&pl.col("captured_1")&pl.col("survived_0")&pl.col("survived_1")).alias("surv_both")])
    capk=cap.group_by(["well","barcode"]).agg([pl.col("cap_both").sum().alias("n_captured_both_resident_cells"),
        pl.col("surv_both").sum().alias("n_survived_both_resident_cells")])
    # sequenced/reference per resident cell at home key from components (resident-source rows are AT home)
    res=components.filter(pl.col("is_resident_source"))
    # observability uses TOTAL observable support at the home key (n_reads/n_umis), not only
    # cell-associated reads: a free molecule of the same cell that returns home is legit support.
    per=res.group_by([pl.col("final_well").alias("well"),"barcode","origin_cell_id"]).agg([
        (pl.col("chain").n_unique()==2).alias("seq_both"),
        ((pl.col("n_reads").min()>=REF_MIN_READS)&(pl.col("n_umis").min()>=REF_MIN_UMIS)&(pl.col("chain").n_unique()==2)).alias("ref_both")])
    seqk=per.group_by(["well","barcode"]).agg([pl.col("seq_both").sum().alias("n_sequenced_both_resident_cells"),
        pl.col("ref_both").sum().alias("n_reference_pairable_resident_cells")])
    observed=components.select([pl.col("final_well").alias("well"),pl.col("barcode")]).unique()
    keys=pl.concat([occ.select(["well","barcode"]),observed]).unique()
    tb=keys.join(occ,on=["well","barcode"],how="left").join(capk,on=["well","barcode"],how="left").join(seqk,on=["well","barcode"],how="left")
    # per-locus dominance with tie detection, by reads and umis
    def dom(loci,by,name):
        col=f"dominant_{name}_source_by_{by.replace('n_','')}"; tie=f"{name}_dominance_is_tied_{by}"
        # aggregate support by source_pair_id FIRST (clonal copies across cells sum, not split)
        f=(components.filter(pl.col("locus").is_in(loci))
             .group_by([pl.col("final_well").alias("well"),"barcode","source_pair_id"]).agg(pl.col(by).sum().alias("supp")))
        g=(f.group_by(["well","barcode"]).agg([
             pl.col("source_pair_id").sort_by("supp",descending=True).alias("srcs"),
             pl.col("supp").sort(descending=True).alias("vals")]))
        return g.with_columns([pl.col("srcs").list.first().alias(col),
            ((pl.col("vals").list.len()>1)&(pl.col("vals").list.get(0,null_on_oob=True)==pl.col("vals").list.get(1,null_on_oob=True))).fill_null(False).alias(tie)]) \
            .select(["well","barcode",col,tie])
    for loci,name in [(["IGH"],"heavy"),(["IGK","IGL"],"light")]:
        for by in ("n_reads","n_umis"):
            tb=tb.join(dom(loci,by,name),on=["well","barcode"],how="left")
    return tb.with_columns([pl.col("n_resident_cells").fill_null(0),
        pl.col("n_captured_both_resident_cells").fill_null(0),pl.col("n_survived_both_resident_cells").fill_null(0),
        pl.col("n_sequenced_both_resident_cells").fill_null(0),pl.col("n_reference_pairable_resident_cells").fill_null(0),
        (pl.col("n_resident_cells").fill_null(0)>=2).alias("is_collision"),
        (pl.col("n_resident_cells").fill_null(0)==0).alias("is_ambient_only")])
