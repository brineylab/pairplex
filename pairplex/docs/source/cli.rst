Here are some details about CLI usage of PairPlex.

```
pairplex run sequencing_data output_directory
```

Options are meant to be used as follows:
```
pairplex run --whitelist_path v2 --platform illumina --clustering_threshold 0.9 --min_cluster_reads 3 --min_cluster_umis 2 --merge_paired_reads UDA pairplexed
```

Note: ``--min_cluster_umis`` defaults to ``1``. A value of ``2`` (as used in the example above) is recommended for datasets with heavy ambient contamination.
