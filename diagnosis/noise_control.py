"""Control: identity between reads sharing barcode+UMI (same molecule), same metric."""
import sys, random
import edlib, polars as pl
def hw(a,b):
    s,l=(a,b) if len(a)<=len(b) else (b,a)
    if not s: return 0.0
    return 1-edlib.align(s,l,mode="HW",task="distance")["editDistance"]/len(s)
df=pl.read_parquet(sys.argv[1],columns=["umi","barcode","sequence"])
g=df.group_by("barcode","umi").agg(pl.col("sequence")).filter(pl.col("sequence").list.len()>=2).sample(n=20000,seed=0)
random.seed(0); ids=[]; exact=0
for seqs in g["sequence"]:
    a,b=random.sample(list(seqs),2); ids.append(hw(a,b)); exact+= a==b
s=pl.Series(ids)
print("same-molecule read pairs:",len(ids),"| exact identical:",round(exact/len(ids),3))
print("identity quantiles q05,q10,q25,q50:",[round(s.quantile(q),4) for q in (.05,.1,.25,.5)])
print("frac below 0.98:",round(float((s<0.98).mean()),3),"| below 0.966:",round(float((s<0.966).mean()),3))
# between-UMI pairs drawn from DIFFERENT barcodes (unrelated molecules) for scale
