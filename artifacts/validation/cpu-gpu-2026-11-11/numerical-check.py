import json, hashlib, sys
from pathlib import Path
import numpy as np
results=[]
for filename in sys.argv[1:]:
 report=json.loads(Path(filename).read_text(encoding="utf-8-sig"))
 if report["Status"]!="complete": raise RuntimeError("Incomplete report: "+filename)
 groups={}
 for record in report["Records"]:
  groups.setdefault((record["Model"],record["Format"],record["RequestedBackend"]),[]).append(record)
 for key,records in groups.items():
  ref=next(r for r in records if r["Variant"]=="baseline")
  expected=[np.fromfile(o["Path"],dtype="<f4") for o in ref["Outputs"]]
  max_abs=max_norm=0.; all_exact=True; comparisons=0
  for r in records:
   for phase in ["FirstOutputs","Outputs"]:
    if len(r[phase])!=len(expected): raise AssertionError("Output count changed")
    for index,o in enumerate(r[phase]):
     raw=Path(o["Path"]).read_bytes()
     if hashlib.sha256(raw).hexdigest().lower()!=o["Sha256"].lower(): raise AssertionError("Output hash changed")
     a=np.frombuffer(raw,dtype="<f4"); b=expected[index]
     if o["Shape"]!=ref["Outputs"][index]["Shape"] or a.shape!=b.shape: raise AssertionError("Shape changed")
     d=np.abs(a.astype(np.float64)-b.astype(np.float64))
     if not np.isfinite(a).all() or not (d<=.003+.0003*np.abs(b)).all():
      raise AssertionError(f"Numerical mismatch {key} {r['Sequence']} {phase} {index}: {d.max()}")
     max_abs=max(max_abs,float(d.max(initial=0)));max_norm=max(max_norm,float((d/np.maximum(1,np.abs(b))).max(initial=0)))
     all_exact &= np.array_equal(a.view(np.uint32),b.view(np.uint32)); comparisons+=1
  results.append({"Report":filename,"Model":key[0],"Format":key[1],"Backend":key[2],"Windows":len(records),"OutputComparisons":comparisons,"MaxAbsError":max_abs,"MaxNormalizedError":max_norm,"AllBitIdentical":bool(all_exact),"Passed":True})
output={"Atol":.003,"Rtol":.0003,"Reference":"First baseline window for each case; includes first and steady outputs from both variants","Results":results,"Passed":all(r["Passed"] for r in results)}
Path(".cache/deep-final-numerical.json").write_text(json.dumps(output,indent=2))
for r in results: print(r)
