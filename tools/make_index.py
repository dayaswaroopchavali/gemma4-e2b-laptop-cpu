import json, struct, sys
src=sys.argv[1]; out=sys.argv[2]
with open(src,"rb") as fh:
    n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n))
with open(out,"w") as o:
    for k,v in hdr.items():
        if k=="__metadata__": continue
        a,b=v["data_offsets"]; o.write(f"{k}\t{v['dtype']}\t{8+n+a}\t{b-a}\t{','.join(map(str,v['shape']))}\n")
print("tensors:",len(hdr)-1)
