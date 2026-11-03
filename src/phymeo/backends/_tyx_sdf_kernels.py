"""Eight-node FP32 trilinear interpolation and analytic derivatives on TYX."""

import tyx as tx


@tx.func
def _mix(a: tx.f32,b: tx.f32,t: tx.f32)->tx.f32:
    return a+(b-a)*t


@tx.kernel
def sample_sdf_nodes(count:tx.i32,nx:tx.i32,nxy:tx.i32,generation:tx.i32,
                     inv_x:tx.f32,inv_y:tx.f32,inv_z:tx.f32,
                     nodes:tx.ndarray(tx.f32,2),addresses:tx.ndarray(tx.i32,1),
                     fraction:tx.ndarray(tx.f32,2),out:tx.ndarray(tx.f32,2)):
    for i in tx.range(count):
        base=addresses[i]
        code=1
        for k in tx.static_range(4):
            out[i,k]=0.0
        if base>=0:
            code=0
            for z in tx.static_range(2):
                for y in tx.static_range(2):
                    for x in tx.static_range(2):
                        if nodes[base+z*nxy+y*nx+x,1]==0.0:
                            code=2
            if code==0:
                a=nodes[base,0]
                b=nodes[base+1,0]
                c=nodes[base+nx,0]
                d=nodes[base+nx+1,0]
                e=nodes[base+nxy,0]
                f=nodes[base+nxy+1,0]
                g=nodes[base+nxy+nx,0]
                h=nodes[base+nxy+nx+1,0]
                u=fraction[i,0]
                v=fraction[i,1]
                w=fraction[i,2]
                ab=_mix(a,b,u)
                cd=_mix(c,d,u)
                ef=_mix(e,f,u)
                gh=_mix(g,h,u)
                near=_mix(ab,cd,v)
                far=_mix(ef,gh,v)
                out[i,0]=_mix(near,far,w)
                out[i,1]=_mix(_mix(b-a,d-c,v),_mix(f-e,h-g,v),w)*inv_x
                out[i,2]=_mix(cd-ab,gh-ef,w)*inv_y
                out[i,3]=(far-near)*inv_z
        # Outside-grid input uses -1, and every in-grid input is nonnegative.
        # This negative completion marker cannot appear in an unprocessed row.
        addresses[i]=-(4*generation+code+1)
