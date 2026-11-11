"""Bounded finite segment--triangle feature enumeration; no HOST search."""
import tyx as tx


@tx.kernel
def triangle_distances(
    count: tx.i32,
    generation: tx.i32,
    geometry: tx.ndarray(tx.f32,3),
    radii: tx.ndarray(tx.f32,1),
    scales: tx.ndarray(tx.f32,1),
    owners: tx.ndarray(tx.i32,2),
    values: tx.ndarray(tx.f32,3),
    valid: tx.ndarray(tx.i32,2),
    summary: tx.ndarray(tx.f32,2),
    meta: tx.ndarray(tx.i32,2),
):
    for row in tx.range(count):
        for c in range(18):
            valid[row,c]=0
            for k in range(11):
                values[row,c,k]=0.0
        for k in range(16):
            summary[row,k]=0.0
        for k in range(7):
            meta[row,k]=0
        meta[row,4]=generation
        meta[row,5]=owners[row,0]
        meta[row,6]=owners[row,1]
        length=scales[row]
        # Normalize arithmetic by the original pair-local length.
        # Coordinates are relative to this input pair's original first vertex.
        bx=(geometry[row,1,0]/length)-(geometry[row,0,0]/length)
        by=(geometry[row,1,1]/length)-(geometry[row,0,1]/length)
        bz=(geometry[row,1,2]/length)-(geometry[row,0,2]/length)
        cx=(geometry[row,2,0]/length)-(geometry[row,0,0]/length)
        cy=(geometry[row,2,1]/length)-(geometry[row,0,1]/length)
        cz=(geometry[row,2,2]/length)-(geometry[row,0,2]/length)
        nx=by*cz-bz*cy
        ny=bz*cx-bx*cz
        nz=bx*cy-by*cx
        nn=nx*nx+ny*ny+nz*nz
        px=(geometry[row,3,0]/length)
        py=(geometry[row,3,1]/length)
        pz=(geometry[row,3,2]/length)
        ax=(geometry[row,4,0]/length)-px
        ay=(geometry[row,4,1]/length)-py
        az=(geometry[row,4,2]/length)-pz
        aa=ax*ax+ay*ay+az*az
        length2=length*length
        condition=1.9073486328125e-6
        tie_error=9.5367431640625e-7
        range_ok=1
        for point in range(5):
            for axis in range(3):
                if tx.abs(geometry[row,point,axis])*4.76837158203125e-7>1.0e-4*length:
                    range_ok=0
        state=1
        if nn==0.0:
            state=3
        elif nn<=condition:
            state=2
        if range_ok==0:
            state=2
        if nn>condition and range_ok!=0:
            for candidate in range(18):
                s=0.0
                w0=0.0
                w1=0.0
                w2=0.0
                allowed=0
                rx=0.0
                ry=0.0
                rz=0.0
                ex=0.0
                ey=0.0
                ez=0.0
                dx=0.0
                dy=0.0
                dz=0.0
                denominator=0.0
                structural=0
                if candidate<3:
                    if candidate<2:
                        if candidate==1:
                            s=1.0
                        allowed=1
                    else:
                        v0=nx*(px-(geometry[row,0,0]/length))+ny*(py-(geometry[row,0,1]/length))+nz*(pz-(geometry[row,0,2]/length))
                        denominator=nx*ax+ny*ay+nz*az
                        v1=v0+denominator
                        if denominator*denominator>condition*nn*aa:
                            s=-v0/denominator
                            if s>=0.0 and s<=1.0:
                                allowed=1
                        else:
                            # Zero products give a structural parallel branch.
                            # A rounded zero sum by itself is not such evidence.
                            structural=0
                            if (nx==0.0 or ax==0.0) and (ny==0.0 or ay==0.0) and (nz==0.0 or az==0.0):
                                structural=1
                            separated=0
                            if v0*v1>0.0 and tx.min(tx.abs(v0),tx.abs(v1))>condition*tx.sqrt(nn):
                                separated=1
                            if aa>0.0 and structural==0 and separated==0:
                                state=2
                    if allowed!=0:
                        rx=px+s*ax-(geometry[row,0,0]/length)
                        ry=py+s*ay-(geometry[row,0,1]/length)
                        rz=pz+s*az-(geometry[row,0,2]/length)
                        w1=((ry*cz-rz*cy)*nx+(rz*cx-rx*cz)*ny+(rx*cy-ry*cx)*nz)/nn
                        w2=((by*rz-bz*ry)*nx+(bz*rx-bx*rz)*ny+(bx*ry-by*rx)*nz)/nn
                        w0=1.0-w1-w2
                        if w0<0.0 or w1<0.0 or w2<0.0:
                            allowed=0
                else:
                    edge=(candidate-3)//5
                    kind=(candidate-3)%5
                    j=(edge+1)%3
                    ex=(geometry[row,j,0]/length)-(geometry[row,edge,0]/length)
                    ey=(geometry[row,j,1]/length)-(geometry[row,edge,1]/length)
                    ez=(geometry[row,j,2]/length)-(geometry[row,edge,2]/length)
                    rx=px-(geometry[row,edge,0]/length)
                    ry=py-(geometry[row,edge,1]/length)
                    rz=pz-(geometry[row,edge,2]/length)
                    ee=ex*ex+ey*ey+ez*ez
                    t=0.0
                    if kind<2:
                        if kind==1:
                            s=1.0
                        t=((rx+s*ax)*ex+(ry+s*ay)*ey+(rz+s*az)*ez)/ee
                        t=tx.max(0.0,tx.min(1.0,t))
                        allowed=1
                    elif kind<4:
                        if kind==3:
                            t=1.0
                        if aa>0.0:
                            s=((t*ex-rx)*ax+(t*ey-ry)*ay+(t*ez-rz)*az)/aa
                            s=tx.max(0.0,tx.min(1.0,s))
                        allowed=1
                    elif aa>0.0:
                        dx=ay*ez-az*ey
                        dy=az*ex-ax*ez
                        dz=ax*ey-ay*ex
                        denominator=dx*dx+dy*dy+dz*dz
                        if denominator>condition*aa*ee:
                            s=((ey*rz-ez*ry)*dx+(ez*rx-ex*rz)*dy+(ex*ry-ey*rx)*dz)/denominator
                            t=((ay*rz-az*ry)*dx+(az*rx-ax*rz)*dy+(ax*ry-ay*rx)*dz)/denominator
                            if s>=0.0 and s<=1.0 and t>=0.0 and t<=1.0:
                                allowed=1
                        else:
                            structural=0
                            if (ay==0.0 or ez==0.0) and (az==0.0 or ey==0.0) and (az==0.0 or ex==0.0) and (ax==0.0 or ez==0.0) and (ax==0.0 or ey==0.0) and (ay==0.0 or ex==0.0):
                                structural=1
                            if structural==0:
                                state=2
                    if edge==0:
                        w0=1.0-t
                        w1=t
                    elif edge==1:
                        w1=1.0-t
                        w2=t
                    else:
                        w2=1.0-t
                        w0=t
                if allowed!=0:
                    sx=px+s*ax
                    sy=py+s*ay
                    sz=pz+s*az
                    # Affine barycentric reconstruction preserves the pivot.
                    txp=(geometry[row,0,0]/length)+w1*bx+w2*cx
                    typ=(geometry[row,0,1]/length)+w1*by+w2*cy
                    tzp=(geometry[row,0,2]/length)+w1*bz+w2*cz
                    dx=txp-sx
                    dy=typ-sy
                    dz=tzp-sz
                    square=(dx*dx+dy*dy+dz*dz)*length2
                    valid[row,candidate]=1
                    values[row,candidate,0]=s
                    values[row,candidate,1]=w0
                    values[row,candidate,2]=w1
                    values[row,candidate,3]=w2
                    values[row,candidate,4]=square
                    values[row,candidate,5]=sx*length
                    values[row,candidate,6]=sy*length
                    values[row,candidate,7]=sz*length
                    values[row,candidate,8]=txp*length
                    values[row,candidate,9]=typ*length
                    values[row,candidate,10]=tzp*length
        selected=-1
        best=0.0
        for candidate in range(18):
            if valid[row,candidate]!=0:
                square=values[row,candidate,4]
                if selected<0 or square<best:
                    selected=candidate
                    best=square
        mask=0
        normal_defined=0
        if selected>=0:
            for candidate in range(18):
                if valid[row,candidate]!=0 and tx.abs(values[row,candidate,4]-best)<=tie_error*length2:
                    mask=mask | (1<<candidate)
            distance=tx.sqrt(tx.max(0.0,best))
            summary[row,0]=distance
            summary[row,1]=distance-radii[row]
            for k in range(4):
                summary[row,k+2]=values[row,selected,k]
            for k in range(6):
                summary[row,k+6]=values[row,selected,k+5]
            if state==1 and distance>condition*length:
                normal_defined=1
                for k in range(3):
                    summary[row,k+12]=(summary[row,k+9]-summary[row,k+6])/distance
        elif state==1:
            state=2
        summary[row,15]=length
        meta[row,0]=state
        meta[row,1]=selected
        meta[row,2]=mask
        meta[row,3]=normal_defined
