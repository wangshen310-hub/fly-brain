"""Anatomy-only coverage search for future larger fixed-board displays.

This independent diagnostic is NOT used by an already-running protocol. It
does not inspect objects, food, actions or rewards, and never changes UVs.
"""
import hashlib
import numpy as np


def find_viewport(uv,size,*,pixel_stride=2,tile_stride=2):
    uv=np.asarray(uv)
    if (uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all()
            or (uv<0).any() or (uv>1).any() or size not in (4,6,8,12)
            or pixel_stride not in (1,2) or tile_stride not in (1,2)):
        raise ValueError('Valid anatomical UVs and board size required')
    pixels=np.unique((uv*np.array([319,179])).astype(int),axis=0)
    histogram=np.zeros((180,320),dtype=int);histogram[pixels[:,1],pixels[:,0]]=1
    integral=np.pad(histogram.cumsum(0).cumsum(1),((1,0),(1,0)))
    best=None
    for tile in range(6,min(32,180//(size+2))+1,tile_stride):
        x,y=np.meshgrid(np.arange(tile,320-(size+1)*tile+1,pixel_stride),
                        np.arange(tile,180-(size+1)*tile+1,pixel_stride))
        x,y=x.ravel(),y.ravel()
        if not len(x):continue
        parts=[]
        for row in range(size):
            for col in range(size):
                for qy in range(2):
                    for qx in range(2):
                        x0=x+col*tile+qx*(tile//2);y0=y+row*tile+qy*(tile//2)
                        # Odd tiles need disjoint parts covering the LAST row
                        # and column too; the two halves differ by one pixel.
                        x1=x+(col+1)*tile if qx else x+col*tile+tile//2
                        y1=y+(row+1)*tile if qy else y+row*tile+tile//2
                        parts.append(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])
        quarters=np.asarray(parts).reshape(size*size,4,len(x));cells=quarters.sum(1)
        # Prefer a complete four-part head display if one exists. Otherwise
        # cover every cell before maximizing the dense part of the image.
        qmin=quarters.min((0,1));cmin=cells.min(0)
        covered=(cells>0).sum(0);qcovered=(quarters>0).sum((0,1));total=cells.sum(0)
        index=np.lexsort((-y,-x,np.full(len(x),tile),total,qcovered,qmin,covered,cmin,qmin>0))[-1]
        score=(bool(qmin[index]>0),int(cmin[index]),int(covered[index]),int(qmin[index]),
               int(qcovered[index]),int(total[index]),tile,-int(x[index]),-int(y[index]))
        if best is None or score>best[0]:
            q=quarters[:,:,index].reshape(size,size,2,2)
            geometry={'projection':'fixed_world','version':'cell-priority-anatomical-viewport-candidate-v2',
                'tile_pixels':tile,'board_cells':size,'board_origin':[int(x[index]),int(y[index])],
                'image_size':[320,180],'wall_cells':1,'unique_sample_pixels':len(pixels),
                'quarter_sample_counts':q.tolist(),'cell_sample_counts':q.sum((2,3)).tolist(),
                'calibration':'Anatomical UV only; no objects, labels, actions or reward',
                'search_strides':{'pixel':pixel_stride,'tile':tile_stride},
                'uv_sha256':hashlib.sha256(np.ascontiguousarray(uv).tobytes()).hexdigest()}
            best=(score,geometry)
    if best is None:raise ValueError('No complete fixed board fits')
    return best[1]
