import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
import numpy as np, json
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
pdb=os.path.join(DATA, 'kras_start_model.pdb')
CA={}
for l in open(pdb):
    if l.startswith(('ATOM','HETATM')) and l[12:16].strip()=='CA' and l[17:20]!=' MG':
        CA[int(l[22:26])]=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
groups={'orange':[88,91,92,116,144,150],'teal':[94,133],'mag':[128,132]}
blobs={'front':{'mag':[(176,571),(146,742)],'teal':[(350,779),(232,782)],'orange':[(400,410),(678,504),(790,494),(308,560),(424,608)]},
       'back':{'mag':[(1361,740)],'teal':[(1268,761),(1162,776)],'orange':[(1109,407),(847,467),(708,481),(813,523),(1186,559),(1069,608)]}}
X=np.array([CA[r] for g in groups for r in groups[g]]); cen=X.mean(0)
def proj(p,pts):
    R=Rot.from_rotvec(p[:3]).as_matrix(); q=(pts-cen)@R.T
    return np.c_[p[3]*q[:,0]+p[4], -p[3]*q[:,1]+p[5]]
def cost(p,view):
    c=0
    for g,rs in groups.items():
        P=proj(p,np.array([CA[r] for r in rs])); B=np.array(blobs[view][g],float)
        d=np.linalg.norm(P[:,None]-B[None],axis=2)
        c+=np.sum(np.minimum(d.min(1),200)**2)+np.sum(np.minimum(d.min(0),200)**2)
    return c
out={}
rng=np.random.default_rng(0)
for view in blobs:
    best=None
    for k in range(3000):
        p0=np.r_[Rot.random(random_state=rng.integers(1e9)).as_rotvec(), rng.uniform(15,30), 750+rng.normal(0,50), 750+rng.normal(0,50)]
        r=minimize(cost,p0,args=(view,),method='Powell',options={'maxiter':3000,'xtol':1e-3,'ftol':1e-3})
        if best is None or r.fun<best.fun: best=r
    p=best.x; print(view,best.fun,p)
    out[view]={'p':list(p)}
    for g,rs in groups.items():
        for r,(x,y) in zip(rs,proj(p,np.array([CA[r] for r in rs]))): print(' ',r,g,int(x),int(y))
json.dump(out,open('fit.json','w'))
