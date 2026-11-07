import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
from common import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import fig2_networks as F   # reuse layout helpers (runs fig2 too; harmless)
xy,curve,spread,ang=F.xy,F.curve,F.spread,F.ang
dms=load_dms(); dms=dms[(dms.i<=167)&(dms.j<=167)].copy()
chi,sasa,hbf=load_chi(),load_sasa(),hb_flat(load_hb())
ov=set()
for E in (chi,sasa,hbf):
    m=E.merge(dms,on=['i','j']); ov|=set(zip(m.i,m.j))
print('DMS pairs 1-167:',len(dms),'overlaps:',len(ov))
BOTH=DMSBF
TC={'Binding':DMSB,'Folding':DMSF,'Binding+Folding':BOTH}
def draw(ax,D,head,toplab,col=None,wcol='eps',ov=ov,labset=None):
    ax.set_aspect('equal'); ax.axis('off')
    t=np.linspace(0,2*np.pi,400); ax.plot(np.cos(t),np.sin(t),color=RULE,lw=.7,zorder=0)
    for n,a,b in REG:
        tt=np.linspace(ang(a-.5),ang(b+.5),20); ax.plot(np.cos(tt),np.sin(tt),color=REGC[n],lw=3.2,solid_capstyle='butt',zorder=1)
    for k in (0,50,100,150):
        x,y=xy(k,.84); ax.text(x,y,str(k),ha='center',va='center',fontsize=8.2,color=MUTE)
    wmax=D[wcol].abs().max()
    for r in D.sort_values(wcol,key=abs).itertuples():
        X,Y=curve(r.i,r.j); o=(r.i,r.j) in ov
        ax.plot(X,Y,color=INK if o else (col or TC[r.trait]),lw=(1.6 if o else .4+1.4*abs(getattr(r,wcol))/wmax),alpha=1 if o else .75,solid_capstyle='round',zorder=5 if o else 2)
    deg=pd.Series(list(D.i)+list(D.j)).value_counts()
    ovn={i for p in ov for i in p if p in set(zip(D.i,D.j))}
    for i in deg.index:
        x,y=xy(i); ax.scatter(x,y,s=2.5+1.0*deg[i],color=INK if i in ovn else '#707070',edgecolor='white',lw=.3,zorder=10)
    labs=sorted(labset if labset is not None else set(deg[deg>=toplab].index)|ovn,key=lambda i:-ang(i))
    th=[-ang(i) for i in labs]; th2=F.spread(th,np.radians(8.0)); order=np.argsort(th)
    for k,idx in enumerate(order):
        i=labs[idx]; a_true=-th[idx]; a_lab=-th2[k]
        ax.plot([np.cos(a_true)*1.02,np.cos(a_true)*1.08,np.cos(a_lab)*1.17],[np.sin(a_true)*1.02,np.sin(a_true)*1.08,np.sin(a_lab)*1.17],color='#b0b0b0',lw=.35,zorder=10)
        d=np.degrees(a_lab)%360; left=90<d<270
        ax.text(np.cos(a_lab)*1.19,np.sin(a_lab)*1.19,lab(i),ha='right' if left else 'left',va='center',rotation=d-180 if left else d,rotation_mode='anchor',
                fontsize=8.8,color=INK,fontweight='semibold' if (i in ovn and labset is None) or (labset is not None and i in labset) else 'normal',zorder=11)
    ax.text(0,1.6,head,ha='center',va='bottom',fontsize=10,color=INK)
    ax.set_xlim(-1.62,1.62); ax.set_ylim(-1.62,1.85)
B=dms[dms.trait.str.contains('Binding')]; Fo=dms[dms.trait.str.contains('Folding')]
print('binding',len(B),'folding',len(Fo),'both',(dms.trait=='Binding+Folding').sum())
# EVcouplings: file indices are ours + 1
EVC='#2a9d8f'
ev=pd.read_csv(os.path.join(DATA, 'EVcouplings_ECs_b0.5.csv')); ev['i']=ev.i-1; ev['j']=ev.j-1
ev=ev[(ev.i>=1)&(ev.j<=167)].copy(); ev['s']=ev.score/ev.score.max()
SA={(100,112),(85,97),(158,167),(100,161)}
# reproducible MD pairs (42): 10 chi, 25 SASA, 7 H-bond hyperedges expanded to residue pairs
C=set(zip(chi[chi.n>=2].i,chi[chi.n>=2].j))-SA; S=set(zip(sasa[sasa.n>=2].i,sasa[sasa.n>=2].j))
hb=load_hb(); H=set()
for r in hb[hb.n_reps>=2].itertuples(): H|=set(it.combinations(r.res,2))
MD=C|S|H
Dset=set(zip(dms.i,dms.j)); Eset=set(zip(ev.i,ev.j))
ovD=MD&Dset; ovE=MD&Eset
print('MD',len(C),len(S),len(H),len(MD),'DMS',len(Dset),'EV',len(Eset),'MD&DMS',sorted(ovD),'MD&EV',sorted(ovE),'DMS&EV',sorted(Dset&Eset))
ev[['i','j','score','s','probability','dist']].to_csv('ev_pairs_ourindex.csv',index=False)
fig=plt.figure(figsize=(7.8,7.4))
top=fig.add_gridspec(1,3,wspace=.08,top=.995,bottom=.53,left=.01,right=.99); axs=[fig.add_subplot(top[k]) for k in range(3)]
draw(axs[0],B,f'RAF1 binding  ·  {len(B)} pairs',4,ov=set())
draw(axs[1],Fo,f'Folding  ·  {len(Fo)} pairs',2,ov=set())
draw(axs[2],ev,f'EVcouplings  ·  {len(ev)} pairs',6,col=EVC,wcol='s',ov=set())
for ax,l in zip(axs,'abc'): ax.text(-1.6,1.6,l,fontsize=13,fontweight='bold',va='bottom')
hd=[Line2D([],[],color=DMSB,lw=1.4,label='binding'),Line2D([],[],color=DMSF,lw=1.4,label='folding'),Line2D([],[],color=BOTH,lw=1.4,label='both'),Line2D([],[],color=EVC,lw=1.4,label='evolutionary coupling')]+[Patch(color=REGC[n],label=n) for n,*_ in REG]
fig.legend(handles=hd,loc='upper center',ncol=5,bbox_to_anchor=(.5,.565),handlelength=1.5,columnspacing=1.2,handleheight=.5,fontsize=9)
# d: sets
from matplotlib.patches import Circle
bot=fig.add_gridspec(1,2,width_ratios=[.9,1.1],wspace=0,top=.475,bottom=.0,left=.02,right=1)
ax=fig.add_subplot(bot[0]); ax.set_aspect('equal'); ax.axis('off')
MDC='#444444'
cen={'MD':(0,.62),'DMS':(-.55,-.3),'EV':(.55,-.3)}; R=.78
for k,c in [('MD',MDC),('DMS',DMSB),('EV',EVC)]:
    ax.add_patch(Circle(cen[k],R,fc=c,alpha=.10,ec=c,lw=1.1))
names={'MD':f'Reproducible MD\n{len(MD)} residue pairs','DMS':f'DMS\n{len(Dset)} pairs','EV':f'EVcouplings\n{len(Eset)} pairs'}
ax.text(0,1.48,names['MD'],ha='center',va='bottom',fontsize=9.5,color=MDC)
ax.text(-1.3,-1.12,names['DMS'],ha='center',va='top',fontsize=9.5,color=DMSB)
ax.text(1.3,-1.12,names['EV'],ha='center',va='top',fontsize=9.5,color=EVC)
reg={'MD':MD-Dset-Eset,'DMS':Dset-MD-Eset,'EV':Eset-MD-Dset,'MD&DMS':(MD&Dset)-Eset,'MD&EV':(MD&Eset)-Dset,'DMS&EV':(Dset&Eset)-MD,'all':MD&Dset&Eset}
pos={'MD':(0,.95),'DMS':(-.85,-.5),'EV':(.85,-.5),'MD&DMS':(-.5,.25),'MD&EV':(.5,.25),'DMS&EV':(0,-.6),'all':(0,.0)}
for k,(x,y) in pos.items(): ax.text(x,y,str(len(reg[k])),ha='center',va='center',fontsize=12,color=INK,fontweight='bold' if '&' in k or k=='all' else 'normal')
ax.set_xlim(-1.75,1.75); ax.set_ylim(-1.5,1.75)
ax.text(-1.75,1.6,'d',fontsize=13,fontweight='bold',va='bottom')
# compact table of shared residue pairs
tg=lambda p:', '.join(x for x,st in [(r'$\chi$',C),('SASA',S),('H-bond',H)] if p in st)
tr=dms.set_index(['i','j']).trait
rows=[(p,tg(p),tr[p].lower().replace('+',' + '),'') for p in sorted(reg['MD&DMS'])]
rows+=[(p,tg(p),'','✓') for p in sorted(reg['MD&EV'])]
rows+=[(p,'',tr[p].lower().replace('+',' + '),'✓') for p in sorted(reg['DMS&EV'])]
lx=fig.add_subplot(bot[1]); lx.axis('off'); lx.set_xlim(0,1); lx.set_ylim(0,1)
X=[.02,.30,.62,.88]; top_=.93; rh=.052
for x,hd in zip(X,['Residue pair','MD observable','DMS','EV']):
    lx.text(x,top_,hd,fontsize=9.5,fontweight='bold',color=INK,va='center',ha='center' if hd=='EV' else 'left')
lx.plot([0,1],[top_-rh/2-.005]*2,color=INK,lw=.8)
y=top_-rh-.01; prev=None
groups=['MD + DMS']*len(reg['MD&DMS'])+['MD + EV']*len(reg['MD&EV'])+['DMS + EV']*len(reg['DMS&EV'])
for k,((p,ob,tri,ev),g) in enumerate(zip(rows,groups)):
    if prev is not None and g!=prev: lx.plot([0,1],[y+rh/2]*2,color=MUTE,lw=.5)
    if k%2==0: lx.add_patch(plt.Rectangle((0,y-rh/2),1,rh,color='#f3f3f3',lw=0,zorder=0))
    lx.text(X[0],y,f'{lab(p[0])}–{lab(p[1])}',fontsize=9,color=INK,va='center')
    lx.text(X[1],y,ob or '–',fontsize=9,color=INK if ob else MUTE,va='center')
    lx.text(X[2],y,tri or '–',fontsize=9,color=DMSB if tri else MUTE,va='center')
    lx.text(X[3],y,ev or '–',fontsize=9.5,color=EVC if ev else MUTE,va='center',ha='center')
    prev=g; y-=rh
lx.plot([0,1],[y+rh/2]*2,color=INK,lw=.8)
lx.text(0,y-.01,'No pair is shared by all three.',fontsize=8.5,color=MUTE,va='top')
save(fig,'Figure5.png')
