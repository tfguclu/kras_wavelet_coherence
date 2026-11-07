from pub_style import *
from common import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
N=186; pos_of=lambda i:177 if i==201 else i
ang=lambda i:np.pi/2-2*np.pi*pos_of(i)/N
xy=lambda i,r=1.0:(r*np.cos(ang(i)),r*np.sin(ang(i)))
def curve(a,b):
    (x0,y0),(x1,y1)=xy(a),xy(b); d=abs(pos_of(a)-pos_of(b)); d=min(d,N-d)/(N/2); k=.75-.6*d; cx,cy=(x0+x1)/2*k,(y0+y1)/2*k; t=np.linspace(0,1,50)
    return (1-t)**2*x0+2*(1-t)*t*cx+t**2*x1,(1-t)**2*y0+2*(1-t)*t*cy+t**2*y1
def spread(angles,minsep):
    a=np.array(sorted(angles)); 
    for _ in range(200):
        moved=False
        for k in range(1,len(a)):
            if a[k]-a[k-1]<minsep:
                m=(a[k]+a[k-1])/2; a[k-1]=m-minsep/2-1e-6; a[k]=m+minsep/2+1e-6; moved=True
        if not moved: break
    return a
def draw(ax,E,overlap,head,extra=(),conn=(),span=(),hc=INK):
    ax.set_aspect('equal'); ax.axis('off')
    t=np.linspace(0,2*np.pi,400); ax.plot(np.cos(t),np.sin(t),color=RULE,lw=.7,zorder=0)
    for n,a,b in REG:
        tt=np.linspace(ang(a-.5),ang(b+.5),20); ax.plot(np.cos(tt),np.sin(tt),color=REGC[n],lw=3.2,solid_capstyle='butt',zorder=1)
    for k in (0,50,100,150):
        x,y=xy(k,.84); ax.text(x,y,str(k),ha='center',va='center',fontsize=5.3,color=MUTE)
    wmax=max(c for *_,c in E)
    for a,b,n,c in sorted(E,key=lambda e:(e[2],e[3])):
        X,Y=curve(a,b); ax.plot(X,Y,color=REP[n],lw=.3 if n==1 else .5+1.5*c/wmax,alpha=.9,solid_capstyle='round',zorder=2+n)
    for (a,b),(c,d),n in conn:
        X1,Y1=curve(a,b); X2,Y2=curve(c,d)
        ax.plot([X1[25],X2[25]],[Y1[25],Y2[25]],color='#e2e2e2' if n==1 else REP[n],lw=.35 if n==1 else .6,ls=(0,(1,1.2)),zorder=2.5 if n==1 else 2+n)
    ov={i for p in list(overlap)+list(span) for i in p}
    for a,b in overlap:
        X,Y=curve(a,b); ax.plot(X,Y,color=DMSB,lw=1.5,solid_capstyle='round',zorder=9)
    for a,b in span:
        X,Y=curve(a,b); ax.plot(X,Y,color=DMSB,lw=1.2,ls=(0,(2.5,1.5)),zorder=9)
    nodes={i for a,b,*_ in E for i in (a,b)}
    deg=pd.Series([i for a,b,*_ in E for i in (a,b)]).value_counts()
    for i in nodes:
        x,y=xy(i); ax.scatter(x,y,s=2.5+1.0*deg[i],color=DMSB if i in ov else '#707070',edgecolor='white',lw=.3,zorder=10)
    labs=sorted((ov|{i for a,b,n,_ in E if n==3 for i in (a,b)}|set(extra))&nodes,key=lambda i:-ang(i))
    th=[-ang(i) for i in labs]; th2=spread(th,np.radians(5.2))
    order=np.argsort(th)
    for k,idx in enumerate(order):
        i=labs[idx]; a_true=-th[idx]; a_lab=-th2[k]
        x0,y0=np.cos(a_true)*1.02,np.sin(a_true)*1.02; x1,y1=np.cos(a_lab)*1.17,np.sin(a_lab)*1.17
        ax.plot([x0,np.cos(a_true)*1.08,x1],[y0,np.sin(a_true)*1.08,y1],color='#b0b0b0',lw=.35,zorder=10)
        d=np.degrees(a_lab)%360; left=90<d<270; lx,ly=np.cos(a_lab)*1.19,np.sin(a_lab)*1.19
        ax.text(lx,ly,lab(i),ha='right' if left else 'left',va='center',rotation=d-180 if left else d,rotation_mode='anchor',
                fontsize=5.5,color=DMSB if i in ov else INK,fontweight='semibold' if i in ov else 'normal',zorder=11)
    ax.text(0,1.55,head,ha='center',va='bottom',fontsize=8.6,color=hc,fontweight='semibold')
    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.7)
chi,sasa,h=load_chi(),load_sasa(),load_hb(); hbf=hb_flat(h); dms=load_dms()
FH=5.45
fig=plt.figure(figsize=(7.1,FH))
NH=2.37*3.2/3/FH
axs=[fig.add_axes([k/3,1-NH,1/3,NH]) for k in range(3)]
def bonds(h):
    d={}; conn=[]
    for r in h.itertuples():
        bb=[tuple(sorted(map(int,x.split('-')))) for x in (r.hb1,r.hb2)]
        for b in bb:
            n0,c0=d.get(b,(0,0)); d[b]=(max(n0,r.n_reps),max(c0,r.max_cov_ns))
        if r.size==4: conn.append((*bb,r.n_reps))
    return pd.DataFrame([dict(i=a,j=b,n=n,cov=c) for (a,b),(n,c) in d.items()]),conn
hbB,conn=bonds(h)
hres=[set(r) for r in h.res]
for ax,(nm,E,head) in zip(axs,[('chi',chi,'χ dihedral  ·  179 pairs'),('sasa',sasa,'SASA  ·  104 pairs'),('hb',hbB,'H-bond  ·  67 hyperedges')]):
    m=E.merge(dms,on=['i','j'])
    if nm=='hb':
        bset=set(zip(hbB.i,hbB.j))
        span=[(r.i,r.j) for r in dms.itertuples() if (r.i,r.j) not in bset and any({r.i,r.j}<=s for s in hres)]
        print('H-bond DMS: bonds',list(zip(m.i,m.j)),'span',span)
        draw(ax,[tuple(r) for r in E[['i','j','n','cov']].itertuples(index=False)],list(zip(m.i,m.j)),head,extra=(201,),conn=conn,span=span,hc=OBS[nm])
    else:
        draw(ax,[tuple(r) for r in E[['i','j','n','cov']].itertuples(index=False)],list(zip(m.i,m.j)),head,hc=OBS[nm])
for ax,l in zip(axs,'abc'): ax.text(-1.5,1.55,l,fontsize=10,fontweight='bold',va='bottom')
hd=[Line2D([],[],color=REP[k],lw=1.4,label=f'{k} replica'+('s' if k>1 else '')) for k in (1,2,3)]+[Line2D([],[],color=DMSB,lw=1.6,label='also a DMS coupling'),Line2D([],[],color=DMSB,lw=1.3,ls=(0,(2.5,1.5)),label='DMS pair spanning two bonds'),Line2D([],[],color='#9a9a9a',lw=.8,ls=(0,(1,1.2)),label='two bonds of one hyperedge')]+\
   [Patch(color=REGC[n],label=n) for n,*_ in REG]
fig.legend(handles=hd,loc='upper center',ncol=5,bbox_to_anchor=(.5,1-NH+.005),handlelength=1.5,columnspacing=1.3,handleheight=.5)
import fig2_funnel_panel as FP
FP.draw(fig.add_axes([0.0,0.0,1.0,2.30/FH]))
save(fig,'_networks_helper.png')
