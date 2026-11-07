import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
from common import *
from matplotlib.patches import FancyArrowPatch, Rectangle
rng=np.random.default_rng(4)
fig=plt.figure(figsize=(7.1,2.05))
W=.20; X=[.01,.265,.52,.775]
axs=[fig.add_axes([x+.012,.1,W,.62]) for x in X]
titles=['3 × 10 µs HMR MD','Three observables','Wavelet coherence','Coherent pairs']
subs=['KRAS4B·GTP·Mg²⁺, 1 ns frames','χ (cos, sin) · SASA · H-bond 0/1','80–120 ns, ≥ 0.75, ≥ 500 ns','pairs · H-bond hyperedges']
for k,(ax,t,s) in enumerate(zip(axs,titles,subs)):
    fig.text(X[k],.92,'abcd'[k],fontsize=9,fontweight='bold',va='baseline')
    fig.text(X[k]+.02,.92,t,fontsize=7.4,fontweight='semibold',color=INK,va='baseline')
    fig.text(X[k]+.02,.855,s,fontsize=5.8,color=INK2,va='baseline')
for k in range(3):
    fig.add_artist(FancyArrowPatch((X[k+1]-.035,.935),(X[k+1]-.012,.935),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=7,color=MUTE,lw=.8))
# a replicas
ax=axs[0]
for r,(ex) in enumerate([6.617,7.129,0.781]):
    y=2-r; ax.add_patch(Rectangle((0,y-.18),10,.36,color='#e6e6e6',lw=0)); ax.add_patch(Rectangle((.722,y-.18),10-2*.722,.36,color='#bdbdbd',lw=0))
    ax.text(-.3,y,f'r{r+1}',ha='right',va='center',fontsize=5.6,color=INK2)
# HMR schematic: mass moved from the heavy atom to its hydrogen allows a 4 fs step
yy=4.35
for x0,sh,sH,lab,mH in [(0.3,60,14,'standard\n2 fs','H 1.0'),(6.0,45,36,'HMR\n4 fs','H 3.0 amu')]:
    ax.plot([x0,x0+2.0],[yy,yy],color=INK2,lw=1.0,zorder=1)
    ax.scatter([x0],[yy],s=sh,color='#9a9a9a',edgecolor=INK2,lw=.5,zorder=2)
    ax.scatter([x0+2.0],[yy],s=sH,color='white',edgecolor=INK2,lw=.6,zorder=2)
    ax.text(x0+2.0,yy+.45,mH,ha='center',va='bottom',fontsize=4.9,color=INK2)
    ax.text(x0+1.0,yy-.55,lab,ha='center',va='top',fontsize=5.2,color=INK)
ax.add_artist(FancyArrowPatch((3.0,yy),(5.3,yy),arrowstyle='-|>',mutation_scale=6,color=MUTE,lw=.7))
ax.set_xlim(-1.2,10.3); ax.set_ylim(-.8,5.4); ax.set_xticks([0,5,10]); ax.set_xticklabels(['0','5','10 µs'],fontsize=5.6)
ax.spines['left'].set_visible(False); ax.set_yticks([]); ax.tick_params(axis='x',length=2)
# b signals
ax=axs[1]; t=np.linspace(0,3000,600)
chi=np.cos(np.cumsum(rng.normal(0,.12,600))+np.sin(t/180)); s=np.convolve(rng.normal(0,1,600),np.ones(20)/20,'same')+np.sin(t/160)*.7
hb=(np.convolve(np.sin(t/150+.3)+rng.normal(0,.5,600),np.ones(15)/15,'same')>0).astype(float)
for k,(v,c,l) in enumerate([(chi,OBS['chi'],'χ'),(s,OBS['sasa'],'SASA'),(hb,OBS['hb'],'H-bond')]):
    v=(v-v.min())/(v.max()-v.min())*.8; y=2-k
    ax.plot(t,v+y-.4,color=c,lw=.6,drawstyle='steps-mid' if l=='H-bond' else 'default'); ax.text(3050,y,l,ha='left',va='center',fontsize=5.6,color=c)
ax.set_xlim(0,3700); ax.set_ylim(-.6,2.7); ax.axis('off')
# c coherence map
ax=axs[2]; tt=np.linspace(0,10,200); per=np.logspace(1,3,80)
Z=np.zeros((80,200))
for _ in range(14):
    c0=rng.uniform(0,10); p0=rng.uniform(1,3); Z+=rng.uniform(.3,1)*np.exp(-((tt[None,:]-c0)/rng.uniform(.4,1.5))**2-((np.log10(per)[:,None]-p0)/.25)**2)
Z+=1.1*np.exp(-((tt[None,:]-4.2)/1.0)**4-((np.log10(per)[:,None]-2.0)/.12)**2)
Z=np.clip(Z/Z.max(),0,1)
ax.pcolormesh(tt,per,Z,cmap='Greys',shading='auto',vmin=0,vmax=1.05,rasterized=True)
ax.set_yscale('log'); ax.axhspan(80,120,color=DMSB,alpha=.0)
ax.axhline(80,color=INK,lw=.6,ls=':'); ax.axhline(120,color=INK,lw=.6,ls=':')
ax.add_patch(Rectangle((3.4,80),1.6,40,fill=False,ec=INK,lw=1.0))
ax.set_xlim(0,10); ax.set_ylim(10,1000); ax.set_xticks([0,5,10]); ax.set_xticklabels(['0','5','10 µs'],fontsize=5.6)
ax.set_yticks([10,100,1000]); ax.set_yticklabels(['10','100','1000'],fontsize=5.6); ax.text(0,1.02,'period (ns)',transform=ax.transAxes,fontsize=5.4,color=INK2,va='bottom')
ax.tick_params(length=2)
# d network
ax=axs[3]; ax.set_aspect('equal'); ax.axis('off')
N=186; ang=lambda i:np.pi/2-2*np.pi*(177 if i==201 else i)/N
th=np.linspace(0,2*np.pi,200); ax.plot(np.cos(th),np.sin(th),color=RULE,lw=.6)
chi_,sasa_,h_=load_chi(),load_sasa(),load_hb(); hbf_=hb_flat(h_)
for nm,E in [('chi',chi_),('sasa',sasa_),('hb',hbf_)]:
    for r in E[E.n>=2].itertuples():
        a,b=ang(r.i),ang(r.j); x0,y0,x1,y1=np.cos(a),np.sin(a),np.cos(b),np.sin(b); tt_=np.linspace(0,1,30); k=.3
        X_=(1-tt_)**2*x0+2*(1-tt_)*tt_*(x0+x1)/2*k+tt_**2*x1; Y_=(1-tt_)**2*y0+2*(1-tt_)*tt_*(y0+y1)/2*k+tt_**2*y1
        ax.plot(X_,Y_,color=OBS[nm],lw=.6 if r.n==2 else 1.1,alpha=.85)
ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15)
ax.text(0,-1.25,'coherent pairs in ≥ 2 replicas',ha='center',va='top',fontsize=5.2,color=MUTE)
save(fig,'fig1_top_workflow.png')
