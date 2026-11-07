import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'analysis')]
os.chdir(HERE)
from pub_style import *
import numpy as np, matplotlib.patheffects as pe
from PIL import Image
HALO=[pe.withStroke(linewidth=2.4,foreground='white')]
F=np.array(Image.open(os.path.join(DATA, 'view_front.png'))); B=np.array(Image.open(os.path.join(DATA, 'view_back.png')))
X0,X1,Y0,Y1=90,1400,60,1430
F=F[Y0:Y1,X0:X1]; B=B[Y0:Y1,X0:X1]
W=7.1; fig=plt.figure(figsize=(W,3.75))
axF=fig.add_axes([0.0,0.02,0.46,0.92]); axB=fig.add_axes([0.54,0.02,0.46,0.92])
for ax,im in ((axF,F),(axB,B)):
    ax.imshow(im,extent=(X0,X1,Y1,Y0)); ax.set_xlim(X0-120,X1+40) if ax is axF else ax.set_xlim(X0-40,X1+120); ax.set_ylim(Y1,Y0-40); ax.axis('off')
def lab(ax,xy,txt,xyt,c=INK,fs=6.6,w='normal'):
    ax.annotate(txt,xy=xy,xytext=xyt,fontsize=fs,color=c,fontweight=w,ha='center',va='center',path_effects=HALO,
                arrowprops=dict(arrowstyle='-',lw=.5,color='#7a7a7a',shrinkA=1,shrinkB=9),zorder=10)
# residue labels, front view (ball positions from the fitted projection)
R=dict(K88=(386,409),E91=(321,564),D92=(426,613),N116=(650,472),T144=(691,524),Q150=(790,492),
       H94=(353,779),L133=(225,768),K128=(170,569),D132=(151,745))
pos=dict(K88=(330,300),E91=(250,470),D92=(470,700),N116=(500,175),T144=(610,690),Q150=(900,175),
         H94=(390,880),L133=(170,880),K128=(40,500),D132=(30,680))
col=dict(K88=OBS['hb'],E91=OBS['hb'],D92=OBS['hb'],N116=OBS['hb'],T144=OBS['hb'],Q150=OBS['hb'],
         H94=OBS['sasa'],L133=OBS['sasa'],K128=OBS['chi'],D132=OBS['chi'])
for k in R: lab(axF,R[k],k,pos[k],c=INK,w='semibold')
# region labels, back view
G=dict(sw1=(469,190),mg=(540,503),gtp=(637,244),ploop=(780,400),sw2=(650,800),a3=(1110,560),iface=(700,1060))
lab(axB,G['sw1'],'Switch I',(330,120),c='#5f8f4a',w='semibold')
lab(axB,G['gtp'],'GTP',(560,90),c=INK2)
lab(axB,G['mg'],'Mg²⁺',(330,520),c=INK2)
lab(axB,G['ploop'],'P-loop',(900,190),c='#b98a1e',w='semibold')
lab(axB,G['sw2'],'Switch II',(380,900),c='#7d64a8',w='semibold')
lab(axB,G['a3'],'α3',(1300,330),c='#5f86ad',w='semibold')
lab(axB,G['iface'],'Switch II–α3\ninterface',(560,1300),c='#3b5b92',w='semibold')
# 180° arrow
fig.text(0.5,0.52,'180°',ha='center',va='bottom',fontsize=7,color=INK2)
fig.patches.append(plt.matplotlib.patches.FancyArrowPatch((0.465,0.5),(0.535,0.5),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,lw=.8,color=INK2,connectionstyle='arc3,rad=-0.35'))
# key
from matplotlib.lines import Line2D
h=[Line2D([],[],marker='o',ls='',ms=5,mfc=OBS['hb'],mec='none',label='H-bond pair'),
   Line2D([],[],marker='o',ls='',ms=5,mfc=OBS['sasa'],mec='none',label='SASA pair'),
   Line2D([],[],marker='o',ls='',ms=5,mfc=OBS['chi'],mec='none',label='χ pair'),
   Line2D([],[],color='#3b5b92',lw=3,label='interface residues of core pairs')]
fig.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=4,fontsize=6.6,handlelength=1.2,columnspacing=1.2,frameon=False)
fig.text(0.005,0.97,'a',fontsize=10,fontweight='bold',color=INK,va='top')
save(fig,'fig2_top_structure.png'); print('ok')
