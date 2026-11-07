import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
INK='#1a1a1a'; INK2='#4d4d4d'; MUTE='#8c8c8c'; RULE='#d9d9d9'; BAND='#f2f0eb'; BG='white'
DMSB='#2b6cb0'; DMSF='#c0504d'; DMSBF='#7e57c2'
OBS={'chi':'#b83b72','sasa':'#2a9d8f','hb':'#d08c1f'}
OBSN={'chi':'χ dihedral','sasa':'SASA','hb':'H-bond'}
REP={1:'#d4d4d4',2:'#8f8f8f',3:'#2b2b2b'}
REG=[('P-loop',10,17),('Switch I',30,38),('Switch II',59,76)]
REGC={'P-loop':'#e0b34a','Switch I':'#7fb069','Switch II':'#a78bd0'}
REGL={'P-loop':'#f8eccb','Switch I':'#e3efdb','Switch II':'#ece5f6'}
SSEL=[('β1',3,9),('α1',16,25),('β2',38,46),('β3',49,57),('α2',65,74),('β4',77,83),('α3',87,104),('β5',110,116),('α4',127,137),('β6',140,145),('α5',152,164)]
plt.rcParams.update({'font.family':'Inter','font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'xtick.labelsize':6.8,'ytick.labelsize':6.8,
 'axes.edgecolor':INK2,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':3.5,'ytick.major.size':3.5,
 'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'in','ytick.direction':'in','xtick.minor.size':2,'ytick.minor.size':2,
 'legend.frameon':False,'legend.fontsize':6.8,'savefig.dpi':400,'figure.dpi':150,'mathtext.fontset':'custom','mathtext.rm':'Inter','mathtext.it':'Inter:italic'})
def panel(ax,letter,x=-0.02,y=1.02,fig=None):
    ax.text(x,y,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',color=INK,va='bottom',ha='left')
MINFS=6.2
def _readable(fig):
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < MINFS: t.set_fontsize(MINFS)
    for ax in fig.axes:
        for axis, nm in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
            tk = axis.get_major_ticks()
            if tk and tk[0].tick1line.get_markersize() == 0 and tk[0].tick2line.get_markersize() == 0:
                continue                      # ticks deliberately hidden (heatmaps, rings)
            ax.tick_params(axis=nm, which='major', direction='in', length=3.5, width=.7)
            ax.tick_params(axis=nm, which='minor', direction='in', length=2, width=.5)
def save(fig,name):
    _readable(fig)
    fig.savefig(name,dpi=400,facecolor=BG,bbox_inches='tight',pad_inches=.03)

def excl(ax, a, b, zorder=1, ymax=1):
    """excluded ends: one style in every figure (white, gray hatching)"""
    ax.axvspan(a, b, fc='white', ec='#b8b8b8', hatch='////', lw=0, zorder=zorder, ymax=ymax)
