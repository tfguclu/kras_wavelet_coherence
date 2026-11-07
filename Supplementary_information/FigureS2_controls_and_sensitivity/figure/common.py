import pandas as pd, numpy as np, networkx as nx, itertools as it
import os
DATA=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')
SEQ='GMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHK'
lab=lambda i:'GTP' if i==201 else f'{SEQ[i]}{i}'
RT=0.001987*303
def load_dms():
    d=pd.read_csv(os.path.join(DATA, 'dms_couplings_canonical.csv'))
    d=d[(d['mean_kcal/mol'].abs()>RT)&(d['ci95_kcal/mol']<1)].copy()
    d['i']=d[['pos_i_can','pos_j_can']].min(axis=1); d['j']=d[['pos_i_can','pos_j_can']].max(axis=1)
    g=d.groupby(['i','j'])
    out=g.agg(eps=('mean_kcal/mol',lambda x:x.iloc[np.argmax(np.abs(x.values))]),
              trait=('trait',lambda x:'+'.join(sorted(set(x)))),n_terms=('trait','size')).reset_index()
    return out
def load_chi():
    c=pd.read_csv(os.path.join(DATA, 'chi_network_edges.csv'))
    return pd.DataFrame(dict(i=c.resid_i,j=c.resid_j,n=c.n_reps_any,cov=c.max_cov_ns))
def load_sasa():
    s=pd.read_csv(os.path.join(DATA, 'sasa_network_edges.csv'))
    return pd.DataFrame(dict(i=s.resid_i,j=s.resid_j,n=s.n_reps_any,cov=s.max_cov_ns))
def load_hb():
    h=pd.read_csv(os.path.join(DATA, 'hbond_hyperedges.csv'))
    h['res']=h.residues.apply(lambda s:sorted(map(int,s.split(';'))))
    return h
def hb_flat(h):
    d={}
    for r in h.itertuples():
        for p in it.combinations(r.res,2):
            n0,c0=d.get(p,(0,0)); d[p]=(max(n0,r.n_reps),max(c0,r.max_cov_ns))
    return pd.DataFrame([dict(i=a,j=b,n=n,cov=c) for (a,b),(n,c) in d.items()])
def graph(E):
    G=nx.Graph(); G.add_edges_from(zip(E.i,E.j)); return G
def overlap(G,dms):
    rows=[]
    for r in dms.itertuples():
        if r.i in G and r.j in G:
            if nx.has_path(G,r.i,r.j):
                L=nx.shortest_path_length(G,r.i,r.j)
                paths=list(nx.all_shortest_paths(G,r.i,r.j))
                inter=sorted({x for p in paths for x in p[1:-1]})
            else: L,paths,inter=None,[],[]
            rows.append(dict(i=r.i,j=r.j,pair=f'{lab(r.i)}–{lab(r.j)}',trait=r.trait,eps=r.eps,links=L,
                             n_paths=len(paths),intermediates=';'.join(lab(x) for x in inter),
                             path=' → '.join(lab(x) for x in paths[0]) if paths else ''))
    return pd.DataFrame(rows)
