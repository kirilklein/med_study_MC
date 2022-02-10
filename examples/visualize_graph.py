import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


n_neg = 15
n_pos = 5
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
BG = nx.Graph()
source = ['s']
negatives = np.arange(n_neg)
positives = np.arange(n_neg, 
                (n_neg+n_pos))
sink = ['t']

BG.add_nodes_from(source, bipartite=0)
BG.add_nodes_from(negatives, bipartite=1)
BG.add_nodes_from(positives, bipartite=2)
BG.add_nodes_from(sink, bipartite=3)
source_negatives_edges = []
negatives_positives_edges = []
positives_sink_edges = []
for neg in negatives:
    source_negatives_edges.append(('s', neg))
for neg in negatives:
    for pos in positives:
        negatives_positives_edges.append((neg, pos))
for pos in positives:
    positives_sink_edges.append((pos, 't'))

BG.add_edges_from(source_negatives_edges)
BG.add_edges_from(negatives_positives_edges)
BG.add_edges_from(positives_sink_edges)


nodes = BG.nodes()
# for each of the parts create a set 

nodes_0  = set([n for n in nodes if  BG.nodes[n]['bipartite']==0])
nodes_1  = set([n for n in nodes if  BG.nodes[n]['bipartite']==1])
nodes_2  = set([n for n in nodes if  BG.nodes[n]['bipartite']==2])
nodes_3  = set([n for n in nodes if  BG.nodes[n]['bipartite']==3])

# set the location of the nodes for each set
pos = dict()
pos.update( (n, (1, i+int(n_neg/2))) for i, n in enumerate(nodes_0) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(nodes_1) ) # put nodes from Y at x=2
pos.update(
     (n, (3, i+int(n_neg/2)-int(n_pos/2))) for i, n in enumerate(nodes_2) ) # put nodes from X at x=1
pos.update( (n, (4, i+int(n_neg/2))) for i, n in enumerate(nodes_3) )

# set the colors
colors = ['k']
colors = colors + ['tab:blue' for i in range(n_neg)]
colors = colors + ['tab:red' for i in range(n_pos)]
colors.append('k')
nx.draw_networkx(BG, pos=pos, ax=ax,
    node_color=colors, with_labels=False,
    
    )
#nx.draw_networkx_edge_labels(
#    BG, pos,
#    edge_labels={('s', 0): 'capacity=1', 
#                 ('s', int(n_neg/2)): 'capacity=1'},
#    font_color='k'
#)

ax.text((pos['s'][0]+pos[1][0])/2, pos[n_neg-1][1],
    'capacity = 1\n weight = 0', 
    ha='center')
ax.text((pos[n_neg-1][0]+pos[n_neg][0])/2, pos[n_neg-1][1],
    'capacity = 1\n weight = distance', 
    ha='center')
ax.text((pos[n_neg+n_pos-1][0]+pos['t'][0])/2, pos[n_neg-1][1],
    'capacity = k\n weight = 0', 
    ha='center')    

ax.text(
    pos['s'][0], pos['s'][1]-2, 'source',
    fontsize=18, ha='center', va='bottom')
ax.text(
    pos[1][0], pos[1][1]-2, 'negatives',
    fontsize=18, ha='center', va='baseline')
ax.text(
    pos[n_neg][0], pos[n_neg][1]-2, 
    'positives',fontsize=18, ha='center', va='bottom')
ax.text(
    pos['t'][0], pos['t'][1]-2, 'sink',
    fontsize=18, ha='center', va='bottom')
fig.tight_layout()

fig.savefig(
    "D:\Thesis\CoBra\cobra\\figs\writeup\methods\matching\graph_max_flow_min_cost.png", 
)
#%%
print(pos) 