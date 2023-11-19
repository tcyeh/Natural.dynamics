# -*- coding: utf-8 -*-
"""
Implementation of the "Natural dynamics" algorithm from Kanoria et al. (2010)
Author: Tzu-chi Yeh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import random as rd

pd.options.mode.chained_assignment = None 

# Define all constants here:
kappa = 0.8                     # "inertia" of dynamics
epsilon = 10**(-6)              # small number for while loop in dynamics
maxIteration = 100              # maximum number of iterations
gamma_change = float("inf")     # initialize value for while loop in dynamics

# Choose the type of uncertainty and the corresponding parameter(s)
# See thesis for more elaboration on the different notions

# Incomplete information (1)
dropEdgesIteration = False     
p_drop = 0.0

# Incomplete information (2)
noUpdatedOffers = False
p_noUpdate = 0.1

# Noise in communication (1)
noiseEveryOffer = False          
cvNoiseEveryOffer = 0.02

# Noise in communication (2)
noiseSomeOffers = False
p_noise = 0.0
cvNoiseSomeOffers = 0.05

# Environmental uncertainty (1)
weightChange = False
cvWeightChange = 0.02

# Environmental uncertainty (2)
leavingAgent = False
p_leave = 0.1

# Further investigation: Drop edge before the start of the dynamics
dropEdgesFixed = False
p_dropFixed = 0.0


# To make randomizations reproducible
rd.seed(1)

# Generation of a random graph
n = 10  # Number of nodes
p = 0.5  # Probability of edge creation
G = nx.gnp_random_graph(n, p)
for u, v in G.edges():
    G[u][v]['weight'] = np.round(rd.random(),2)
originalNumberEdges = len(G.edges)

# Manual generation of a graph
# =============================================================================
# G = nx.Graph()
# #edges = [("a","b",1), ("b","c",1)]
# edges = [("u","v",3),("u","w",2), ("v","w",2), ("w","x",1), ("x","y",2), ("x","z",2), ("y","z",3)]
# G.add_weighted_edges_from(edges)
# =============================================================================


# Define two helper functions
def relu(x):
    return(np.maximum(0,x))

def findLine(From, To, df):
    return df[(df["From"] == From) & (df["To"] == To)].index[0]


# =============================================================================
# # Further investigation: Delete edges with probability p_dropFixed before dynamics start
if dropEdgesFixed == True:
    edges_to_drop = []
    counter = 0
    for u, v in G.edges():
        if rd.random() < p_dropFixed:
            counter += 1
            edges_to_drop.append((u, v))
    G.remove_edges_from(edges_to_drop)
# =============================================================================

# Store the weights of the edges
weights_unique = np.array([edge[2]['weight'] for edge in G.edges(data=True)])

# Draw the graph
pos = nx.circular_layout(G)
nx.draw(G,pos, with_labels=True, node_color="orange")
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
plt.show()

# Extract edge information & duplicate them with reverse directions
edge_info = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
edge_info += [(v, u, weight) for u, v, weight in edge_info]

# Create a DataFrame from edge information
df = pd.DataFrame(edge_info, columns=["From", "To", "weight"])

# Initialize other columns in the DataFrame
df["alpha"] = 0.0
df["m"] = 0.0
df["gamma"] = 0.0

df = df.sort_values(by=["From", "To"])

# "Regular updates" (see Pseudocodes 2-4 in thesis)

def update_alpha():
    helper = []
    alpha_old = df["alpha"].copy()
    
    for i in df["From"].unique():   
        subframeTo = df.loc[df["To"] == i]
        for j in range(len(subframeTo)):
            subframeTo_modified = subframeTo.drop(subframeTo.index[j])
            if (len(subframeTo_modified) == 0):
                helper.append(0)
            else:
                helper.append(max(subframeTo_modified["m"]))
    
    df["alpha"] = [i * kappa for i in helper] + (1-kappa)*alpha_old
    
def update_m():
    for i in range(len(df)):
        first_term = df["weight"][i] - df["alpha"][i]
        second_term = df["weight"][i] - df["alpha"][i]-df["alpha"][findLine(df["To"][i],df["From"][i],df)]
        df["m"][i] = relu(first_term) - 0.5*relu(second_term)

def update_gamma():
    for i in df["To"].unique():
        subframeTo = df.loc[df["To"] == i]
        m_max = max(subframeTo["m"])
        for j in range(len(subframeTo)):
            df["gamma"][subframeTo.index[j]] = m_max

# Update alpha under the uncertainty uncertain receipt of offers (see Pseudocode 5 in thesis)
def updateAlphaUncertainReceipt():
    alpha_new = []
    alpha_old = df["alpha"].copy()
    
    edgesArray = list(G.edges())
    edges_to_drop = []
    for i in edgesArray:
        if rd.random() < p_drop: 
            edges_to_drop.append(i)
    
    for i in df["From"].unique():   
        subframeTo = df.loc[df["To"] == i]
        for j in range(len(subframeTo)):
            subframeTo_modified = subframeTo.drop(subframeTo.index[j])
            
            condition = ~df[['From', 'To']].apply(tuple, axis=1).isin(edges_to_drop)
            subframeTo_modified = subframeTo_modified.loc[condition]
            
            if (len(subframeTo_modified) == 0):
                alpha_new.append(0)
            else:
                alpha_new.append(max(subframeTo_modified["m"]))
    
    df["alpha"] = [i * kappa for i in alpha_new] + (1-kappa)*alpha_old
    
# "Update" offers under the uncertainty of non-updated offers (see Pseudocode 6 in thesis)
def runNoUpdatedOffers():
    m_old = df["m"].copy()
    for i in range(len(df)):
        if rd.random() < p_noUpdate:
            df["m"][i] = m_old[i]
            
# Add noise to every offer (see Pseudocode 7 in thesis)
def runNoiseEveryOffer():
    for i in range(len(df)):
        mean = df["m"][i]
        df["m"][i] = relu(np.random.normal(mean,cvNoiseEveryOffer*mean))

# Add noise to some offers (see Pseudocode 7 in thesis)
def runNoiseSomeOffers():
    for i in range(len(df)):
        if rd.random() < p_noise:
            mean = df["m"][i]
            df["m"][i] = relu(np.random.normal(mean,cvNoiseSomeOffers*mean))
    
# Add noise to the weights (see Pseudocode 8 in thesis)           
def changeWeights():
    for i in G.edges:
        mean = df["weight"][findLine(i[0],i[1],df)]
        newWeight = np.random.normal(mean, cvWeightChange*mean)
        df["weight"][findLine(i[0],i[1],df)] = newWeight
        df["weight"][findLine(i[1],i[0],df)] = newWeight

# Add the uncertainty of leaving agents (see Pseudocode 9 in thesis)
def runLeavingAgent():
    agents_to_drop = []
    for i in df["From"].unique():
        if rd.random() < p_leave: 
            agents_to_drop.append(i)
    df.loc[df['From'].isin(agents_to_drop), 'm'] = 0
    print(iteration, agents_to_drop)
        
# IMPLEMENTATION OF ACTUAL DYNAMICS STARTS HERE

"""
Remember: 
- "alpha" is the best alternative an agent has
- "m" is the offer an agent makes
- "gamma" is the payoff an agent (in the "To" column) receives
    = the maximum offer it receives from other agents
"""

iteration = 0

while gamma_change > np.linalg.norm(weights_unique,ord=1)*epsilon:

    
    df_old = df.copy()  # needed for comparison of current and previous gamma
    
    # main loop contains if-conditions because of the potential incorporations of the uncertainty notions
    
    if weightChange:
        changeWeights()
    
    if dropEdgesIteration:
        updateAlphaUncertainReceipt()
    else:
        update_alpha()
        
    update_m()
    
    if noUpdatedOffers:
        runNoUpdatedOffers()
    if noiseEveryOffer:
        runNoiseEveryOffer()
    if noiseSomeOffers:
        runNoiseSomeOffers()
    if leavingAgent:
        runLeavingAgent()
    
    update_gamma()
    
    if weightChange:
        df["weight"] = df_old["weight"]
    
    # Compute how large the change in gamma values was 
    gamma_old = df_old.groupby('To')['gamma'].first().reset_index()["gamma"]
    gamma_new = df.groupby('To')['gamma'].first().reset_index()["gamma"]
    gamma_change = np.linalg.norm(gamma_old - gamma_new, ord=2)
    
    iteration += 1
    
    if (iteration >= maxIteration): 
        print("Maximum number of iterations reached!")
        break
    
# Write a mechanism that includes a check whether the result is reasonable or not (compare weights with gammas)

# Print final outcome            
print(df.round(decimals=4))
print("{} iterations completed. Result:".format(iteration))

# Print the value the agents receive
for i in df["From"].unique():
    print("{} receives {}".format(i, round(df.loc[df["To"] == i]["gamma"].iloc[0],3)))
    
# Check whether the final outcome matches the maximum weight matching
matching = nx.max_weight_matching(G, maxcardinality=True)
accumulated_weight = sum(G[edge[0]][edge[1]]['weight'] for edge in matching)
sum = 0.0
for i in df["From"].unique():
    sum = sum + df.loc[df["To"] == i]["gamma"].iloc[0]
print(round(accumulated_weight,3) == round(sum,3))
