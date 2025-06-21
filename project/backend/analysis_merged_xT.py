import json
import requests
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from mplsoccer import Pitch  # Added for soccer pitch visualization

# 1. Data Loading & Helper Functions
def get_url_mapping():
    return {
        "Barcelona vs Atletico Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773372.json",
        "Barcelona vs Real Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773585.json",
        "Barcelona vs Sevilla": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773672.json",
        "Spain vs Russia": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/7582.json"
    }

def load_event_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def extract_starting_players(events, team_name="Barcelona"):
    players = set()
    positions = {}
    for e in events:
        if e.get("type", {}).get("name") == "Starting XI" and e.get("team", {}).get("name") == team_name:
            for p in e["tactics"]["lineup"]:
                name = p["player"]["name"]
                players.add(name)
                positions[name] = p["position"]["name"]
    return players, positions

def get_grid_cell(x, y, pitch_length=120, pitch_width=80, grid_length=16, grid_width=12):
    x_norm = max(0, min(x, pitch_length))
    y_norm = max(0, min(y, pitch_width))
    i = int((x_norm / pitch_length) * grid_length)
    j = int((y_norm / pitch_width) * grid_width)
    return min(i, grid_length-1), min(j, grid_width-1)

# 2. xT-based Graph Construction
def compute_pass_xT_gain_avg(events, starting_players, team_name="Barcelona"):
    xT_matrix = np.load("xT_map.npy")
    pass_counts = defaultdict(lambda: defaultdict(int))
    xT_total = defaultdict(lambda: defaultdict(float))

    for e in events:
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        if "recipient" not in e.get("pass", {}): continue

        passer = e["player"]["name"]
        receiver = e["pass"]["recipient"]["name"]

        if passer in starting_players and receiver in starting_players:
            pass_counts[passer][receiver] += 1
            grid_start = get_grid_cell(*e["location"])
            grid_end = get_grid_cell(*e["pass"]["end_location"])
            xT_total[passer][receiver] += xT_matrix[grid_end] - xT_matrix[grid_start]

    xT_avg = defaultdict(lambda: defaultdict(float))
    for passer in pass_counts:
        for receiver in pass_counts[passer]:
            count = pass_counts[passer][receiver]
            if count > 0:
                xT_avg[passer][receiver] = xT_total[passer][receiver] / count
    return xT_avg

def build_passing_graph_with_xt(events, starting_players, team_name="Barcelona"):
    xT_avg = compute_pass_xT_gain_avg(events, starting_players, team_name)
    location_data = defaultdict(list)

    for e in events:
        if e.get("type", {}).get("name") == "Pass" and e.get("team", {}).get("name") == team_name:
            if "location" in e and "recipient" in e.get("pass", {}):
                passer = e["player"]["name"]
                receiver = e["pass"]["recipient"]["name"]
                if passer in starting_players and receiver in starting_players:
                    location_data[passer].append(e["location"])
                    location_data[receiver].append(e["pass"]["end_location"])

    positions = {
        p: (sum(x for x, y in locs)/len(locs), sum(y for x, y in locs)/len(locs))
        for p, locs in location_data.items() if locs
    }

    basic_g = nx.DiGraph()
    attack_g = nx.DiGraph()
    defense_g = nx.DiGraph()

    for passer in xT_avg:
        for receiver in xT_avg[passer]:
            avg_xt = xT_avg[passer][receiver]
            count = len([e for e in events if e.get("player", {}).get("name") == passer and e.get("pass", {}).get("recipient", {}).get("name") == receiver])

            basic_g.add_edge(passer, receiver, weight=count)
            if avg_xt > 0:
                attack_g.add_edge(passer, receiver, weight=avg_xt)
            else:
                defense_g.add_edge(passer, receiver, weight=abs(avg_xt))

    return basic_g, attack_g, defense_g, positions

# 3. Visualization with mplsoccer
def draw_network(G, positions, title="Passing Network", edge_weights=True, figsize=(12, 8)):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # Draw edges (passes)
    for u, v in G.edges():
        x1, y1 = positions.get(u, (0, 0))
        x2, y2 = positions.get(v, (0, 0))
        weight = G[u][v]['weight']
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                lw=max(1.5, weight),
                alpha=0.6
            )
        )

    # Draw nodes (players)
    for node, (x, y) in positions.items():
        ax.scatter(x, y, s=1200, color='#1f78b4', edgecolors='black', zorder=3)
        ax.text(x, y, node.split()[0], ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 4. Graph Cost, Path, Centralities
def compute_realistic_cost(G, positions, accuracy):
    clustering = nx.clustering(G.to_undirected())
    pagerank = nx.pagerank(G)
    alpha, beta, gamma, delta, epsilon = 1.0, 1.0, 1.0, 0.5, 0.5

    for u, v in G.edges():
        freq = G[u][v]['weight']
        freq_cost = 1 / (freq + 1e-6)
        dist = 0.5
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            dist = math.hypot(x2 - x1, y2 - y1) / 100
        acc_penalty = 1 - accuracy.get(u, 0.8)
        cluster_penalty = 1 / (clustering.get(u, 0.5) + 1e-6)
        pagerank_target = 1 / (pagerank.get(v, 0.1) + 1e-6)

        cost = (
            alpha * freq_cost +
            beta * dist +
            gamma * acc_penalty +
            delta * cluster_penalty +
            epsilon * pagerank_target
        )

        G[u][v]['cost'] = cost

    return G

def get_shortest_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target, weight='cost')
        cost = sum(G[u][v]['cost'] for u, v in zip(path[:-1], path[1:]))
        return path, cost
    except nx.NetworkXNoPath:
        return [], float('inf')

def analyze_centralities(G):
    return {
        'degree': dict(G.degree()),
        'in_degree': dict(G.in_degree()),
        'out_degree': dict(G.out_degree()),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'pagerank': nx.pagerank(G)
    }
