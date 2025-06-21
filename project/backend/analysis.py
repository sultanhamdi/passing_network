import json
import requests
import networkx as nx
import math
from collections import defaultdict
from functools import lru_cache
import matplotlib.pyplot as plt

# Load file lokal
def load_event_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load file dari URL
@lru_cache(maxsize=10)
def load_event_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Ambil pemain Starting XI
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

# Buat graf berdasarkan frekuensi passing
def build_passing_graph(events, starting_players, team_name="Barcelona", threshold=1):
    passes = defaultdict(int)
    location_data = defaultdict(list)
    pass_total = defaultdict(int)
    pass_success = defaultdict(int)

    for e in events:
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        if "recipient" not in e.get("pass", {}): continue

        passer = e["player"]["name"]
        receiver = e["pass"]["recipient"]["name"]
        if passer in starting_players and receiver in starting_players:
            passes[(passer, receiver)] += 1
            pass_total[passer] += 1
            if e.get("pass", {}).get("outcome") is None:
                pass_success[passer] += 1
                if e.get("location"):
                    location_data[passer].append(e["location"])
                    location_data[receiver].append(e["pass"]["end_location"])

    G = nx.DiGraph()
    for (u, v), w in passes.items():
        if w >= threshold:
            G.add_edge(u, v, weight=w)

    # Rata-rata lokasi
    positions_raw = {
        p: (sum(x for x, y in locs) / len(locs), sum(y for x, y in locs) / len(locs))
        for p, locs in location_data.items() if locs
    }

    # Normalisasi (StatsBomb skala 120 x 80)
    positions = {
        p: (x * 1.0, y * 1.0)
        for p, (x, y) in positions_raw.items()
    }

    accuracy = {p: pass_success[p] / pass_total[p] for p in pass_total if pass_total[p] > 0}

    return G, positions, accuracy

# Buat graf menyerang berbobot
def build_attacking_weighted_graph(events, starting_players, team_name="Barcelona", threshold=1):
    passes = defaultdict(float)
    for e in events:
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        if "recipient" not in e.get("pass", {}): continue

        passer = e["player"]["name"]
        receiver = e["pass"]["recipient"]["name"]
        if passer in starting_players and receiver in starting_players:
            loc_start = e.get("location", [0, 0])
            loc_end = e.get("pass", {}).get("end_location", [0, 0])
            y_gain = loc_end[1] - loc_start[1]
            value = max(y_gain, 0)
            passes[(passer, receiver)] += value

    G = nx.DiGraph()
    for (u, v), weight in passes.items():
        if weight >= threshold:
            G.add_edge(u, v, weight=weight)
    return G

# Buat graf bertahan berbobot
def build_defensive_weighted_graph(events, starting_players, team_name="Barcelona", threshold=1):
    passes = defaultdict(float)
    for e in events:
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        if "recipient" not in e.get("pass", {}): continue

        passer = e["player"]["name"]
        receiver = e["pass"]["recipient"]["name"]
        if passer in starting_players and receiver in starting_players:
            loc_start = e.get("location", [0, 0])
            loc_end = e.get("pass", {}).get("end_location", [0, 0])
            y_loss = loc_start[1] - loc_end[1]
            value = max(y_loss, 0)
            passes[(passer, receiver)] += value

    G = nx.DiGraph()
    for (u, v), weight in passes.items():
        if weight >= threshold:
            G.add_edge(u, v, weight=weight)
    return G

# Hitung cost realistis untuk shortest path
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

# Shortest path dari A ke B
def get_shortest_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target, weight='cost')
        cost = sum(G[u][v]['cost'] for u, v in zip(path[:-1], path[1:]))
        return path, cost
    except nx.NetworkXNoPath:
        return [], float('inf')

# Centrality metrics
def analyze_centralities(G):
    return {
        'degree': dict(G.degree()),
        'in_degree': dict(G.in_degree()),
        'out_degree': dict(G.out_degree()),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'pagerank': nx.pagerank(G)
    }

# URL mapping dataset
def get_url_mapping():
    return {
        "Barcelona vs Atletico Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773372.json",
        "Barcelona vs Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773585.json",
        "Barcelona vs Seville": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773672.json",
        "Spain vs Russia": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/7582.json"
    }

# 🎯 Visualisasi passing graph
def visualize_passing_graph(G, positions, title="Passing Network"):
    plt.figure(figsize=(12, 8))
    plt.title(title)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=positions, node_size=1000, node_color="skyblue", edgecolors='k')

    # Draw edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos=positions, edge_color="gray", width=[w / max(weights) * 5 for w in weights])

    # Draw labels
    nx.draw_networkx_labels(G, pos=positions, font_size=10)

    plt.xlim(0, 120)
    plt.ylim(0, 80)
    plt.gca().set_facecolor("green")
    plt.axis("off")
    plt.show()
