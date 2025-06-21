import json
import requests
import networkx as nx
import math
import numpy as np
from collections import defaultdict

# # untuk load file local (kalo ada)
# def load_event_data(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

def get_url_mapping():
    return {
        "Barcelona vs Atletico Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773372.json",
        "Barcelona vs Real Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773585.json",
        "Barcelona vs Sevilla": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773672.json",
        "Spain vs Russia" : "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/7582.json"
    }

# load file data dari urls (hugging face)
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
        # Pastikan hanya starting player yang dicek
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

    positions = {
        p: (sum(x for x, y in locs) / len(locs), sum(y for x, y in locs) / len(locs))
        for p, locs in location_data.items() if locs
    }

    accuracy = {p: pass_success[p] / pass_total[p] for p in pass_total if pass_total[p] > 0}

    return G, positions, accuracy

def get_grid_cell(x, y):
    """Ubah koordinat lapangan ke index grid lapangan"""
    pitch_length, pitch_width = 120, 80
    grid_length, grid_width = 16, 12
    
    x_norm = max(0, min(x, pitch_length)) # [0, 120)
    y_norm = max(0, min(y, pitch_width)) # [0, 80)
    i = int((x_norm / pitch_length) * grid_length) # [0, 16)
    j = int((y_norm / pitch_width) * grid_width) # [0, 12)
    return min(i, grid_length - 1), min(j, grid_width - 1)

# TODO:
# 5. xT path finding

def compute_pass_xT_gain_avg(events, starting_players, team_name="Barcelona"):
    # TODO: Take xT of all passes, then see which passes gain the most xT
    # GA BISA CUMA ambil avg position trus cocokin sama grid.
    # Liat dlu average xT gain dari kombinasi2 pass DIMANAPUN pass itu dilakukan di lapangan.

    # Allow cek Barcelona aja
    if team_name != "Barcelona":
        print("Maaf, baru ada xT map untuk Barcelona saja.")
        print("Pilih tim Barcelona untuk menampilkan attacking-weighted passing network.")
        return

    xT_matrix = np.load("barcelona_xt.npy") # working
    
    pass_total = defaultdict(lambda: defaultdict(int))  # pass_total[passer][receiver] = count
    xT_total = defaultdict(lambda: defaultdict(float))  # xT_total[passer][receiver] = sum(xT_gain)

    for e in events:
        # Ambil pass aja
        if e.get("type", {}).get("name") != "Pass": continue
        # Ambil yang dari tim diinginkan aja
        if e.get("team", {}).get("name") != team_name: continue
        # Ambil pass yang complete (ada penerima pass)
        if "recipient" not in e.get("pass", {}): continue

        passer = e["player"]["name"]
        receiver = e["pass"]["recipient"]["name"]
        # Ambil dari starting players aja, non-cadangan
        if passer in starting_players and receiver in starting_players:
            pass_total[passer][receiver] += 1

            pitch_start = e.get("location", [0, 0])
            pitch_end = e.get("pass", {}).get("end_location", [0, 0])
            # Konversi ke koordinat grid xT map
            grid_start = get_grid_cell(pitch_start[0], pitch_start[1]) # Pastiin dia dapat dlm list
            grid_end = get_grid_cell(pitch_end[0], pitch_end[1])

            # Compute dengan value2 di xT map
            xT_gain = xT_matrix[grid_end] - xT_matrix[grid_start]
            xT_total[passer][receiver] += xT_gain
    
    xT_avg = defaultdict(lambda: defaultdict(float))
    for passer in pass_total:
        for receiver in pass_total[passer]:
            count = pass_total[passer][receiver]
            if count > 0:
                xT_avg[passer][receiver] = xT_total[passer][receiver] / count

    return xT_avg

# NOTE: KHUSUS BARCELONA karena pakai xT Barca
def build_attacking_weighted_graph(events, starting_players, team_name="Barcelona", threshold=0):
    """
    Build a directed graph weighted by average xT gain per pass.
    Positive weights indicate passes that increase expected threat.
    """
    xT_avg = compute_pass_xT_gain_avg(events, starting_players, team_name)
    
    G = nx.DiGraph()
    
    # Add edges with xT gain as weights
    for passer in xT_avg:
        for receiver in xT_avg[passer]:
            avg_xt_gain = xT_avg[passer][receiver]
            if avg_xt_gain > threshold:  # Only include passes with positive xT gain
                G.add_edge(passer, receiver, weight=avg_xt_gain)
    
    return G

def build_defensive_weighted_graph(events, starting_players, team_name="Barcelona", threshold=0):
    """
    Build a directed graph weighted by defensive value (negative xT gain).
    Positive weights indicate passes that decrease opponent's threat.
    """
    xT_avg = compute_pass_xT_gain_avg(events, starting_players, team_name)
    
    G = nx.DiGraph()
    
    # Add edges with inverse xT gain (defensive value)
    for passer in xT_avg:
        for receiver in xT_avg[passer]:
            defensive_value = -xT_avg[passer][receiver]  # Flip the sign
            if defensive_value > threshold:  # Only include valuable defensive passes
                G.add_edge(passer, receiver, weight=defensive_value)
    
    return G

# Runner tester
urls = get_url_mapping()["Barcelona vs Real Madrid"]
events = load_event_data_from_url(urls)
players, _ = extract_starting_players(events, "Barcelona")
print(players)

xT_avg_counted = compute_pass_xT_gain_avg(events, players, team_name="Barcelona")
print(build_attacking_weighted_graph(events, players, team_name="Barcelona"))
print(build_defensive_weighted_graph(events, players, team_name="Barcelona"))

# def compute_realistic_cost(G, positions, accuracy):
#     clustering = nx.clustering(G.to_undirected())
#     pagerank = nx.pagerank(G)

#     alpha, beta, gamma, delta, epsilon = 1.0, 1.0, 1.0, 0.5, 0.5

#     for u, v in G.edges():
#         freq = G[u][v]['weight']
#         freq_cost = 1 / (freq + 1e-6)

#         dist = 0.5
#         if u in positions and v in positions:
#             x1, y1 = positions[u]
#             x2, y2 = positions[v]
#             dist = math.hypot(x2 - x1, y2 - y1) / 100

#         acc_penalty = 1 - accuracy.get(u, 0.8)
#         cluster_penalty = 1 / (clustering.get(u, 0.5) + 1e-6)
#         pagerank_target = 1 / (pagerank.get(v, 0.1) + 1e-6)

#         cost = (
#             alpha * freq_cost +
#             beta * dist +
#             gamma * acc_penalty +
#             delta * cluster_penalty +
#             epsilon * pagerank_target
#         )

#         G[u][v]['cost'] = cost

#     return G

# def analyze_centralities(G):
#     return {
#         'degree': dict(G.degree()),
#         'in_degree': dict(G.in_degree()),
#         'out_degree': dict(G.out_degree()),
#         'betweenness': nx.betweenness_centrality(G),
#         'closeness': nx.closeness_centrality(G),
#         'pagerank': nx.pagerank(G)
#     }

# def get_shortest_path(G, source, target):
#     try:
#         path = nx.shortest_path(G, source=source, target=target, weight='cost')
#         cost = sum(G[u][v]['cost'] for u, v in zip(path[:-1], path[1:]))
#         return path, cost
#     except nx.NetworkXNoPath:
#         return [], float('inf')

# # TODO: EDIT HERE xT
# def xT_based_goal_pathfinding(_):
#     # NOTE: cara hitung xT (position-based):
#     # xT di x,y =
#     # (potensi shoot di x,y * potensi gol dari x,y)
#     # + (potensi move di x,y * sigma potensi move ke semua area lapangan * xT di x,y)
#     # yes, ini ada rekursi xT di x,y (prediksi 5 move ke depan)
    
#     # problem
#     # kan xT ada potensi gol dari x,y, aneh ya kalo potensi gol dari x,y di avg position player?
#     # avg position kan kemungkinan pada dari posisi yg ga cocok buat shooting
#     # masalahnya kan bisa gerak ke posisi yg bagus buat shoot dan dihitung dlm xT
#     # btw TODO: avg position kita hitungnya gmn?
#         # mending ambil pass & shot aja
#     # mending hitung pake rata2 xG player per 90 aja?
#     # OR, which player has the best average position? (tetep pake avg position)

#     # SOLN: pake xG per 90
#     # NOTE: xG setiap shot sudah ada di dataset statsbomb
#     # XXX: Optional: return bareng xT map utk kasi tau posisi terbaik depan gawang

#     return #temp

# # if __name__ == "__main__": 
# #     print ("Executed when invoked directly")