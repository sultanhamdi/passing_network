import json
import requests
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. Data Loading Functions
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

# 2. xT Calculation Functions
def get_grid_cell(x, y, pitch_length=120, pitch_width=80, grid_length=16, grid_width=12):
    x_norm = max(0, min(x, pitch_length))
    y_norm = max(0, min(y, pitch_width))
    i = int((x_norm / pitch_length) * grid_length)
    j = int((y_norm / pitch_width) * grid_width)
    return min(i, grid_length-1), min(j, grid_width-1)

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

# 3. Graph Building Functions
def build_passing_graph_with_xt(events, starting_players, team_name="Barcelona"):
    xT_avg = compute_pass_xT_gain_avg(events, starting_players, team_name)
    location_data = defaultdict(list)
    
    # Collect positions
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

    # Build graphs
    basic_g = nx.DiGraph()
    attack_g = nx.DiGraph()
    defense_g = nx.DiGraph()

    for passer in xT_avg:
        for receiver in xT_avg[passer]:
            avg_xt = xT_avg[passer][receiver]
            count = len([e for e in events 
                        if e.get("player", {}).get("name") == passer 
                        and e.get("pass", {}).get("recipient", {}).get("name") == receiver])
            
            basic_g.add_edge(passer, receiver, weight=count)
            if avg_xt > 0:
                attack_g.add_edge(passer, receiver, weight=avg_xt)
            else:
                defense_g.add_edge(passer, receiver, weight=abs(avg_xt))

    return basic_g, attack_g, defense_g, positions

# 4. Visualization Functions
def draw_network(G, positions, title, edge_weights=True, figsize=(12,8)):
    plt.figure(figsize=figsize)
    
    node_size = [2000 + 500 * G.degree(node) for node in G.nodes()]
    edge_width = [d['weight']*3 for u,v,d in G.edges(data=True)] if edge_weights else 1.5
    
    nx.draw_networkx_nodes(G, positions, node_size=node_size, node_color='#1f78b4', alpha=0.8)
    nx.draw_networkx_edges(G, positions, width=edge_width, edge_color='#666666', 
                          alpha=0.6, arrowsize=20, arrowstyle='->')
    nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 5. Main Runner
def analyze_match(match_name, team_name="Barcelona"):
    print(f"\nAnalyzing {match_name}...")
    url = get_url_mapping()[match_name]
    events = load_event_data_from_url(url)
    players, _ = extract_starting_players(events, team_name)
    
    basic_g, attack_g, defense_g, positions = build_passing_graph_with_xt(events, players, team_name)
    
    print("\nBasic Passing Network:")
    draw_network(basic_g, positions, f"{team_name} Passing Frequency")
    
    print("\nAttacking xT Network:")
    draw_network(attack_g, positions, f"{team_name} Attacking Threat")
    
    print("\nDefensive Value Network:")
    draw_network(defense_g, positions, f"{team_name} Defensive Passing")
    
    return basic_g, attack_g, defense_g

# Run analysis
if __name__ == "__main__":
    # Make sure 'barcelona_xt.npy' is in your working directory
    analyze_match("Barcelona vs Real Madrid")