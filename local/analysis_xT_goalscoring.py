import json
import requests
import networkx as nx
import math
import numpy as np
from collections import defaultdict
from heapq import heappush, heappop
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load mapping of match names to URLs
def get_url_mapping():
    return {
        "Atletico Madrid vs Barcelona (2020/21 La Liga)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773372.json",
        "Barcelona vs Real Madrid (2020/21 La Liga)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773585.json",
        "Barcelona vs Sevilla (2020/21 La Liga)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773672.json",
        "Spain vs Russia (2018 World Cup RO16)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/7582.json",
        "Argentina vs France (2022 World Cup Final)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3869685.json",
        "Liverpool vs Milan (2005 UCL Final)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/2302764.json",
        "Manchester United vs Arsenal (2003/04 Premier League)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3749246.json",
        "Arsenal vs Leicester City (2003/04 Premier League)": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3749257.json"
    }

# Load event data from URL
def load_event_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Extract starting XI players for a given team
def extract_starting_players(events, team_name):
    players = []
    seen = set()
    for e in events:
        if e.get("type", {}).get("name") == "Starting XI" and e.get("team", {}).get("name") == team_name:
            for p in e.get("tactics", {}).get("lineup", []):
                name = p.get("player", {}).get("name")
                if name and name not in seen:
                    seen.add(name)
                    players.append(name)
    return players

# Convert pitch coords to xT grid index
def get_grid_cell(x, y):
    pitch_length, pitch_width = 120, 80
    grid_length, grid_width = 16, 12
    x_norm = max(0, min(x, pitch_length))
    y_norm = max(0, min(y, pitch_width))
    i = int((x_norm / pitch_length) * grid_length)
    j = int((y_norm / pitch_width) * grid_width)
    return min(i, grid_length - 1), min(j, grid_width - 1)

# Compute average xT gain per pass between players
def compute_pass_xT_gain_avg(events, starting_players, team_name):
    xT_matrix = np.load("xT_map.npy")
    pass_total = defaultdict(lambda: defaultdict(int))
    xT_total = defaultdict(lambda: defaultdict(float))
    for e in events:
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        p = e.get("pass", {})
        if "recipient" not in p: continue
        passer = e.get("player", {}).get("name")
        receiver = p.get("recipient", {}).get("name")
        if passer not in starting_players or receiver not in starting_players: continue
        pass_total[passer][receiver] += 1
        start_loc = e.get("location", [0, 0])
        end_loc = p.get("end_location", [0, 0])
        gs = get_grid_cell(start_loc[0], start_loc[1])
        ge = get_grid_cell(end_loc[0], end_loc[1])
        xT_gain = xT_matrix[ge] - xT_matrix[gs]
        xT_total[passer][receiver] += xT_gain
    xT_avg = defaultdict(lambda: defaultdict(float))
    for passer, recs in pass_total.items():
        for receiver, count in recs.items():
            if count > 0:
                xT_avg[passer][receiver] = xT_total[passer][receiver] / count
    return xT_avg

# Build weighted graph for pathfinding
def build_weighted_graph(events, starting_players, team_name):
    xT_matrix = np.load("xT_map.npy")
    player_data = defaultdict(lambda: {"positions": [], "xG": 0, "shots": 0})
    for e in events:
        if e.get("team", {}).get("name") != team_name: continue
        player = e.get("player", {}).get("name")
        if player not in starting_players: continue
        loc = e.get("location")
        if loc and len(loc) >= 2:
            player_data[player]["positions"].append(loc)
        if e.get("type", {}).get("name") == "Shot":
            player_data[player]["xG"] += e.get("shot", {}).get("statsbomb_xg", 0)
            player_data[player]["shots"] += 1
    positions, xG_rates = {}, {}
    for p in starting_players:
        data = player_data[p]
        if data["positions"]:
            xs = [pos[0] for pos in data["positions"]]
            ys = [pos[1] for pos in data["positions"]]
            positions[p] = (np.mean(xs), np.mean(ys))
        if data["shots"] > 0:
            xG_rates[p] = data["xG"] / data["shots"]
    xT_avg = compute_pass_xT_gain_avg(events, starting_players, team_name)
    G = nx.DiGraph()
    for u, receivers in xT_avg.items():
        for v, avg_xt in receivers.items():
            grid = get_grid_cell(*positions.get(v, (0,0)))
            pot = 0.2 * xT_matrix[grid] + 0.1 * xG_rates.get(v, 0)
            weight = avg_xt + pot
            G.add_edge(u, v, weight=weight)
    return G, positions

def xT_based_goal_pathfinding(G, origin_player, max_passes=3):
    heap = []
    heappush(heap, (0, origin_player, [origin_player], 0))
    results = []
    while heap:
        neg_val, current, path, depth = heappop(heap)
        val = -neg_val
        if depth > max_passes: continue
        if val > 0 and depth > 0: results.append((path, val))
        for nbr in G.successors(current):
            if nbr in path: continue
            w = G[current][nbr]['weight']
            heappush(heap, (-(val + w), nbr, path + [nbr], depth + 1))
    top = sorted(results, key=lambda x: -x[1])[:5]
    return [{"path": p, "total_xT": v} for p, v in top]

# Compute realistic pass cost
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

# Safe but dangerous pathfinder (maximize xT/cost)
def safe_dangerous_goal_path(G, origin_player, positions, accuracy, max_passes=3, top_k=5):
    G = compute_realistic_cost(G, positions, accuracy)
    heap = []
    heappush(heap, (0, 0, origin_player, [origin_player], 0))
    results = []
    while heap:
        neg_score, total_cost, current, path, depth = heappop(heap)
        score = -neg_score
        if depth > max_passes:
            continue
        if score > 0 and depth > 0:
            results.append((score, total_cost, path))
        for nbr in G.successors(current):
            if nbr in path:
                continue
            xT = G[current][nbr]['weight']
            cost = G[current][nbr].get('cost', 1e-6)
            new_xT = score * total_cost + xT
            new_cost = total_cost + cost
            new_score = new_xT / new_cost if new_cost > 0 else 0
            heappush(heap, (-new_score, new_cost, nbr, path + [nbr], depth + 1))
    top = sorted(results, key=lambda x: -x[0])[:top_k]
    return [{"path": p, "xT": round(s * c, 4), "cost": round(c, 4), "xT_per_cost": round(s, 4)} for s, c, p in top]

# Animate a given path on the pitch
def animate_path(G, positions, path, interval=800, title='Top xT Path (Animasi)', figsize=(12,8)):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # Prepare animation steps
    steps = []
    for i, node in enumerate(path):
        steps.append(('node', node))
        if i < len(path) - 1:
            steps.append(('edge', (node, path[i+1])))

    artists = []
    def update(frame):
        kind, item = steps[frame]
        if kind == 'node':
            x, y = positions[item]
            artist = ax.scatter(x, y, s=1200, facecolor="#3333C8", edgecolors="black", linewidths=2, zorder=4)
            name_parts = item.split()
            last_name = name_parts[-1] if len(name_parts) > 0 else item
            text = ax.text(x, y, last_name, ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=5)
            artists.extend([artist, text])
        else:
            u, v = item
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            line, = ax.plot([x1, x2], [y1, y2], linewidth=4, color='black', alpha=0.8, zorder=3)
            artists.append(line)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=interval, blit=True, repeat=False)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    return ani

# Main runner
def main():
    urls = get_url_mapping()
    print("\n📁 Daftar Pertandingan:")
    matches = list(urls.keys())
    for i, m in enumerate(matches, 1):
        print(f"{i}. {m}")
    try:
        idx = int(input("Pilih nomor pertandingan: ")) - 1
        match_name = matches[idx]
    except:
        print("❌ Pilihan tidak valid.")
        return

    events = load_event_data_from_url(urls[match_name])

    # Select team
    teams = sorted({e.get("team", {}).get("name") for e in events if e.get("team")})
    print("\n🔍 Tim yang tersedia:")
    for i, t in enumerate(teams, 1):
        print(f"{i}. {t}")
    try:
        team_name = teams[int(input("Pilih tim: ")) - 1]
    except:
        print("❌ Pilihan tidak valid.")
        return

    players = extract_starting_players(events, team_name)
    if not players:
        print(f"❌ Tidak ada Starting XI untuk {team_name}.")
        return

    print("\n👥 Starting XI:")
    for i, p in enumerate(players, 1):
        print(f"{i}. {p}")
    try:
        origin = players[int(input("Pilih pemain asal: ")) - 1]
    except:
        print("❌ Pilihan tidak valid.")
        return

    # Path preference
    print("\n🔍 Pilih jenis path:")
    print("1. Paling produktif (xT tertinggi)")
    print("2. Paling aman (memperhatikan akurasi & risiko)")
    try:
        mode = int(input("Pilihan: "))
    except:
        print("❌ Pilihan tidak valid.")
        return

    G, positions = build_weighted_graph(events, players, team_name)

    # Dummy accuracy assumption (real data can be substituted here)
    accuracy = {p: 0.85 for p in players}

    if mode == 1:
        optimal = xT_based_goal_pathfinding(G, origin)
        print(f"\n⚽ Top xT Paths dari {origin} ({team_name}):")
        for i, res in enumerate(optimal, 1):
            print(f"\nPath {i}: {' → '.join(res['path'])}\nTotal xT: {res['total_xT']:.4f}")
    else:
        optimal = safe_dangerous_goal_path(G, origin, positions, accuracy)
        print(f"\n🛡️  Top Safe Goal Paths dari {origin} ({team_name}):")
        for i, res in enumerate(optimal, 1):
            print(f"\nPath {i}: {' → '.join(res['path'])}")
            print(f"Total xT: {res['xT']:.4f}, Cost: {res['cost']:.4f}, xT/Cost: {res['xT_per_cost']:.4f}")

    if optimal:
        print("\n🔍 Visualisasi Path 1 (Animasi):")
        animate_path(G, positions, optimal[0]['path'], interval=800,
                     title=f"Top Path dari {origin} (Visualisasi)")

if __name__ == "__main__":
    main()