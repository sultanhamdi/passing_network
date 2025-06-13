
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import math
from mplsoccer import Pitch

def load_graph_with_context(file_path, threshold=1):
    with open(file_path, "r", encoding="utf-8") as f:
        events_data = json.load(f)

    starting_players = set()
    for event in events_data:
        if event.get("type", {}).get("name") == "Starting XI" and event.get("team", {}).get("name") == "Barcelona":
            for player in event["tactics"]["lineup"]:
                starting_players.add(player["player"]["name"])

    passes = []
    player_pass_total = defaultdict(int)
    player_pass_success = defaultdict(int)
    player_locations = defaultdict(list)

    for event in events_data:
        if (
            event.get("type", {}).get("name") == "Pass"
            and event.get("team", {}).get("name") == "Barcelona"
            and "recipient" in event.get("pass", {})
        ):
            passer = event["player"]["name"]
            receiver = event["pass"]["recipient"]["name"]
            if passer in starting_players and receiver in starting_players:
                passes.append((passer, receiver))
                player_pass_total[passer] += 1
                if event.get("pass", {}).get("outcome") is None:
                    player_pass_success[passer] += 1
                    if event.get("location"):
                        player_locations[passer].append(event["location"])
                        player_locations[receiver].append(event["pass"]["end_location"])

    pass_counts = defaultdict(int)
    for passer, receiver in passes:
        pass_counts[(passer, receiver)] += 1

    G = nx.DiGraph()
    for (u, v), weight in pass_counts.items():
        if weight >= threshold:
            G.add_edge(u, v, weight=weight)

    G.remove_nodes_from(list(nx.isolates(G)))

    avg_positions = {}
    for player, locs in player_locations.items():
        if locs:
            xs, ys = zip(*locs)
            avg_positions[player] = (sum(xs)/len(xs), sum(ys)/len(ys))

    player_accuracy = {
        player: player_pass_success[player] / player_pass_total[player]
        for player in player_pass_total
        if player_pass_total[player] > 0
    }

    return G, avg_positions, player_accuracy

def realistic_heuristic(G, u, v, positions, accuracy):
    weight = G[u][v]['weight']
    freq_cost = 1 / (weight + 1e-6)
    if u in positions and v in positions:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dist = math.hypot(x2 - x1, y2 - y1) / 100
    else:
        dist = 0.5
    acc_penalty = 1 - accuracy.get(u, 0.8)
    return freq_cost + dist + acc_penalty

def build_weighted_graph(G, positions, accuracy):
    G_real = G.copy()
    for u, v in G_real.edges():
        G_real[u][v]['cost'] = realistic_heuristic(G, u, v, positions, accuracy)
    return G_real

def find_realistic_path(G_real, start, end):
    try:
        path = nx.shortest_path(G_real, source=start, target=end, weight='cost')
        total_cost = sum(G_real[u][v]['cost'] for u, v in zip(path[:-1], path[1:]))
        return path, total_cost
    except nx.NetworkXNoPath:
        return None, float('inf')

def animate_path(path, positions):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#3f995b', line_color='white',
                  stripe=True, stripe_color='#5c8a64')
    fig, ax = pitch.draw(figsize=(14, 8))
    fig.set_facecolor('#3f995b')

    dots = []
    arrows = []
    texts = []

    def update(frame):
        if frame < len(path):
            player = path[frame]
            x, y = positions.get(player, (60, 40))
            dot = pitch.scatter([x], [y], ax=ax, s=600, color='orange', edgecolors='black', zorder=3)
            label = ax.text(x, y, player.split()[0], ha='center', va='center', fontsize=10, color='white', zorder=4)
            dots.append(dot)
            texts.append(label)

        if frame > 0 and frame < len(path):
            x1, y1 = positions.get(path[frame - 1], (60, 40))
            x2, y2 = positions.get(path[frame], (60, 40))
            arrow = pitch.arrows(x1, y1, x2, y2, ax=ax, color='white', width=2, headwidth=5, zorder=2)
            arrows.append(arrow)

    anim = FuncAnimation(fig, update, frames=len(path) + 1, interval=1000, repeat=False)
    plt.tight_layout()
    plt.show()

def main():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "3773587.json")
    G, positions, accuracy = load_graph_with_context(file_path, threshold=1)
    G_real = build_weighted_graph(G, positions, accuracy)

    player_list = list(G_real.nodes())
    print("=== Pilih Pemain Berdasarkan Nomor ===")
    for i, name in enumerate(player_list):
        print(f"{i+1}. {name}")

    try:
        idx_start = int(input("\nPilih nomor pemain awal: ")) - 1
        idx_end = int(input("Pilih nomor pemain tujuan: ")) - 1

        start = player_list[idx_start]
        end = player_list[idx_end]
    except (ValueError, IndexError):
        print("Input tidak valid.")
        return

    path, cost = find_realistic_path(G_real, start, end)
    if path:
        print("\nJalur optimal berdasarkan heuristic realistis:")
        for i, p in enumerate(path):
            print(f"{i+1}. {p}")
        print(f"Total heuristic cost: {cost:.4f}")
        animate_path(path, positions)
    else:
        print("Tidak ada jalur tersedia.")

if __name__ == "__main__":
    main()
