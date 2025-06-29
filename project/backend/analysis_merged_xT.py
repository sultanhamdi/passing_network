import json
import requests
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from mplsoccer import Pitch
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation

# Data Loading & Helper Functions

def get_url_mapping():
    return {
        "Barcelona vs Atletico Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773372.json",
        "Barcelona vs Real Madrid": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773585.json",
        "Barcelona vs Sevilla": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3773672.json",
        "Spain vs Russia": "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/7582.json",
        "Argentina vs France" : "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3869685.json",
        "Liverpool vs Milan" : "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/2302764.json",
        "Arsenal vs Manchester United":"https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3749246.json",
        "Arsenal vs Leicester City" : "https://huggingface.co/datasets/sultanhamdi/passnet/resolve/main/3749257.json"
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

# xT-based Graph Construction
def compute_pass_xT_gain_avg(events, starting_players, team_name="Barcelona"):
    xT_matrix = np.load("xT_map.npy")
    pass_counts = defaultdict(lambda: defaultdict(int))
    xT_total = defaultdict(lambda: defaultdict(float))

    for e in events:
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        p = e.get("pass", {})
        # hanya successful passes dan pastikan end_location ada
        if "outcome" in p or "end_location" not in p or "recipient" not in p:
            continue

        passer = e["player"]["name"]
        receiver = p["recipient"]["name"]
        if passer in starting_players and receiver in starting_players:
            pass_counts[passer][receiver] += 1
            grid_start = get_grid_cell(*e["location"])
            grid_end = get_grid_cell(*p["end_location"])
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
        if e.get("type", {}).get("name") != "Pass": continue
        if e.get("team", {}).get("name") != team_name: continue
        p = e.get("pass", {})
        # hanya successful passes dengan end_location
        if "outcome" in p or "end_location" not in p or "recipient" not in p:
            continue
        if "location" not in e:
            continue

        passer = e["player"]["name"]
        receiver = p["recipient"]["name"]
        if passer in starting_players and receiver in starting_players:
            location_data[passer].append(e["location"])
            location_data[receiver].append(p["end_location"])

    positions = {
        player: (
            sum(x for x, y in locs) / len(locs),
            sum(y for x, y in locs) / len(locs)
        )
        for player, locs in location_data.items() if locs
    }

    basic_g = nx.DiGraph()
    attack_g = nx.DiGraph()
    defense_g = nx.DiGraph()

    for passer, receivers in xT_avg.items():
        for receiver, avg_xt in receivers.items():
            # hitung jumlah successful passes
            count = sum(
                1 for e in events
                if e.get("type", {}).get("name") == "Pass"
                   and e.get("team", {}).get("name") == team_name
                   and e["player"]["name"] == passer
                   and e["pass"].get("recipient", {}).get("name") == receiver
                   and "outcome" not in e["pass"]
            )
            if count == 0:
                continue
            
            # Graph builder
            basic_g.add_edge(passer, receiver, weight=count)
            if avg_xt > 0:
                attack_g.add_edge(passer, receiver, weight=avg_xt)
            else:
                defense_g.add_edge(passer, receiver, weight=abs(avg_xt))

            # Potential graph builder
            # basic_g.add_edge(passer, receiver, weight=count)
            # attack_g.add_edge(passer, receiver, weight=avg_xt)
            # defense_g.add_edge(passer, receiver, weight=(avg_xt * -1))

    return basic_g, attack_g, defense_g, positions

# Visualization Graph

def draw_network(G, positions, title="Passing Network", figsize=(12, 8)):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    max_w = max((G[u][v]['weight'] for u, v in G.edges()), default=1.0)
    tl = title.lower()

    for u, v in G.edges():
        x1, y1 = positions.get(u, (0, 0))
        x2, y2 = positions.get(v, (0, 0))
        w = G[u][v]['weight']
        width = (w / max_w) * 5

        # tentukan warna sesuai mode
        if 'attack' in tl:
            edge_color = "#0F6C2E"
        elif 'defens' in tl:
            edge_color = "#FF6B6B"
        else:
            edge_color = "#000000"

        if 'attack' in tl or 'defens' in tl:
            # panah dengan head dijauhkan sebelum node
            arr = FancyArrowPatch(
                posA=(x1, y1),
                posB=(x2, y2),
                arrowstyle='-|>',
                shrinkA=0,         # tidak shrink di tail
                shrinkB=15,        # geser head 15 poin sebelum node
                mutation_scale=10 + width * 2,
                linewidth=width,
                color=edge_color,
                alpha=0.8,
                zorder=2
            )
            ax.add_patch(arr)
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=edge_color,
                    linewidth=width,
                    alpha=0.6,
                    zorder=1)

    # gambar node di atas semua edge
    node_color = "#2B2B67"
    outline_color = "#0C1B2B"
    outline_width = 2
    for player, (x, y) in positions.items():
        ax.scatter(x, y,
                   s=1200,
                   color=node_color,
                   edgecolors=outline_color,
                   linewidths=outline_width,
                   zorder=3)
        ax.text(x, y,
                player.split()[0],
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white',
                zorder=4)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    # plt.show()
    return fig

def draw_shortest_path(G, positions, path, title="Shortest Path", figsize=(12, 8)):
    from mplsoccer import Pitch
    from matplotlib.patches import FancyArrowPatch

    # 1. Buat subgraph hanya dengan edge di path
    subG = nx.DiGraph()
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            subG.add_edge(u, v, weight=G[u][v]['weight'])

    # 2. Gambar pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # 3. Plot panah sesuai bobot
    max_w = max((subG[u][v]['weight'] for u, v in subG.edges()), default=1.0)
    for u, v in subG.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        w = subG[u][v]['weight']
        width = (w / max_w) * 5
        arr = FancyArrowPatch(
            posA=(x1, y1),
            posB=(x2, y2),
            arrowstyle='-|>',
            shrinkA=0,               # jangan shrink di tail
            shrinkB=15,              # geser kepala panah 15 poin dari node
            mutation_scale=10 + width*2,
            linewidth=width,
            color= 'black',
            alpha=0.8,
            zorder=4
        )
        ax.add_patch(arr)

    # 4. Plot node path
    for player in path:
        x, y = positions[player]
        ax.scatter(x, y,
                   s=1200,
                   color='#2B2B67',
                   edgecolors='#2C3E50',
                   linewidths=2,
                   zorder=3)
        ax.text(x, y, player.split()[0],
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white', zorder=4)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Animation for Shortest Path

def animate_shortest_path(G, positions, path, interval=800, figsize=(12, 8)):
    from mplsoccer import Pitch
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # Buat pitch dan axes
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # Siapkan urutan langkah: node → edge → node → …
    steps = []
    for i, player in enumerate(path):
        steps.append(('node', player))
        if i < len(path) - 1:
            steps.append(('edge', (player, path[i + 1])))

    artists = []

    def update(frame):
        kind, item = steps[frame]

        if kind == 'node':
            # Gambar node sebagai titik hitam
            x, y = positions[item]
            artist = ax.scatter(
                x, y,
                s=1200,
                color='black',
                edgecolors='white',
                linewidths=2,
                zorder=4
            )
            text = ax.text(
                x, y,
                item.split()[0],
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white',
                zorder=5
            )
            artists.extend([artist, text])

        else:
            # Gambar edge sebagai garis hitam
            u, v = item
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            w = G[u][v]['weight']
            max_w = max((G[a][b]['weight'] for a, b in G.edges()), default=1.0)
            width = (w / max_w) * 5

            line, = ax.plot(
                [x1, x2], [y1, y2],
                linewidth=width,
                color='black',
                alpha=0.8,
                zorder=3
            )
            artists.append(line)

        return artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(steps),
        interval=interval,
        blit=True,
        repeat=False
    )

    plt.show()
    return ani


# Graph Cost, Path, Centralities

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


def get_shortest_path(G, source, target, positions, accuracy):
    G = compute_realistic_cost(G, positions, accuracy)
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
