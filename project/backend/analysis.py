import json
import requests
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from heapq import heappush, heappop
from mplsoccer import Pitch
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Data Loading & Helper Functions

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

def load_event_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def extract_all_starting_players(events):
    starting_info = {}
    for e in events:
        if e.get("type", {}).get("name") == "Starting XI":
            team = e.get("team", {}).get("name")
            players = set()
            for p in e["tactics"]["lineup"]:
                players.add(p["player"]["name"])
            starting_info[team] = players
    return starting_info

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


def build_passing_graph_with_xt(events, starting_info):
    xT_matrix = np.load("xT_map.npy")

    basic_g = nx.DiGraph()
    attack_g = nx.DiGraph()
    defense_g = nx.DiGraph()
    location_data = defaultdict(list)
    player_team = {}

    pass_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    xT_totals  = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for e in events:
        if e.get("type", {}).get("name") != "Pass":
            continue
        team = e.get("team", {}).get("name")
        if team not in starting_info:
            continue
        p = e.get("pass", {})
        if "outcome" in p or "end_location" not in p or "recipient" not in p:
            continue
        if "location" not in e:
            continue

        passer = e["player"]["name"]
        receiver = p["recipient"]["name"]
        if passer not in starting_info[team] or receiver not in starting_info[team]:
            continue

        pass_counts[team][passer][receiver] += 1
        gs = get_grid_cell(*e["location"])
        ge = get_grid_cell(*p["end_location"])
        xT_gain = xT_matrix[ge] - xT_matrix[gs]
        xT_totals[team][passer][receiver] += xT_gain

        location_data[passer].append(tuple(e["location"]))
        location_data[receiver].append(tuple(p["end_location"]))

        player_team[passer] = team
        player_team[receiver] = team

    for team, players_dict in pass_counts.items():
        for passer, receivers in players_dict.items():
            for receiver, count in receivers.items():
                if count == 0:
                    continue
                avg_xt = xT_totals[team][passer][receiver] / count

                basic_g.add_edge(passer, receiver, weight=count, team=team)
                if avg_xt > 0:
                    attack_g.add_edge(passer, receiver, weight=avg_xt, team=team)
                else:
                    defense_g.add_edge(passer, receiver, weight=abs(avg_xt), team=team)

    positions = {
        player: (
            sum(x for x, y in locs) / len(locs),
            sum(y for x, y in locs) / len(locs)
        )
        for player, locs in location_data.items() if locs
    }

    return basic_g, attack_g, defense_g, positions, player_team

def draw_network(G, positions, player_team, selected_team,
                 title="Passing Network", figsize=(12, 8)):

    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    pitch_length = 120

    positions_plot = {}
    for player, (x, y) in positions.items():
        if player_team.get(player) == selected_team:
            positions_plot[player] = (x, y)
        else:
            positions_plot[player] = (pitch_length - x, y)

    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0
    tl = title.lower()

    for u, v, data in G.edges(data=True):
        team = data.get('team')
        w = data['weight']
        width = (w / max_w) * 5

        if team == selected_team:
            if 'attack' in tl:
                edge_color = "#0F6C2E"
            elif 'defens' in tl:
                edge_color = "#FF6B6B"
            else:
                edge_color = "#000000"
            alpha = 0.8
        else:
            edge_color = "#CCCCCC"
            alpha = 0.3

        x1, y1 = positions_plot[u]
        x2, y2 = positions_plot[v]

        if 'attack' in tl or 'defens' in tl:
            arr = FancyArrowPatch(
                posA=(x1, y1), posB=(x2, y2),
                arrowstyle='-|>', shrinkA=0, shrinkB=15,
                mutation_scale=10 + width * 2,
                linewidth=width, color=edge_color,
                alpha=alpha, zorder=2
            )
            ax.add_patch(arr)
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=edge_color,
                    linewidth=width,
                    alpha=alpha,
                    zorder=1)

    for player, (x, y) in positions_plot.items():
        team = player_team.get(player)
        if team == selected_team:
            node_color = "#2B2B67"
            outline_color = "#0C1B2B"
            text_color = 'white'
        else:
            node_color = "#E0E0E0"
            outline_color = "#888888"
            text_color = 'black'

        ax.scatter(x, y,
                   s=1200,
                   color=node_color,
                   edgecolors=outline_color,
                   linewidths=2,
                   zorder=3)
        
        name_parts = player.split()
        last_name = name_parts[-1] if len(name_parts) > 1 else player

        ax.text(x, y,
                last_name,
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color=text_color,
                zorder=4)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    return fig

# Animation for Shortest Path

def animate_shortest_path(G, positions, path, interval=800, figsize=(12, 8)):
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
        #blit=True,
        blit=False,
        repeat=False
    )

    # plt.show()
    return ani


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
    
# Build weighted graph for pathfinding
def build_weighted_graph(events, starting_players, team_name):
    xT_matrix = np.load("xT_map.npy")
    player_data = defaultdict(lambda: {"positions": [], "xG": 0, "shots": 0})
    for e in events:
        if e.get("team", {}).get("name") != team_name: continue
        player = e.get("player", {}).get("name")
        if player not in starting_players: continue
        loc = e.get("location")
        if loc and len(loc)>=2:
            player_data[player]["positions"].append(loc)
        if e.get("type", {}).get("name") == "Shot":
            player_data[player]["xG"] += e.get("shot",{}).get("statsbomb_xg",0)
            player_data[player]["shots"] += 1
    positions = {}
    xG_rates = {}
    for p in starting_players:
        data = player_data[p]
        if data["positions"]:
            xs=[pos[0] for pos in data["positions"]]; ys=[pos[1] for pos in data["positions"]]
            positions[p] = (np.mean(xs), np.mean(ys))
        if data["shots"]>0:
            xG_rates[p] = data["xG"]/data["shots"]
    xT_avg = compute_pass_xT_gain_avg(events, starting_players, team_name)
    G = nx.DiGraph()
    for u, recs in xT_avg.items():
        for v, avg_xt in recs.items():
            grid = get_grid_cell(*positions.get(v,(0,0)))
            pot = 0.2*xT_matrix[grid] + 0.1*xG_rates.get(v,0)
            weight = avg_xt + pot
            G.add_edge(u, v, weight=weight)
    return G, positions

# Find top optimal xT paths
def xT_based_goal_pathfinding(G, origin_player, max_passes=3):
    heap=[]
    heappush(heap,(0, origin_player, [origin_player], 0))
    results=[]
    while heap:
        neg_val, curr, path, depth = heappop(heap)
        val = -neg_val
        if depth>max_passes: continue
        if depth>0 and val>0: results.append((path, val))
        for nbr in G.successors(curr):
            if nbr in path: continue
            w = G[curr][nbr]['weight']
            heappush(heap, (-(val+w), nbr, path+[nbr], depth+1))
    top = sorted(results, key=lambda x: -x[1])[:5]
    return [{"path":p, "total_xT":v} for p, v in top]

# Animate a given path on the pitch
def animate_path(G, positions, path, interval=800, title='Top xT Path (Animasi)', figsize=(12,8)):
    pitch=Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)
    steps=[]
    for i, p in enumerate(path):
        steps.append(('node', p))
        if i< len(path)-1: steps.append(('edge', (p, path[i+1])))
    artists=[]
    def update(frame):
        kind, itm = steps[frame]
        if kind=='node':
            x,y=positions.get(itm,(60,40))
            dot=ax.scatter(x,y, s=1200, facecolor='#005f73', edgecolors='black', linewidths=2, zorder=4)
            txt=ax.text(x,y,itm.split()[0], ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=5)
            artists.extend([dot, txt])
        else:
            u,v=itm; x1,y1=positions.get(u,(60,40)); x2,y2=positions.get(v,(60,40))
            ln,=ax.plot([x1,x2],[y1,y2], linewidth=4, alpha=0.8, zorder=3)
            artists.append(ln)
        return artists
    ani=animation.FuncAnimation(fig, update, frames=len(steps), interval=interval, blit=True, repeat=False)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    return ani


def analyze_centralities(G):
    return {
        'degree': dict(G.degree()),
        'in_degree': dict(G.in_degree()),
        'out_degree': dict(G.out_degree()),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'pagerank': nx.pagerank(G)
    }
