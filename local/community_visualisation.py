import json
import requests
import networkx as nx
import community as community_louvain
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from collections import defaultdict

from analysis_merged_xT import get_url_mapping, load_event_data_from_url


def build_basic_graph_and_positions(events, team_name):
    """
    Bangun basic passing graph dan posisi rata-rata pemain berdasarkan event Pass.
    """
    G = nx.DiGraph()
    positions_data = defaultdict(list)

    for e in events:
        if e.get("type", {}).get("name") != "Pass":
            continue
        if e.get("team", {}).get("name") != team_name:
            continue
        p = e.get("pass", {})
        # hanya operan sukses
        if "recipient" not in p or "outcome" in p:
            continue
        passer = e["player"]["name"]
        receiver = p["recipient"]["name"]
        # hitung frekuensi
        if G.has_edge(passer, receiver):
            G[passer][receiver]["weight"] += 1
        else:
            G.add_edge(passer, receiver, weight=1)
        # simpan lokasi
        loc = e.get("location")
        if loc and len(loc) >= 2:
            positions_data[passer].append(tuple(loc))
        end_loc = p.get("end_location")
        if end_loc and len(end_loc) >= 2:
            positions_data[receiver].append(tuple(end_loc))

    # hitung posisi rata-rata
    positions = {
        player: (
            sum(x for x, y in locs) / len(locs),
            sum(y for x, y in locs) / len(locs)
        )
        for player, locs in positions_data.items() if locs
    }
    return G, positions


def detect_communities(G):
    """
    Deteksi komunitas menggunakan metode Louvain.
    Mengembalikan dict {node: community_id}.
    """
    return community_louvain.best_partition(G.to_undirected())


def visualize_communities(G, positions, partition, title="Community Detection", figsize=(12, 8)):
    """
    Visualisasi passing network pada lapangan dengan node diwarnai berdasarkan komunitas.
    """
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=figsize)

    # gambar semua edge
    for u, v in G.edges():
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            w = G[u][v]['weight']
            max_w = max(nx.get_edge_attributes(G, 'weight').values())
            thick = 1 + (w / max_w) * 4  # misal: rentang 1–5
            ax.plot([x1, x2], [y1, y2],
                    color='black',
                    linewidth=thick,
                    alpha=0.5,
                    zorder=1)


    # siapkan colormap
    communities = sorted(set(partition.values()))
    cmap = plt.cm.get_cmap('tab10', len(communities))

    # gambar node per komunitas
    for node, comm_id in partition.items():
        if node in positions:
            x, y = positions[node]
            color = cmap(communities.index(comm_id))
            ax.scatter(
                x, y,
                s=1200,
                facecolor=color,
                edgecolors='black',
                linewidths=2,
                zorder=2
            )
            ax.text(
                x, y,
                node.split()[0],
                ha='center', va='center',
                color='white', fontsize=8, fontweight='bold', zorder=3
            )

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    # pilih pertandingan
    urls = get_url_mapping()
    print("\n📁 Daftar Pertandingan:")
    matches = list(urls.keys())
    for i, m in enumerate(matches, start=1):
        print(f"{i}. {m}")
    try:
        idx = int(input("Pilih nomor pertandingan: ")) - 1
        match_name = matches[idx]
    except Exception:
        print("❌ Pilihan tidak valid.")
        return
    events = load_event_data_from_url(urls[match_name])

    # pilih tim
    teams = sorted({e.get('team', {}).get('name') for e in events if e.get('team')})
    print("\n🔍 Tim yang tersedia:")
    for i, t in enumerate(teams, start=1):
        print(f"{i}. {t}")
    try:
        team_name = teams[int(input("Pilih tim: ")) - 1]
    except Exception:
        print("❌ Pilihan tidak valid.")
        return

    # build basic graph & positions
    G, positions = build_basic_graph_and_positions(events, team_name)
    if G.number_of_nodes() == 0:
        print(f"❌ Graph kosong, tidak ada operan untuk {team_name}.")
        return

    # deteksi komunitas
    partition = detect_communities(G)

    # cetak komunitas
    comm_map = defaultdict(list)
    for node, cid in partition.items():
        comm_map[cid].append(node)
    print("\n🔗 Komunitas yang terdeteksi:")
    for cid, members in comm_map.items():
        print(f"Komunitas {cid+1}: {', '.join(sorted(members))}")

    # visualisasi
    visualize_communities(
        G,
        positions,
        partition,
        title=f"Komunitas Passing: {team_name}"
    )


if __name__ == '__main__':
    main()
