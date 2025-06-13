# backend/simulator.py

import os
from analysis import (
    load_event_data,
    extract_starting_players,
    build_passing_graph,
    compute_realistic_cost,
    get_shortest_path,
    analyze_centralities
)
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

def detect_communities(G):
    try:
        from networkx.algorithms.community import girvan_newman
        communities = next(girvan_newman(G.to_undirected()))
        return [list(c) for c in communities]
    except ImportError:
        return []

def draw_pitch_with_positions(pos, G, title="Visualisasi Posisi Pemain"):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#a8bc95', line_color='white', stripe=True, stripe_color='#c2d59d')
    fig, ax = pitch.draw(figsize=(14, 8))
    fig.set_facecolor('#a8bc95')

    for player, (x, y) in pos.items():
        pitch.scatter(x, y, ax=ax, s=500, color='orange', edgecolors='black', zorder=3)
        ax.text(x, y, player.split()[0], ha='center', va='center', fontsize=10, color='white', zorder=4)

    for u, v in G.edges():
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            pitch.arrows(x1, y1, x2, y2, ax=ax, color='black', width=2, headwidth=5, zorder=2)

    ax.set_title(title, fontsize=16, color='white')
    plt.tight_layout()
    plt.show()

def main():
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "../data")
    match_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    print("\n📁 Daftar Pertandingan yang Tersedia:")
    for i, file in enumerate(match_files):
        print(f"{i+1}. {file}")

    try:
        match_choice = int(input("\nPilih nomor file pertandingan: ")) - 1
        match_file = os.path.join(data_dir, match_files[match_choice])
    except (ValueError, IndexError):
        print("\n❌ Input tidak valid.")
        return

    print("\n📦 Loading match data...")
    events = load_event_data(match_file)

    teams = sorted({e['team']['name'] for e in events if 'team' in e})
    teams = list(dict.fromkeys(teams))
    print("\n🔍 Tim yang tersedia dalam match:")
    for i, t in enumerate(teams):
        print(f"{i+1}. {t}")

    try:
        team_idx = int(input("\nPilih tim yang akan dianalisis: ")) - 1
        team_name = teams[team_idx]
    except (ValueError, IndexError):
        print("\n❌ Input tim tidak valid.")
        return

    players, _ = extract_starting_players(events, team_name)

    print("\n👥 Starting XI:")
    player_list = sorted(players)
    for i, p in enumerate(player_list):
        print(f"{i+1}. {p}")

    try:
        idx_start = int(input("\nPilih nomor pemain awal: ")) - 1
        idx_end = int(input("Pilih nomor pemain tujuan: ")) - 1
        source = player_list[idx_start]
        target = player_list[idx_end]
    except (ValueError, IndexError):
        print("\n❌ Input tidak valid.")
        return

    print("\n📊 Membuat graf passing...")
    G, pos, acc = build_passing_graph(events, players, threshold=1)
    G = compute_realistic_cost(G, pos, acc)

    print(f"🔍 Mencari jalur optimal dari {source} ke {target}...")
    path, cost = get_shortest_path(G, source, target)

    if path:
        print("\n✅ Jalur ditemukan:")
        for i, p in enumerate(path):
            print(f"{i+1}. {p}")
        print(f"Total heuristic cost: {cost:.4f}")
    else:
        print("\n⚠️ Tidak ada jalur tersedia.")

    print(f"\n🎯 Ranking Target Umpan dari {source}:")
    if source in G:
        neighbors = G[source]
        ranked = sorted(neighbors.items(), key=lambda item: item[1]['weight'], reverse=True)
        for i, (target_player, data) in enumerate(ranked):
            print(f"  {i+1}. {target_player} (jumlah passing: {data['weight']})")
    else:
        print("  Pemain tidak memiliki data passing keluar.")

    print("\n📈 Analisis Centrality:")
    centralities = analyze_centralities(G)
    top_degree = max(centralities['degree'], key=centralities['degree'].get)
    top_betweenness = max(centralities['betweenness'], key=centralities['betweenness'].get)
    top_pagerank = max(centralities['pagerank'], key=centralities['pagerank'].get)

    print(f"- Degree Centrality Tertinggi: {top_degree}")
    print(f"- Betweenness Centrality Tertinggi: {top_betweenness}")
    print(f"- PageRank Tertinggi: {top_pagerank}")

    clustering = nx.clustering(G.to_undirected())
    top_clustering = max(clustering, key=clustering.get)
    avg_clustering = sum(clustering.values()) / len(clustering) if clustering else 0
    print(f"- Clustering Coefficient Tertinggi: {top_clustering} ({clustering[top_clustering]:.2f})")
    print(f"- Rata-rata Clustering Coefficient: {avg_clustering:.2f}")

    print("\n🔗 Deteksi Komunitas (Clustering):")
    communities = detect_communities(G)
    for i, c in enumerate(communities):
        print(f"  Komunitas {i+1}: {', '.join(sorted(c))}")

    print("\n🧪 Simulasi: Dampak Penghapusan Pemain")
    try:
        remove_idx = int(input("Pilih nomor pemain yang ingin dihapus dari graf: ")) - 1
        removed_player = player_list[remove_idx]
        G_removed = G.copy()
        G_removed.remove_node(removed_player)
        new_path, new_cost = get_shortest_path(G_removed, source, target)

        print(f"\n📉 Setelah menghapus {removed_player}:")
        if new_path:
            print(f"  Jalur baru ditemukan dengan total cost {new_cost:.4f}:")
            for i, p in enumerate(new_path):
                print(f"  {i+1}. {p}")
        else:
            print("  Tidak ada jalur baru yang tersedia.")
    except (ValueError, IndexError, nx.NetworkXError):
        print("  ❌ Input atau penghapusan tidak valid.")

    print("\n🔁 Simulasi: Tukar Posisi Dua Pemain")
    try:
        swap_a = int(input("Masukkan nomor pemain pertama: ")) - 1
        swap_b = int(input("Masukkan nomor pemain kedua: ")) - 1
        a, b = player_list[swap_a], player_list[swap_b]
        pos_swapped = pos.copy()
        pos_swapped[a], pos_swapped[b] = pos_swapped[b], pos_swapped[a]
        print(f"  Posisi {a} dan {b} ditukar. Menampilkan visualisasi...")
        draw_pitch_with_positions(pos_swapped, G, title=f"Tukar Posisi: {a} <-> {b}")
    except (ValueError, IndexError):
        print("  ❌ Input tidak valid.")

if __name__ == "__main__":
    main()
