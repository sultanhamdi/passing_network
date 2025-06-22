from analysis_merged_xT import (
    get_url_mapping,
    load_event_data_from_url,
    extract_starting_players,
    build_passing_graph_with_xt,
    get_shortest_path,
    analyze_centralities,
    draw_network
)

import networkx as nx
from collections import defaultdict
import community as community_louvain  # Louvain clustering

def main():
    urls = get_url_mapping()
    print("\n📁 Daftar Pertandingan yang Tersedia:")
    match_keys = list(urls.keys())
    for i, name in enumerate(match_keys):
        print(f"{i+1}. {name}")

    try:
        match_choice = int(input("\nPilih nomor pertandingan: ")) - 1
        match_url = urls[match_keys[match_choice]]
        match_name = match_keys[match_choice]
    except (ValueError, IndexError):
        print("\n❌ Input tidak valid.")
        return

    events = load_event_data_from_url(match_url)
    teams = sorted({e['team']['name'] for e in events if 'team' in e})
    print("\n🔍 Tim yang tersedia:")
    for i, t in enumerate(teams):
        print(f"{i+1}. {t}")

    try:
        team_idx = int(input("\nPilih tim yang akan dianalisis: ")) - 1
        team_name = teams[team_idx]
    except (ValueError, IndexError):
        print("❌ Input tidak valid.")
        return

    players, _ = extract_starting_players(events, team_name)
    player_list = sorted(players)
    print("\n👥 Pemain (Starting XI):")
    for i, p in enumerate(player_list):
        print(f"{i+1}. {p}")

    print("\n📊 Jenis passing network (berbasis xT):")
    print("1. Basic Frequency Graph")
    print("2. Attacking xT Graph")
    print("3. Defensive xT Graph")

    try:
        mode = int(input("Pilih mode network (1/2/3): "))
    except ValueError:
        mode = 1

    basic_g, attack_g, defense_g, pos = build_passing_graph_with_xt(events, players, team_name)

    if mode == 1:
        G = basic_g
        title = f"{team_name} Passing Frequency Network"
    elif mode == 2:
        G = attack_g
        title = f"{team_name} Attacking xT Network"
    elif mode == 3:
        G = defense_g
        title = f"{team_name} Defensive xT Network"
    else:
        print("❌ Mode tidak dikenali, menggunakan Basic Graph.")
        G = basic_g
        title = f"{team_name} Passing Frequency Network"

    draw_network(G, pos, title=title)

    print("\n🔁 Pemain yang tersedia:")
    for i, p in enumerate(player_list):
        print(f"{i+1}. {p}")

    try:
        src = int(input("\nPilih pemain awal (nomor): ")) - 1
        tgt = int(input("Pilih pemain tujuan (nomor): ")) - 1
        source = player_list[src]
        target = player_list[tgt]
    except (ValueError, IndexError):
        print("❌ Input pemain tidak valid.")
        return

    # Dummy akurasi (belum dihitung di xT)
    accuracy = {p: 0.9 for p in G.nodes()}
    path, cost = get_shortest_path(G, source, target, pos, accuracy)

    print(f"\n🔍 Shortest path dari {source} ke {target}:")
    if path:
        for i, p in enumerate(path):
            print(f"{i+1}. {p}")
        print(f"Total cost: {cost:.4f}")
    else:
        print("Tidak ditemukan jalur.")

    print("\n📊 Centrality Analysis:")
    central = analyze_centralities(G)
    for k in ['degree', 'betweenness', 'pagerank']:
        top = max(central[k], key=central[k].get)
        print(f"- {k.capitalize()} tertinggi: {top} ({central[k][top]:.4f})")

    print("\n🔗 Deteksi Komunitas (Clustering via Louvain):")
    try:
        partition = community_louvain.best_partition(G.to_undirected())
        comm_map = defaultdict(list)
        for player, group in partition.items():
            comm_map[group].append(player)
        for i, comm in enumerate(comm_map.values(), 1):
            print(f"  Komunitas {i}: {', '.join(sorted(comm))}")
    except Exception as e:
        print(f"  ❌ Gagal mendeteksi komunitas: {e}")

if __name__ == "__main__":
    main()
