
from analysis import (
    get_url_mapping,
    load_event_data_from_url,
    extract_starting_players,
    build_passing_graph,
    compute_realistic_cost,
    get_shortest_path,
    analyze_centralities,
    build_attacking_weighted_graph,
    build_defensive_weighted_graph
)

import networkx as nx

def main():
    urls = get_url_mapping()
    print("\n📁 Daftar Pertandingan yang Tersedia:")
    match_keys = list(urls.keys())
    for i, name in enumerate(match_keys):
        print(f"{i+1}. {name}")

    try:
        match_choice = int(input("\nPilih nomor pertandingan: ")) - 1
        match_url = urls[match_keys[match_choice]]
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

    try:
        src = int(input("\nPilih pemain awal (nomor): ")) - 1
        tgt = int(input("Pilih pemain tujuan (nomor): ")) - 1
        source = player_list[src]
        target = player_list[tgt]
    except (ValueError, IndexError):
        print("❌ Input pemain tidak valid.")
        return

    print("\n📊 Jenis passing network:")
    print("1. Default (frekuensi)")
    print("2. Attacking-weighted")
    print("3. Defensive-weighted")

    try:
        mode = int(input("Pilih mode network: "))
    except ValueError:
        mode = 1

    if mode == 1:
        G, pos, acc = build_passing_graph(events, players, threshold=1)
        G = compute_realistic_cost(G, pos, acc)
    elif mode == 2:
        G = build_attacking_weighted_graph(events, players, threshold=1)
    elif mode == 3:
        G = build_defensive_weighted_graph(events, players, threshold=1)
    else:
        print("❌ Mode tidak dikenali, menggunakan default.")
        G, pos, acc = build_passing_graph(events, players, threshold=1)
        G = compute_realistic_cost(G, pos, acc)

    path, cost = get_shortest_path(G, source, target)

    print(f"\n🔍 Shortest path dari {source} ke {target}:")
    if path:
        for i, p in enumerate(path):
            print(f"{i+1}. {p}")
        print(f"Total heuristic cost: {cost:.4f}")
    else:
        print("Tidak ditemukan jalur.")

    print("\n📊 Centrality Analysis:")
    central = analyze_centralities(G)
    for k in ['degree', 'betweenness', 'pagerank']:
        top = max(central[k], key=central[k].get)
        print(f"- {k.capitalize()} tertinggi: {top} ({central[k][top]:.4f})")

    print("\n🔗 Deteksi Komunitas (Clustering via Girvan-Newman):")
    try:
        from networkx.algorithms.community import girvan_newman
        communities = next(girvan_newman(G.to_undirected()))
        for i, comm in enumerate(communities):
            print(f"  Komunitas {i+1}: {', '.join(sorted(comm))}")
    except Exception as e:
        print(f"  ❌ Gagal mendeteksi komunitas: {e}")

if __name__ == "__main__":
    main()
