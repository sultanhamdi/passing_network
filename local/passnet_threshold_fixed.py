
# 1. Import library
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os

# 7. Layout posisi pemain realistis (skala StatsBomb 105 x 68 meter)
realistic_role_coords = {
    "Goalkeeper": (5, 34),
    "Right Back": (25, 15),
    "Right Center Back": (20, 25),
    "Center Back": (20, 34),
    "Left Center Back": (20, 43),
    "Left Back": (25, 53),
    "Right Defensive Midfield": (40, 25),
    "Left Defensive Midfield": (40, 43),
    "Defensive Midfield": (40, 34),
    "Right Midfield": (55, 20),
    "Left Midfield": (55, 48),
    "Central Midfield": (55, 34),
    "Right Wing": (80, 10),
    "Left Wing": (80, 58),
    "Attacking Midfield": (70, 34),
    "Second Striker": (80, 34),
    "Center Forward": (90, 34),
    "Striker": (90, 34)
}

def main(min_pass_threshold=0):
    # 2. Load file event JSON
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "3773587.json")

    with open(file_path, "r", encoding="utf-8") as f:
        events_data = json.load(f)

    # 3. Ambil pemain Starting XI Barcelona
    starting_players = set()
    starting_xi_positions = {}
    for event in events_data:
        if event.get("type", {}).get("name") == "Starting XI" and event.get("team", {}).get("name") == "Barcelona":
            for player in event["tactics"]["lineup"]:
                name = player["player"]["name"]
                starting_players.add(name)
                starting_xi_positions[name] = player["position"]["name"]

    # 4. Filter operan antar pemain starting saja
    passes = []
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

    # 5. Hitung jumlah passing antar pemain
    pass_counts = defaultdict(int)
    for passer, receiver in passes:
        pass_counts[(passer, receiver)] += 1

    # 6. Buat graf passing network dengan threshold
    G = nx.DiGraph()
    for (passer, receiver), weight in pass_counts.items():
        if weight >= min_pass_threshold:
            G.add_edge(passer, receiver, weight=weight)

    G.remove_nodes_from(list(nx.isolates(G)))

    # 8. Buat posisi setiap pemain sesuai role
    pos_real = {}
    for player in G.nodes():
        role = starting_xi_positions.get(player, "Central Midfield")
        if player == "Norberto Murara Neto":
            pos_real[player] = realistic_role_coords["Goalkeeper"]
        else:
            pos_real[player] = realistic_role_coords.get(role, realistic_role_coords["Central Midfield"])

    # 9. Visualisasi dengan mplsoccer
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#a8bc95',
        line_color='#ffffff',
        stripe=True,
        stripe_color='#c2d59d'
    )

    fig, ax = pitch.draw(figsize=(14, 10))
    fig.set_facecolor('#a8bc95')

    for u, v, data in G.edges(data=True):
        x1, y1 = pos_real[u]
        x2, y2 = pos_real[v]
        pitch.lines(x1, y1, x2, y2, lw=data['weight'] * 0.5, color='black', ax=ax, zorder=1, comet=True)

    for player in G.nodes():
        x, y = pos_real[player]
        size = 300 + 100 * G.degree(player)
        pitch.scatter(x, y, s=size, color='orange', edgecolors='black', ax=ax, zorder=2)

    for player in G.nodes():
        x, y = pos_real[player]
        ax.text(x, y, player.split()[0], color='white', fontsize=10, ha='center', va='center', zorder=3)

    plt.title(f"Passing Network - Barcelona vs Real Madrid (Min Passing: {min_pass_threshold})",
              fontsize=16, color='white')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
