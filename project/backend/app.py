from flask import render_template
from flask import Flask, request, jsonify
from flask_cors import CORS
from analysis import *
import networkx as nx
import community as community_louvain
from collections import defaultdict

app = Flask(__name__)
CORS(app)  # izinkan akses dari frontend

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/matches")
def get_matches():
    mapping = get_url_mapping()
    return jsonify([{"name": k, "url": v} for k, v in mapping.items()])

@app.route("/teams")
def get_teams():
    url = request.args.get("url")
    events = load_event_data_from_url(url)
    teams = list()
    # Ambil nama kedua tim dari lineup AJA
    for e in events:
        if e['type']['name'] == 'Starting XI':
            teams.append(e['team']['name'])
            if len(teams) == 2:
                break
    return jsonify(sorted(teams))

@app.route("/players")
def get_players():
    url = request.args.get("url")
    team = request.args.get("team")
    events = load_event_data_from_url(url)
    players, _ = extract_starting_players(events, team)
    return jsonify(sorted(players))

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data["url"]
    team = data["team"]
    source = data["source"]
    target = data["target"]
    mode = data.get("mode", 1)

    events = load_event_data_from_url(url)
    players, _ = extract_starting_players(events, team)

    if mode == 1:
        G, pos, acc = build_passing_graph(events, players, team_name=team, threshold=1)
        G = compute_realistic_cost(G, pos, acc)
    elif mode == 2:
        G = build_attacking_weighted_graph(events, players, team_name=team, threshold=1)
    elif mode == 3:
        G = build_defensive_weighted_graph(events, players, team_name=team, threshold=1)
    else:
        return jsonify({"error": "Mode tidak valid"}), 400

    path, cost = get_shortest_path(G, source, target)
    central = analyze_centralities(G)

    # Ambil node tertinggi dari beberapa centrality
    def get_top(d):
        return max(d, key=d.get) if d else ""

    partition = community_louvain.best_partition(G.to_undirected())
    comm_map = defaultdict(list)
    for player, group in partition.items():
        comm_map[group].append(player)
    communities = [sorted(members) for members in comm_map.values()]

    return jsonify({
        "path": path,
        "cost": cost,
        "central": {
            "degree": get_top(central["degree"]),
            "betweenness": get_top(central["betweenness"]),
            "pagerank": get_top(central["pagerank"])
        },
        "communities": communities
    })

if __name__ == "__main__":
    app.run(debug=True)