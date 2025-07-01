from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx

# Import backend logic
from analysis_merged_xT import (
    get_url_mapping,
    load_event_data_from_url,
    extract_all_starting_players,
    build_passing_graph_with_xt,
    draw_network,
    get_shortest_path,
    draw_shortest_path
)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/matches', methods=['GET'])
def matches():
    urls = get_url_mapping()
    return jsonify(list(urls.keys()))

@app.route('/teams', methods=['GET'])
def teams():
    match = request.args.get('match')
    urls = get_url_mapping()
    if not match or match not in urls:
        return jsonify({'error': 'Match not found'}), 404
    events = load_event_data_from_url(urls[match])
    starting_info = extract_all_starting_players(events)
    teams = sorted(starting_info.keys())
    return jsonify(teams)

@app.route('/players', methods=['GET'])
def players():
    match = request.args.get('match')
    team = request.args.get('team')
    urls = get_url_mapping()
    if not match or match not in urls or not team:
        return jsonify({'error': 'Match or team not found'}), 404
    events = load_event_data_from_url(urls[match])
    starting_info = extract_all_starting_players(events)
    players_list = list(starting_info.get(team, []))
    return jsonify(sorted(players_list))

@app.route('/graph-image', methods=['GET'])
def graph_image():
    match = request.args.get('match')
    team = request.args.get('team')
    mode = request.args.get('mode', 'default')
    urls = get_url_mapping()
    if not match or match not in urls or not team:
        return 'Match or team not specified', 404
    events = load_event_data_from_url(urls[match])
    starting_info = extract_all_starting_players(events)
    basic_g, attack_g, defense_g, positions, player_team = build_passing_graph_with_xt(events, starting_info)
    if mode == 'attacking':
        G = attack_g
    elif mode == 'defensive':
        G = defense_g
    else:
        G = basic_g
    fig = draw_network(
        G,
        positions,
        player_team,
        selected_team=team,
        title=f"{team} - {mode.title()} Passing Network"
    )
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/analysis/shortest-path-gif', methods=['GET'])
def shortest_path_gif():
    match = request.args.get('match')
    team = request.args.get('team')
    source = request.args.get('src')
    target = request.args.get('tgt')
    urls = get_url_mapping()
    if not match or match not in urls or not team or not source or not target:
        return 'Missing parameters', 400
    try:
        events = load_event_data_from_url(urls[match])
        starting_info = extract_all_starting_players(events)
        basic_g, _, _, positions, player_team = build_passing_graph_with_xt(events, starting_info)
        # Filter to selected team edges
        G_team = nx.DiGraph()
        for u, v, data in basic_g.edges(data=True):
            if data.get('team') == team:
                G_team.add_edge(u, v, **data)
        # Compute shortest path
        accuracy = {p: 1.0 for p in starting_info.get(team, [])}
        path, cost = get_shortest_path(G_team, source, target, positions, accuracy)
        if not path:
            return 'No path found', 404
        fig = draw_shortest_path(G_team, positions, path, title=f"{team} Shortest Path")
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        response = send_file(buf, mimetype='image/png')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        app.logger.exception("Error generating shortest-path")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
