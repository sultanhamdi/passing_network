from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain

# Import backend logic
from analysis_merged_xT import (
    get_url_mapping,
    load_event_data_from_url,
    extract_starting_players,
    build_passing_graph_with_xt,
    draw_network
)
from simulator_xT import animate_shortest_path

# Initialize Flask app (uses `templates/` & `static/` dirs)
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
    teams = sorted({e.get('team', {}).get('name') for e in events if e.get('team')})
    return jsonify(teams)

@app.route('/graph-image', methods=['GET'])
def graph_image():
    match = request.args.get('match')
    team = request.args.get('team')
    mode = request.args.get('mode', 'default')
    urls = get_url_mapping()
    if not match or match not in urls:
        return 'Match not found', 404
    events = load_event_data_from_url(urls[match])
    players, positions = extract_starting_players(events, team)
    basic_g, attack_g, defense_g, pos = build_passing_graph_with_xt(events, players, team)
    if mode == 'attacking':
        G = attack_g
    elif mode == 'defensive':
        G = defense_g
    else:
        G = basic_g

    # Plot using draw_network and capture figure
    fig = draw_network(G, pos, title=f"{team} - {mode.title()} Graph")
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
    if not match or match not in urls:
        return 'Match not found', 404
    events = load_event_data_from_url(urls[match])
    players, positions = extract_starting_players(events, team)
    basic_g, _, _, pos = build_passing_graph_with_xt(events, players, team)
    accuracy = {p: 1.0 for p in players}
    ani = animate_shortest_path(basic_g, pos, source, target, accuracy)
    buf = BytesIO()
    ani.save(buf, writer='imagemagick', format='gif')
    buf.seek(0)
    return send_file(buf, mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
