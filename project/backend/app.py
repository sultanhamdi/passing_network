from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import PillowWriter
import tempfile
from pathlib import Path

# Import backend logic
from analysis import (
    get_url_mapping,
    load_event_data_from_url,
    extract_all_starting_players,
    build_passing_graph_with_xt,
    draw_network,
    get_shortest_path,
    animate_shortest_path,
    build_weighted_graph,
    xT_based_goal_pathfinding,
    safe_dangerous_goal_path,
    animate_path
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
    return jsonify(sorted(starting_info.keys()))

@app.route('/players', methods=['GET'])
def players():
    match = request.args.get('match')
    team = request.args.get('team')
    urls = get_url_mapping()
    if not match or match not in urls or not team:
        return jsonify({'error': 'Match or team not found'}), 404
    events = load_event_data_from_url(urls[match])
    starting_info = extract_all_starting_players(events)
    return jsonify(sorted(starting_info.get(team, [])))

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
    G = {'default': basic_g, 'attacking': attack_g, 'defensive': defense_g}.get(mode, basic_g)
    fig = draw_network(
        G, positions, player_team,
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
    match  = request.args.get('match')
    team   = request.args.get('team')
    source = request.args.get('src')
    target = request.args.get('tgt')
    urls   = get_url_mapping()
    if not all([match, team, source, target]) or match not in urls:
        return jsonify({'error': 'Missing parameters'}), 400
    try:
        events = load_event_data_from_url(urls[match])
        starting_info = extract_all_starting_players(events)
        basic_g, _, _, positions, _ = build_passing_graph_with_xt(events, starting_info)
        if source not in basic_g or target not in basic_g:
            return jsonify({'error': 'Source or target not in graph'}), 404
        accuracy = {}
        path, _ = get_shortest_path(basic_g, source, target, positions, accuracy)
        default_pos = (60, 40)
        for node in path:
            positions.setdefault(node, default_pos)
        ani = animate_shortest_path(basic_g, positions, path)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "shortest_path.gif"
            ani.save(str(tmp_path), writer=PillowWriter(fps=1))
            plt.close(ani._fig)
            buf = BytesIO(tmp_path.read_bytes())
            buf.seek(0)
            return send_file(buf, mimetype='image/gif')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# xT-based Goal Pathfinding
@app.route('/analysis/xt-goal-paths', methods=['GET'])
def xt_goal_paths():
    match      = request.args.get('match')
    team       = request.args.get('team')
    origin     = request.args.get('origin')
    max_passes = int(request.args.get('max_passes', 3))
    urls       = get_url_mapping()
    if not all([match, team, origin]) or match not in urls:
        return jsonify({'error': 'Missing parameters'}), 400
    events   = load_event_data_from_url(urls[match])
    starting = extract_all_starting_players(events).get(team, [])
    if origin not in starting:
        return jsonify({'error': 'Origin player not found'}), 404
    G_w, positions = build_weighted_graph(events, starting, team)
    paths = xT_based_goal_pathfinding(G_w, origin, max_passes)
    return jsonify(paths)

@app.route('/analysis/xt-goal-path-gif', methods=['GET'])
def xt_goal_path_gif():
    match      = request.args.get('match')
    team       = request.args.get('team')
    origin     = request.args.get('origin')
    index      = int(request.args.get('index', 0))
    max_passes = int(request.args.get('max_passes', 3))
    urls       = get_url_mapping()
    if not all([match, team, origin]) or match not in urls:
        return jsonify({'error': 'Missing parameters'}), 400
    events   = load_event_data_from_url(urls[match])
    starting = extract_all_starting_players(events).get(team, [])
    if origin not in starting:
        return jsonify({'error': 'Origin player not found'}), 404
    G_w, positions = build_weighted_graph(events, starting, team)
    paths = xT_based_goal_pathfinding(G_w, origin, max_passes)
    if index < 0 or index >= len(paths):
        return jsonify({'error': 'Invalid path index'}), 404
    ani = animate_path(G_w, positions, paths[index]['path'])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"xt_path_{index}.gif"
        ani.save(str(tmp_path), writer=PillowWriter(fps=1))
        plt.close(ani._fig)
        buf = BytesIO(tmp_path.read_bytes())
        buf.seek(0)
        return send_file(buf, mimetype='image/gif')

# Safe/Dangerous Goal Pathfinding
@app.route('/analysis/safe-dangerous-paths', methods=['GET'])
def safe_dangerous_paths():
    match      = request.args.get('match')
    team       = request.args.get('team')
    origin     = request.args.get('origin')
    max_passes = int(request.args.get('max_passes', 3))
    top_k      = int(request.args.get('top_k', 5))
    urls       = get_url_mapping()
    if not all([match, team, origin]) or match not in urls:
        return jsonify({'error': 'Missing parameters'}), 400
    events   = load_event_data_from_url(urls[match])
    starting = extract_all_starting_players(events).get(team, [])
    if origin not in starting:
        return jsonify({'error': 'Origin player not found'}), 404
    G_w, positions = build_weighted_graph(events, starting, team)
    paths = safe_dangerous_goal_path(G_w, origin, positions, {}, max_passes=max_passes, top_k=top_k)
    return jsonify(paths)

@app.route('/analysis/safe-dangerous-path-gif', methods=['GET'])
def safe_dangerous_path_gif():
    match      = request.args.get('match')
    team       = request.args.get('team')
    origin     = request.args.get('origin')
    index      = int(request.args.get('index', 0))
    max_passes = int(request.args.get('max_passes', 3))
    top_k      = int(request.args.get('top_k', 5))
    urls       = get_url_mapping()
    if not all([match, team, origin]) or match not in urls:
        return jsonify({'error': 'Missing parameters'}), 400
    events   = load_event_data_from_url(urls[match])
    starting = extract_all_starting_players(events).get(team, [])
    if origin not in starting:
        return jsonify({'error': 'Origin player not found'}), 404
    G_w, positions = build_weighted_graph(events, starting, team)
    paths = safe_dangerous_goal_path(G_w, origin, positions, {}, max_passes=max_passes, top_k=top_k)
    if index < 0 or index >= len(paths):
        return jsonify({'error': 'Invalid path index'}), 404
    ani = animate_path(G_w, positions, paths[index]['path'])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"sd_path_{index}.gif"
        ani.save(str(tmp_path), writer=PillowWriter(fps=1))
        plt.close(ani._fig)
        buf = BytesIO(tmp_path.read_bytes())
        buf.seek(0)
        return send_file(buf, mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
