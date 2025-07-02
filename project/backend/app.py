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
    animate_shortest_path
)

# Import PillowWriter for GIF output
from matplotlib.animation import PillowWriter

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
    match  = request.args.get('match')
    team   = request.args.get('team')
    source = request.args.get('src')
    target = request.args.get('tgt')
    urls   = get_url_mapping()

    # 1) Validasi parameter
    if not all([match, team, source, target]) or match not in urls:
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        # 2) Muat data dan bangun graph
        events        = load_event_data_from_url(urls[match])
        starting_info = extract_all_starting_players(events)
        basic_g, _, _, positions, _ = build_passing_graph_with_xt(events, starting_info)
        G = basic_g

        # 3) Guard: pastikan node ada di graph
        if source not in G or target not in G:
            return jsonify({'error': f'{source} or {target} not in graph'}), 404

        # 4) Hitung shortest path
        accuracy = {}
        path, total_cost = get_shortest_path(G, source, target, positions, accuracy)
        if not path:
            return jsonify({'error': 'No path found'}), 404

        # 5) Fallback posisi untuk setiap node
        default_pos = (60, 40)
        for node in path:
            positions.setdefault(node, default_pos)

        # 6) Buat animasi
        ani = animate_shortest_path(G, positions, path)

        # 7) Simpan ke file sementara di temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "shortest_path.gif"
            writer   = PillowWriter(fps=1)
            ani.save(str(tmp_path), writer=writer)
            plt.close(ani._fig)

            # 8) Baca file GIF ke memori
            gif_bytes = tmp_path.read_bytes()
            buf = BytesIO(gif_bytes)
            buf.seek(0)

            # 9) Kirim GIF
            return send_file(buf, mimetype='image/gif')

    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        return jsonify({'error': str(e)}), 404

    except KeyError as e:
        return jsonify({'error': f'Posisi untuk pemain {e.args[0]} tidak tersedia'}), 500

    except Exception as e:
        print(type(e), e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
