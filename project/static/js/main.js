document.addEventListener("DOMContentLoaded", () => {
  const matchSelect = document.getElementById("matchSelect");
  const teamSelect = document.getElementById("teamSelect");
  const sourcePlayer = document.getElementById("sourcePlayer");
  const targetPlayer = document.getElementById("targetPlayer");
  const simulateBtn = document.getElementById("simulatePath");
  const modeSelect = document.getElementById("graphMode");
  const graphDiv = document.getElementById("graph");

  fetch("/matches")
    .then(res => res.json())
    .then(data => {
      matchSelect.innerHTML = data.map(d => `<option value="${d.url}">${d.name}</option>`).join("");
      matchSelect.dispatchEvent(new Event("change"));
    });

  matchSelect.addEventListener("change", () => {
    fetch(`/teams?url=${matchSelect.value}`)
      .then(res => res.json())
      .then(teams => {
        teamSelect.innerHTML = teams.map(t => `<option value="${t}">${t}</option>`).join("");
        teamSelect.dispatchEvent(new Event("change"));
      });
  });

  teamSelect.addEventListener("change", () => {
    fetch(`/players?url=${matchSelect.value}&team=${teamSelect.value}`)
      .then(res => res.json())
      .then(players => {
        sourcePlayer.innerHTML = players.map(p => `<option>${p}</option>`).join("");
        targetPlayer.innerHTML = sourcePlayer.innerHTML;
      });
  });

  simulateBtn.addEventListener("click", () => {
    const body = {
      url: matchSelect.value,
      team: teamSelect.value,
      source: sourcePlayer.value,
      target: targetPlayer.value,
      mode: parseInt(modeSelect.value)
    };

    fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
    .then(res => res.json())
    .then(data => {
      if (!data.path || !data.positions) {
        alert("Data tidak valid.");
        return;
      }

      drawGraph(data.path, data.positions, data.communities);
    });
  });

  function drawGraph(path, positions, communities) {
    graphDiv.innerHTML = "";

    const scaleX = 10;
    const scaleY = 10;
    const nodeCoords = {};

    const colors = [
      "#ff4d4d", "#4dd2ff", "#ffa64d", "#4dff4d",
      "#ffff4d", "#b84dff", "#ff66cc", "#00e6ac"
    ];
    const playerColor = {};
    communities.forEach((group, i) => {
      group.forEach(player => {
        playerColor[player] = colors[i % colors.length];
      });
    });

    // Buat node
    path.forEach((player, i) => {
      if (!positions[player]) return;
      const [x, y] = positions[player];

      const node = document.createElement("div");
      node.className = "player-node";
      node.textContent = player;
      node.style.left = `${x * scaleX}px`;
      node.style.top = `${y * scaleY}px`;
      node.style.backgroundColor = playerColor[player] || "#ffcc00";
      node.style.animationDelay = `${i * 0.2}s`;
      graphDiv.appendChild(node);

      nodeCoords[player] = { x: x * scaleX, y: y * scaleY };
    });

    // Garis antar node
    for (let i = 0; i < path.length - 1; i++) {
      const u = nodeCoords[path[i]];
      const v = nodeCoords[path[i + 1]];
      if (!u || !v) continue;

      const dx = v.x - u.x;
      const dy = v.y - u.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx) * 180 / Math.PI;

      const edge = document.createElement("div");
      edge.className = "edge-line";
      edge.style.width = `${distance}px`;
      edge.style.left = `${u.x}px`;
      edge.style.top = `${u.y}px`;
      edge.style.transform = `rotate(${angle}deg)`;
      edge.style.animationDelay = `${i * 0.2}s`;
      graphDiv.appendChild(edge);
    }
  }
});
