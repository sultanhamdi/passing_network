document.addEventListener("DOMContentLoaded", () => {
  const matchSelect = document.getElementById("match-select");
  const teamSelect = document.getElementById("team-select");
  const sourcePlayer = document.getElementById("source-player");
  const targetPlayer = document.getElementById("target-player");
  const modeSelect = document.getElementById("mode-select");
  const analyzeBtn = document.getElementById("analyze-btn");
  const results = document.getElementById("results");

  const teamSection = document.getElementById("team-section");
  const playerSection = document.getElementById("player-section");
  const modeSection = document.getElementById("mode-section");
  const resultSection = document.getElementById("result-section");

  const apiBase = "http://localhost:5000"; // Ganti jika backend beda port atau deploy

  async function fetchMatches() {
    const res = await fetch(`${apiBase}/matches`);
    const matches = await res.json();
    matchSelect.innerHTML = matches.map(
      (m, i) => `<option value="${m.url}">${m.name}</option>`
    ).join("");
    teamSection.classList.remove("hidden");
  }

  matchSelect.addEventListener("change", async () => {
    const url = matchSelect.value;
    const res = await fetch(`${apiBase}/teams?url=${encodeURIComponent(url)}`);
    const teams = await res.json();
    teamSelect.innerHTML = teams.map(
      t => `<option value="${t}">${t}</option>`
    ).join("");
    playerSection.classList.add("hidden");
    modeSection.classList.add("hidden");
    resultSection.classList.add("hidden");
  });

  teamSelect.addEventListener("change", async () => {
    const url = matchSelect.value;
    const team = teamSelect.value;
    const res = await fetch(`${apiBase}/players?url=${encodeURIComponent(url)}&team=${encodeURIComponent(team)}`);
    const players = await res.json();
    sourcePlayer.innerHTML = targetPlayer.innerHTML = players.map(
      p => `<option value="${p}">${p}</option>`
    ).join("");
    playerSection.classList.remove("hidden");
    modeSection.classList.remove("hidden");
  });

  analyzeBtn.addEventListener("click", async () => {
    const payload = {
      url: matchSelect.value,
      team: teamSelect.value,
      source: sourcePlayer.value,
      target: targetPlayer.value,
      mode: parseInt(modeSelect.value)
    };

    const res = await fetch(`${apiBase}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    results.textContent = `
🔍 Shortest Path (${payload.source} → ${payload.target}):
${data.path.join(" -> ")}
Cost: ${data.cost.toFixed(4)}

📊 Centrality:
Degree: ${data.central.degree}
Betweenness: ${data.central.betweenness}
PageRank: ${data.central.pagerank}

🔗 Komunitas:
${data.communities.map((c, i) => `Komunitas ${i+1}: ${c.join(", ")}`).join("\n")}
    `;
    resultSection.classList.remove("hidden");
  });

  fetchMatches();
});
