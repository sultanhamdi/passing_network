// static/script.js
const API = window.location.origin;

// Helper to fetch JSON from an API endpoint
async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

// Initialize match selector
async function initMatches() {
  const sel = document.getElementById('matchSelect');
  const matches = await fetchJSON(`${API}/matches`);
  matches.forEach(m => sel.add(new Option(m, m)));
}

// Hide both graph and animation images
function clearDisplays() {
  document.getElementById('graphImage').style.display = 'none';
  document.getElementById('animImage').style.display = 'none';
}

// Load and display the passing network image
function loadGraphImage(mode) {
  clearDisplays();
  const match = encodeURIComponent(document.getElementById('matchSelect').value);
  const team = encodeURIComponent(document.getElementById('teamSelect').value);
  if (!match || !team) return;
  const img = document.getElementById('graphImage');
  img.src = `${API}/graph-image?match=${match}&team=${team}&mode=${mode}`;
  img.onload = () => { img.style.display = 'block'; };
  img.onerror = () => { console.error('Failed to load graph image'); };
}

// Run and display the shortest-path animation
function runShortest() {
  clearDisplays();
  const m   = encodeURIComponent(document.getElementById('matchSelect').value);
  const t   = encodeURIComponent(document.getElementById('teamSelect').value);
  const s   = encodeURIComponent(document.getElementById('srcSelect').value);
  const tgt = encodeURIComponent(document.getElementById('tgtSelect').value);
  if (!m || !t || !s || !tgt) return;
  const anim = document.getElementById('animImage');
  anim.src = `${API}/analysis/shortest-path-gif?match=${m}&team=${t}&src=${s}&tgt=${tgt}`;
  anim.onload = () => { anim.style.display = 'block'; };
  anim.onerror = () => { console.error('Failed to load animation'); };
}

// Populate players dropdown via API
async function loadPlayers() {
  const match = document.getElementById('matchSelect').value;
  const team  = document.getElementById('teamSelect').value;
  if (!match || !team) return;
  const players = await fetchJSON(`${API}/players?match=${encodeURIComponent(match)}&team=${encodeURIComponent(team)}`);
  ['srcSelect','tgtSelect','xtSrcSelect'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = '<option value="">-- Pilih Pemain --</option>';
    players.forEach(p => sel.add(new Option(p, p)));
    sel.disabled = false;
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  await initMatches();

  const matchSel = document.getElementById('matchSelect');
  const teamSel  = document.getElementById('teamSelect');

  // When a match is selected, load teams
  matchSel.onchange = async () => {
    clearDisplays();
    teamSel.disabled = true;
    teamSel.innerHTML = '<option value="">-- Pilih Tim --</option>';
    if (!matchSel.value) return;
    const teams = await fetchJSON(`${API}/teams?match=${encodeURIComponent(matchSel.value)}`);
    teams.forEach(t => teamSel.add(new Option(t, t)));
    teamSel.disabled = false;
  };

  // When a team is selected, load players and enable buttons
  teamSel.onchange = () => {
    clearDisplays();
    loadPlayers();
    document.querySelectorAll(
      '.mode-buttons button, #runShortest, #runXtGoal, #runCommunities'
    ).forEach(b => b.disabled = false);
  };

  // Mode buttons
  document.querySelectorAll('.mode-buttons button').forEach(btn => {
    btn.onclick = () => loadGraphImage(btn.dataset.mode);
  });

  // Shortest-path
  document.getElementById('runShortest').onclick = runShortest;

  // xT Goal Path
  document.getElementById('runXtGoal').onclick = async () => {
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    const s = encodeURIComponent(document.getElementById('xtSrcSelect').value);
    if (!m || !t || !s) return;
    const res = await fetchJSON(`${API}/analysis/xt-goal-path?match=${m}&team=${t}&src=${s}`);
    alert(JSON.stringify(res, null, 2));
  };

  // Communities
  document.getElementById('runCommunities').onclick = async () => {
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    if (!m || !t) return;
    const res = await fetchJSON(`${API}/analysis/communities?match=${m}&team=${t}`);
    alert(JSON.stringify(res, null, 2));
  };
});
