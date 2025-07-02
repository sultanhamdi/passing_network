// static/script.js
const API = window.location.origin;

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

async function initMatches() {
  const sel = document.getElementById('matchSelect');
  const matches = await fetchJSON(`${API}/matches`);
  matches.forEach(m => sel.add(new Option(m, m)));
}

function clearDisplays() {
  document.getElementById('graphImage').style.display = 'none';
  document.getElementById('animImage').style.display = 'none';
  document.getElementById('xtAnimImage').style.display = 'none';
  document.getElementById('sdAnimImage').style.display = 'none';
}

function loadGraphImage(mode) {
  clearDisplays();
  const match = encodeURIComponent(document.getElementById('matchSelect').value);
  const team  = encodeURIComponent(document.getElementById('teamSelect').value);
  if (!match || !team) return;
  const img = document.getElementById('graphImage');
  img.src = `${API}/graph-image?match=${match}&team=${team}&mode=${mode}`;
  img.onload  = () => img.style.display = 'block';
  img.onerror = () => console.error('Failed to load graph image');
}

async function runShortest() {
  clearDisplays();
  const m   = encodeURIComponent(matchSelect.value);
  const t   = encodeURIComponent(teamSelect.value);
  const s   = encodeURIComponent(srcSelect.value);
  const tgt = encodeURIComponent(tgtSelect.value);
  const url = `${API}/analysis/shortest-path-gif?match=${m}&team=${t}&src=${s}&tgt=${tgt}`;

  try {
    const res = await fetch(url);
    if (!res.ok) {
      const { error } = await res.json();
      throw new Error(error || res.statusText);
    }
    const blob = await res.blob();
    animImage.src = URL.createObjectURL(blob);
    animImage.style.display = 'block';
  } catch (err) {
    alert('Gagal memuat animasi: ' + err.message);
  }
}

async function loadPlayers() {
  const match = document.getElementById('matchSelect').value;
  const team  = document.getElementById('teamSelect').value;
  if (!match || !team) return;
  const players = await fetchJSON(
    `${API}/players?match=${encodeURIComponent(match)}&team=${encodeURIComponent(team)}`
  );
  ['srcSelect', 'tgtSelect', 'xtSrcSelect', 'sdSrcSelect'].forEach(id => {
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
  const srcSelect = document.getElementById('srcSelect');
  const tgtSelect = document.getElementById('tgtSelect');

  matchSel.onchange = async () => {
    clearDisplays();
    teamSel.disabled = true;
    teamSel.innerHTML = '<option value="">-- Pilih Tim --</option>';
    if (!matchSel.value) return;
    const teams = await fetchJSON(
      `${API}/teams?match=${encodeURIComponent(matchSel.value)}`
    );
    teams.forEach(t => teamSel.add(new Option(t, t)));
    teamSel.disabled = false;
  };

  teamSel.onchange = () => {
    clearDisplays();
    loadPlayers();
    document.querySelectorAll(
      '.mode-buttons button, #runShortest, #runXtGoal, #runSafeDanger, #runCommunities'
    ).forEach(b => b.disabled = false);
  };

  document.querySelectorAll('.mode-buttons button').forEach(btn => {
    btn.onclick = () => loadGraphImage(btn.dataset.mode);
  });

  document.getElementById('runShortest').onclick = runShortest;

  document.getElementById('runXtGoal').onclick = async () => {
    clearDisplays();
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    const o = encodeURIComponent(document.getElementById('xtSrcSelect').value);
    if (!m || !t || !o) return;
    try {
      const paths = await fetchJSON(
        `${API}/analysis/xt-goal-paths?match=${m}&team=${t}&origin=${o}`
      );
      console.log('xT paths:', paths);
      const gifRes = await fetch(
        `${API}/analysis/xt-goal-path-gif?match=${m}&team=${t}&origin=${o}&index=0`
      );
      if (!gifRes.ok) throw new Error('Gagal load GIF xT-path');
      const blob = await gifRes.blob();
      const img = document.getElementById('xtAnimImage');
      img.src = URL.createObjectURL(blob);
      img.style.display = 'block';
    } catch (err) {
      alert('Error xT-path: ' + err.message);
    }
  };

  // Safe/Dangerous Goal Path
  document.getElementById('runSafeDanger').onclick = async () => {
    clearDisplays();
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    const o = encodeURIComponent(
      document.getElementById('sdSrcSelect').value
    );
    if (!m || !t || !o) return;
    try {
      const res = await fetchJSON(
        `${API}/analysis/safe-dangerous-paths?match=${m}&team=${t}&origin=${o}`
      );
      console.log('Safe/Dangerous paths:', res);
      const gifRes = await fetch(
        `${API}/analysis/safe-dangerous-path-gif?match=${m}&team=${t}&origin=${o}&index=0`
      );
      if (!gifRes.ok) throw new Error('Gagal load GIF SD-path');
      const blob = await gifRes.blob();
      const img = document.getElementById('sdAnimImage');
      img.src = URL.createObjectURL(blob);
      img.style.display = 'block';
    } catch (err) {
      alert('Error Safe/Dangerous: ' + err.message);
    }
  };

  document.getElementById('runCommunities').onclick = async () => {
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    if (!m || !t) return;
    const res = await fetchJSON(
      `${API}/analysis/communities?match=${m}&team=${t}`
    );
    alert(JSON.stringify(res, null, 2));
  };
});
