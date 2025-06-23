const API = window.location.origin;

// Bantu fetch JSON dari API
async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json();
}

// Inisialisasi daftar match
async function initMatches() {
  const sel = document.getElementById('matchSelect');
  const matches = await fetchJSON('/matches');
  matches.forEach(m => sel.add(new Option(m, m)));
}

// Sembunyikan kedua <img>
function clearDisplays() {
  document.getElementById('graphImage').style.display = 'none';
  document.getElementById('animImage').style.display = 'none';
}

// Load & tampilkan graph PNG
async function loadGraphImage(mode) {
  clearDisplays();
  const match = encodeURIComponent(document.getElementById('matchSelect').value);
  const team  = encodeURIComponent(document.getElementById('teamSelect').value);
  const img   = document.getElementById('graphImage');
  img.src     = `${API}/graph-image?match=${match}&team=${team}&mode=${mode}`;
  img.onload  = () => img.style.display = 'block';
  img.onerror = () => console.error('Failed to load graph image');
}

// Load & tampilkan animasi shortest-path
async function runShortest() {
  clearDisplays();
  const m   = encodeURIComponent(document.getElementById('matchSelect').value);
  const t   = encodeURIComponent(document.getElementById('teamSelect').value);
  const s   = encodeURIComponent(document.getElementById('srcSelect').value);
  const tgt = encodeURIComponent(document.getElementById('tgtSelect').value);
  const anim = document.getElementById('animImage');
  anim.src    = `${API}/analysis/shortest-path-gif?match=${m}&team=${t}&src=${s}&tgt=${tgt}`;
  anim.onload = () => anim.style.display = 'block';
  anim.onerror= () => console.error('Failed to load animation');
}

document.addEventListener('DOMContentLoaded', async () => {
  await initMatches();

  const matchSel = document.getElementById('matchSelect');
  const teamSel  = document.getElementById('teamSelect');

  // Ketika match dipilih → load tim
  matchSel.onchange = async () => {
    clearDisplays();
    teamSel.disabled = true;
    teamSel.innerHTML = '<option value="">-- Pilih Tim --</option>';
    if (!matchSel.value) return;
    const teams = await fetchJSON(`/teams?match=${encodeURIComponent(matchSel.value)}`);
    teams.forEach(t => teamSel.add(new Option(t, t)));
    teamSel.disabled = false;
  };

  // Setelah tim dipilih → isi player dropdown & enable tombol
  teamSel.onchange = () => {
    clearDisplays();
    const players = Array.from(teamSel.options)
                         .map(o => o.value).filter(v => v);
    ['srcSelect','tgtSelect','xtSrcSelect'].forEach(id => {
      const sel = document.getElementById(id);
      sel.innerHTML = '';
      players.forEach(p => sel.add(new Option(p, p)));
      sel.disabled = false;
    });
    // aktifkan semua tombol mode & analisis
    document.querySelectorAll('.mode-buttons button, #runShortest, #runXtGoal, #runCommunities')
      .forEach(b => b.disabled = false);
  };

  // Tombol graph
  document.querySelectorAll('.mode-buttons button')
    .forEach(btn => btn.onclick = () => loadGraphImage(btn.dataset.mode));

  // Shortest-path
  document.getElementById('runShortest').onclick = runShortest;

  // xT Goal Path & Communities masih tampil via alert
  document.getElementById('runXtGoal').onclick = async () => {
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    const s = encodeURIComponent(document.getElementById('xtSrcSelect').value);
    const res = await fetchJSON(`/analysis/xt-goal-path?match=${m}&team=${t}&src=${s}`);
    alert(JSON.stringify(res, null, 2));
  };
  document.getElementById('runCommunities').onclick = async () => {
    const m = encodeURIComponent(matchSel.value);
    const t = encodeURIComponent(teamSel.value);
    const res = await fetchJSON(`/analysis/communities?match=${m}&team=${t}`);
    alert(JSON.stringify(res, null, 2));
  };
});
