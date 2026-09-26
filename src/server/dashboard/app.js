// live refresh: pulls /stats (cluster-shards + counters) and re-renders
function fmt(n) { return n === null || n === undefined ? "–" : n.toLocaleString(); }
function uptime(s) {
  if (s === null || s === undefined) return "–";
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
}
async function tick() {
  try {
    const j = await (await fetch("/stats")).json();
    document.getElementById("v-docs").textContent = fmt(j.docs);
    document.getElementById("v-segments").textContent = fmt(j.segments);
    document.getElementById("v-gen").textContent = fmt(j.generation);
    const first = (j.shards || []).find((x) => x && x.addr && !x.down) || {};
    document.getElementById("v-searches").textContent = fmt(first.searches);
    document.getElementById("v-ingested").textContent = fmt(first.ingested);
    document.getElementById("v-uptime").textContent = uptime(first.uptime_s);
    const dot = document.getElementById("health-dot");
    dot.className = "dot" + (j.degraded ? " down" : "");
    const rows = (j.shards || []).map((s) => {
      if (!s || s.down || s.searches === undefined) {
        return (
          '<tr class="shard-row"><td>' + s.addr + "</td>" +
          '<td colspan="6" class="muted">unreachable</td>' +
          '<td class="down">DOWN</td></tr>'
        );
      }
      return (
        '<tr class="shard-row"><td><span class="addr">' + s.addr + "</span></td>" +
        "<td>" + fmt(s.docs) + "</td><td>" + fmt(s.segments) + "</td><td>" + fmt(s.generation) + "</td>" +
        '<td style="color:var(--mauve)">' + fmt(s.searches) + "</td>" +
        '<td style="color:var(--teal)">' + fmt(s.ingested) + "</td><td>" + uptime(s.uptime_s) + "</td>" +
        '<td class="up">UP</td></tr>'
      );
    });
    document.getElementById("shard-rows").innerHTML = rows.join("");
  } catch (e) {
    document.getElementById("health-dot").className = "dot down";
  }
}
tick();
// reactive: pushes over Server-Sent Events drive the same render; the timer
// stays as a slow fallback for missed events / idle-socket proxies
if (typeof EventSource !== "undefined") {
  const es = new EventSource("/dashboard/events");
  es.addEventListener("stats", () => tick());
  es.onerror = () => {
    // EventSource auto-reconnects; nothing to do here
  };
}
setInterval(tick, 8000);
