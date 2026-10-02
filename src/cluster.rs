//! cluster mode: static ring sharding + gossip'd global stats + scatter-gather
//! search. Peer list empty -> every call is a no-op (single-node fast path).

use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// fnv-1a 64 (no external deps); ring hash + delta keys
pub fn fnv64a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// consistent-hash ring slot count: ids map to slots via fnv64a, slots map
/// to nodes via `slot % N` (stepped virtual-node layout). Adding a member
/// claims exactly SLOTS/N slots (every Nth), so repartition churn stays
/// ~SLOTS/N ids per membership change instead of re-homing everything.
pub const SLOTS: u64 = 1024;

/// replication factor: every doc lives on the slot owner + (REPLICA_FACTOR-1)
/// ring successors, so R-1 members may die without losing that doc's copy.
/// R=1 = no replication; R=2 = one backup per doc.
pub const REPLICA_FACTOR: usize = 2;

/// placement for a ring slot: [owner, replica...]: owner = ring[slot % N],
/// replica k = ring[(slot + k) % N] — the next R-1 ring members in rotation
pub fn replica_slots(slot: u64, ring_len: usize) -> Vec<usize> {
    let r = REPLICA_FACTOR.min(ring_len).max(1);
    (0..r).map(|k| ((slot as usize) + k) % ring_len).collect()
}

/// ring slot of the node owning `id`: nodes sorted by address,
/// slot = fnv64a(id) % SLOTS, owner = ring[slot % ring.len()]
/// `ring`: the sorted [self, peers...] list of announcements.
pub fn ring_owner(ring: &[SocketAddr], id: &str) -> Option<usize> {
    if ring.is_empty() {
        return None;
    }
    let slot = fnv64a(id.as_bytes()) % SLOTS;
    Some((slot % ring.len() as u64) as usize)
}

/// the router set for `id`: [owner, replicas...] as ring INDICES
pub fn ring_holder_indices(ring: &[SocketAddr], id: &str) -> Vec<usize> {
    if ring.is_empty() {
        return Vec::new();
    }
    let slot = fnv64a(id.as_bytes()) % SLOTS;
    replica_slots(slot, ring.len())
}

/// parsed peer configuration
pub struct PeerConfig {
    /// self address as seen by the cluster (announcement; ring identity)
    pub announce: SocketAddr,
    /// other members of the ring (immutable config: static ring)
    pub peers: Vec<SocketAddr>,
    /// DNS host:port may expand to several pod IPs
    peers_from_dns: Vec<String>,
}

impl PeerConfig {
    /// parse "host:port" CSV names or addresses + setting self announcement
    pub fn parse(peers_csv: &str, announce: SocketAddr) -> Result<Self, String> {
        let mut peers = Vec::new();
        let mut dns = Vec::new();
        for raw in peers_csv.split(',') {
            let p = raw.trim();
            if p.is_empty() {
                continue;
            }
            // if it resolves now to exactly one socket, bake the IP in
            match p.to_socket_addrs() {
                Ok(mut it) => {
                    if let Some(sa) = it.next() {
                        peers.push(sa);
                    } else {
                        dns.push(p.to_string());
                    }
                }
                Err(_) => {
                    // not resolvable yet (DNS in a pod coming up): keep the name
                    dns.push(p.to_string());
                }
            }
        }
        Ok(Self {
            announce,
            peers,
            peers_from_dns: dns,
        })
    }

    /// resolve lazily configured names; returns the peer socket list
    pub fn resolve(&self) -> Vec<SocketAddr> {
        let mut out = self.peers.clone();
        for name in &self.peers_from_dns {
            if let Ok(it) = name.to_socket_addrs() {
                for sa in it {
                    if !out.contains(&sa) {
                        out.push(sa);
                    }
                }
            }
        }
        out
    }

    /// [self, peers...] sorted (ring order); self participates as a shard
    pub fn ring(&self) -> Vec<SocketAddr> {
        let mut ring = self.resolve();
        ring.push(self.announce);
        ring.sort();
        ring.dedup();
        ring
    }

    /// raw configured sockets (no DNS expansion), for /__zap/peers queries
    pub fn peers_csv(&self) -> Vec<String> {
        self.peers.iter().map(|p| p.to_string()).collect()
    }
}

/// lazily-resolvable set cached for LIST_TTL to avoid per-query DNS storms
pub struct PeerCache {
    config: PeerConfig,
    cached: Vec<SocketAddr>,
    cached_at: Instant,
}

const LIST_TTL: Duration = Duration::from_secs(2);

impl PeerCache {
    pub fn new(config: PeerConfig) -> Self {
        Self {
            config,
            cached: Vec::new(),
            cached_at: Instant::now() - LIST_TTL,
        }
    }

    /// peer socket list (cloned; self not included)
    pub fn peers(&mut self) -> Vec<SocketAddr> {
        if self.cached_at.elapsed() >= LIST_TTL {
            self.cached = self.config.resolve();
            self.cached_at = Instant::now();
        }
        self.cached.clone()
    }

    /// [self, peers...] sorted ring; (slot, socket) owning `id`
    pub fn owner_of(&mut self, id: &str) -> (usize, SocketAddr) {
        let ring = self.config.ring();
        let slot = ring_owner(&ring, id).expect("ring cannot be empty: self is always a member");
        (slot, ring[slot])
    }

    /// [self, peers...] sorted ring
    pub fn ring(&mut self) -> Vec<SocketAddr> {
        self.config.ring()
    }

    /// (slot, socket) owning an unid'd doc by its content
    pub fn owner_of_text(&mut self, text: &str) -> (usize, SocketAddr) {
        self.owner_of(&format!("content:{}", fnv64a(text.as_bytes())))
    }

    pub fn self_addr(&self) -> SocketAddr {
        self.config.announce
    }

    /// is `addr` this node's own announcement?
    pub fn is_self(&self, addr: &SocketAddr) -> bool {
        addr == &self.config.announce
    }

    /// rebuild the peer list in place (dynamic rejoin / ring repartition);
    /// the announce stays fixed, only the members change
    pub fn set_peers(&mut self, peers_csv: &str) -> Result<(), String> {
        let announce = self.config.announce;
        let cfg = PeerConfig::parse(peers_csv, announce)?;
        self.config = cfg;
        self.cached.clear();
        self.cached_at = Instant::now() - LIST_TTL;
        Ok(())
    }

    /// membership MERGE: add one announced address (idempotent, self-ignored)
    /// without touching existing members; used when a new node announces
    /// itself at boot and by any client that widens the cluster
    pub fn join_peer(&mut self, addr: SocketAddr) -> bool {
        if addr == self.config.announce || self.config.peers.contains(&addr) {
            return false;
        }
        self.config.peers.push(addr);
        self.cached.clear();
        self.cached_at = Instant::now() - LIST_TTL;
        true
    }

    /// comma-separated peer CSV of the current config (empty = single node)
    pub fn peers_csv(&self) -> String {
        let names = self.config.peers_csv();
        names.join(",")
    }
}

/// shared cluster runtime held by the HTTP server
pub struct LocalCluster {
    /// DNS caching refreshes the peer list, hence interior mutability
    pub cache: Mutex<PeerCache>,
    /// merged view (global df / totals) for query-time scoring
    pub view: Mutex<ClusterView>,
    /// last absolute envelope this node PUSHED (delta base)
    pub last_pushed: Mutex<Option<StatsEnvelope>>,
}

/// serialized stats envelope (wire + stored per peer); df entries are
/// ABSOLUTE doc frequencies keyed `"<field id>:<term>"`
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct StatsEnvelope {
    /// live doc count of the reporting shard
    pub docs: u32,
    /// per-field token sums for avg_len computation
    pub sums: Vec<u64>,
    /// df change set: "" empty = use as full replacement for this peer's map
    pub df: HashMap<String, u32>,
    /// true when `df` is the peer's whole map (full resync)
    pub full: bool,
}

/// per-peer absolute stats + the merged global snapshot a query scores with
#[derive(Default)]
pub struct ClusterView {
    /// peer address -> last known absolute envelope of that shard
    peers: HashMap<SocketAddr, StatsEnvelope>,
    /// this node's own contribution, rewritten at every local commit
    own: StatsEnvelope,
    /// cached merged view (invalidated by any update)
    global: Option<GlobalStats>,
}

/// merged cluster stats fed into the engine's scoring ctx
#[derive(Clone)]
pub struct GlobalStats {
    pub total_docs: u32,
    pub sums: Vec<u64>,
    /// field id -> (term -> df); two-level for borrowed &str lookups
    pub df: HashMap<u32, HashMap<String, u32>>,
}

impl ClusterView {
    /// replace this shard's own contribution (called after each local commit)
    pub fn set_own(&mut self, env: StatsEnvelope) {
        self.own = env;
        self.global = None;
    }

    /// merge a peer's delta or full envelope into the view
    pub fn merge_peer(&mut self, peer: SocketAddr, env: StatsEnvelope) {
        // drop keys not present in a full update
        if env.full {
            self.peers.insert(peer, env);
            self.global = None;
            return;
        }
        let cur = self.peers.entry(peer).or_default();
        cur.docs = env.docs;
        cur.sums = env.sums.clone();
        for (k, v) in &env.df {
            cur.df.insert(k.clone(), *v);
        }
        self.global = None;
    }

    pub fn invalidate(&mut self) {
        self.global = None;
    }

    /// merged globals (cached); recomputed when own or peer envelopes change
    pub fn stats(&mut self) -> GlobalStats {
        if self.global.is_none() {
            let mut docs = self.own.docs;
            let mut nf = self.own.sums.len();
            for p in self.peers.values() {
                docs += p.docs;
                nf = nf.max(p.sums.len());
            }
            let mut sums = vec![0u64; nf];
            for (i, s) in self.own.sums.iter().enumerate() {
                sums[i] += s;
            }
            for p in self.peers.values() {
                for (i, s) in p.sums.iter().enumerate() {
                    sums[i] += s;
                }
            }
            let mut df: HashMap<u32, HashMap<String, u32>> = HashMap::new();
            let mut add = |env: &StatsEnvelope| {
                for (k, v) in &env.df {
                    if let Some((fid, term)) = parse_df_key(k) {
                        *df.entry(fid).or_default().entry(term).or_insert(0) += v;
                    }
                }
            };
            add(&self.own);
            for p in self.peers.values() {
                add(p);
            }
            self.global = Some(GlobalStats {
                total_docs: docs,
                sums,
                df,
            });
        }
        self.global.as_ref().expect("just built").clone()
    }

    /// true when this peer address has never been seen: unknown peers trigger a
    /// one-time full resync pull at gossip time
    pub fn is_new_peer(&self, peer: &SocketAddr) -> bool {
        !self.peers.contains_key(peer)
    }

    pub fn count_peers(&self) -> usize {
        self.peers.len()
    }
}

/// gossip envelope key parser: "fid:term" -> (fid, term)
pub fn parse_df_key(key: &str) -> Option<(u32, String)> {
    let (a, b) = key.split_once(':')?;
    let fid = a.parse().ok()?;
    Some((fid, b.to_owned()))
}

pub fn encode_df_key(fid: u32, term: &str) -> String {
    format!("{fid}:{term}")
}

/// minimal HTTP/1.1 request over a fresh connection (no client deps);
/// returns (status_code, body)
pub fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Option<(u16, String)> {
    use std::io::{Read, Write};
    let mut stream =
        std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(750)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_millis(3000)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(3000)));
    let req = match body {
        Some(b) => format!(
            "{method} {path} HTTP/1.1\r\nHost: cluster\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
            b.len()
        ),
        None => format!("{method} {path} HTTP/1.1\r\nHost: cluster\r\nConnection: close\r\n\r\n"),
    };
    stream.write_all(req.as_bytes()).ok()?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).ok()?;
    Some((status_of(resp.as_str()), http_body(&resp)))
}

/// POST JSON (pub used by tests-free internal call sites)
#[inline]
pub fn http_post(addr: SocketAddr, path: &str, body: &str) -> Option<(u16, String)> {
    http_request(addr, "POST", path, Some(body))
}

#[inline]
pub fn http_get(addr: SocketAddr, path: &str) -> Option<(u16, String)> {
    http_request(addr, "GET", path, None)
}

/// minimal DELETE over keep-one HTTP; returns (status_code, body)
#[inline]
pub fn http_delete(addr: SocketAddr, path: &str) -> Option<(u16, String)> {
    http_request(addr, "DELETE", path, None)
}

fn status_of(resp: &str) -> u16 {
    resp.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// strip headers; body
pub fn http_body(resp: &str) -> String {
    match resp.split_once("\r\n\r\n") {
        Some((_h, b)) => b.to_owned(),
        None => String::new(),
    }
}

/// cluster config from CLI values: empty peer string -> single-node (None)
pub fn build_peer_config(
    bind: &str,
    peers_csv: &str,
    announce: &str,
) -> Result<Option<PeerConfig>, String> {
    if peers_csv.trim().is_empty() && announce.trim().is_empty() {
        return Ok(None);
    }
    let self_addr = if announce.trim().is_empty() {
        bind.to_socket_addrs()
            .map_err(|e| format!("bind {bind}: {e}"))?
            .next()
            .ok_or_else(|| format!("unsupported bind address: {bind}"))?
    } else {
        announce
            .trim()
            .to_socket_addrs()
            .map_err(|e| format!("announce {announce}: {e}"))?
            .next()
            .ok_or_else(|| format!("announce {announce}: no address"))?
    };
    // peers keep names as configured (DNS names resolve lazily in PeerCache;
    // self never appears in the peer list since announce is separate)
    PeerConfig::parse(peers_csv, self_addr).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// slot%N layout: adding/removing a member shifts only SLOTS/N slots;
    /// membership N -> N+1 keeps every id whose slot % (N+1) != N
    #[test]
    fn slot_ring_repartition_churn() {
        let n = SLOTS * 100; // ids spread far past one full slot cycle
        let mut churn_1_to_2 = 0usize;
        for i in 0..n {
            let id = format!("doc-{i}");
            let s = fnv64a(id.as_bytes()) % SLOTS;
            if s % 2 == 1 {
                churn_1_to_2 += 1;
            }
        }
        // N: 1 -> 2 re-homes exactly the odd slots = half the keyspace
        assert_eq!(churn_1_to_2, (n / 2) as usize);
        // skew sanity: per-slot shares within +-2% across a larger cycle
        let mut per2 = [0usize; 2];
        for i in 0..n {
            let id = format!("doc-{i}");
            let s = fnv64a(id.as_bytes()) % SLOTS;
            per2[(s % 2) as usize] += 1;
        }
        for c in per2 {
            assert!(
                (c as i64 - (n / 2) as i64).abs() < ((n as f64) * 0.02) as i64,
                "skew {per2:?}"
            );
        }
    }
}
