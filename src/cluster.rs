//! cluster mode: static ring sharding + gossip'd global stats + scatter-gather
//! search. Peer list empty -> every call is a no-op (single-node fast path).

use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, Mutex};
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

/// virtual nodes per member: the token ring gives every member VNODES
/// tokens, so the token space is N*VNODES wide and a membership change moves
/// only the keys whose range the new member claims (~1/(N+1) of the
/// keyspace). The OLD layout (`slot % N`, SLOTS=1024) re-homed ~N/(N+1) of
/// the keys on every N -> N+1 change — see `token_ring_churn` below.
///
/// VNODES also bounds the placement skew: a member owns a random set of
/// intervals, so its share deviates by ~1/sqrt(VNODES) (1024 vnodes -> a few
/// percent; 256 vnodes -> ~6%).
pub const VNODES: usize = 1024;

/// replication factor: every doc lives on the slot owner + (REPLICA_FACTOR-1)
/// ring successors, so R-1 members may die without losing that doc's copy.
/// R=1 = no replication; R=2 = one backup per doc.
pub const REPLICA_FACTOR: usize = 2;

/// golden-ratio odd constant: spreads the vnode indices of one member over
/// the token space (multiplication by an odd constant is a bijection)
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64 finalizer. FNV-1a alone mixes WEAKLY into the low bits (a
/// difference in the last input byte only shows up in the high bits), and a
/// token ring needs tokens AND keys spread uniformly over the whole 64-bit
/// space — otherwise every member's vnodes cluster in one region and the ring
/// degenerates into a list.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// placement hash of a routing key (an external id, or "text:<content-hash>"
/// for unid'd docs): fnv64a (the stable identity hash used by the wire
/// formats) + the mixer (uniform spread)
pub fn key_hash(id: &str) -> u64 {
    mix64(fnv64a(id.as_bytes()))
}

/// the ring view: the member list plus its virtual-token table.
///
/// Placement: a key hashes to `key_hash(key)` and lands on the FIRST token
/// at or above that hash (wrapping to the lowest token). Replicas are the
/// next distinct members in token order (successor replication), so a
/// member that dies does not take the doc with it.
#[derive(Clone, Debug, Default)]
pub struct Ring {
    /// members sorted by address (ring order)
    members: Vec<SocketAddr>,
    /// (token, member index) ascending by token
    tokens: Vec<(u64, usize)>,
}

impl Ring {
    /// build the token table for `members` (sorted + deduped)
    pub fn new(members: Vec<SocketAddr>) -> Self {
        let mut members = members;
        members.sort();
        members.dedup();
        let mut tokens = Vec::with_capacity(members.len() * VNODES);
        for (i, addr) in members.iter().enumerate() {
            let base = fnv64a(addr.to_string().as_bytes());
            for v in 0..VNODES {
                tokens.push((mix64(base ^ (v as u64).wrapping_mul(GOLDEN)), i));
            }
        }
        tokens.sort_unstable();
        Self { members, tokens }
    }

    pub fn members(&self) -> &[SocketAddr] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// the member index of `addr` in this view
    pub fn index_of(&self, addr: &SocketAddr) -> Option<usize> {
        self.members.iter().position(|m| m == addr)
    }

    /// index of the token that owns the key hash `h`: the first token >= h,
    /// wrapping around to the lowest token
    fn token_pos(&self, h: u64) -> usize {
        match self.tokens.binary_search_by(|(t, _)| t.cmp(&h)) {
            Ok(i) => i,
            Err(i) => {
                if i == self.tokens.len() {
                    0
                } else {
                    i
                }
            }
        }
    }

    /// the member index owning the key hash `h`
    pub fn owner_index(&self, h: u64) -> Option<usize> {
        if self.tokens.is_empty() {
            return None;
        }
        Some(self.tokens[self.token_pos(h)].1)
    }

    /// the holder set for the key hash `h`: the owner plus the next distinct
    /// members in token order
    pub fn holder_indices(&self, h: u64) -> Vec<usize> {
        if self.tokens.is_empty() {
            return Vec::new();
        }
        let r = REPLICA_FACTOR.min(self.members.len()).max(1);
        let start = self.token_pos(h);
        let mut out = vec![self.tokens[start].1];
        let mut i = 1usize;
        while out.len() < r && i < self.tokens.len() {
            let m = self.tokens[(start + i) % self.tokens.len()].1;
            if !out.contains(&m) {
                out.push(m);
            }
            i += 1;
        }
        out
    }

    /// the owner member index of a routing key
    pub fn owner_of_id(&self, id: &str) -> Option<usize> {
        self.owner_index(key_hash(id))
    }

    /// the holder member indices of a routing key
    pub fn holders_of_id(&self, id: &str) -> Vec<usize> {
        self.holder_indices(key_hash(id))
    }
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
    /// the ring view (member list + token table) for the current membership
    ring: Arc<Ring>,
}

const LIST_TTL: Duration = Duration::from_secs(2);

impl PeerCache {
    pub fn new(config: PeerConfig) -> Self {
        Self {
            config,
            cached: Vec::new(),
            cached_at: Instant::now() - LIST_TTL,
            // built on the first refresh (the TTL starts backdated)
            ring: Arc::new(Ring::default()),
        }
    }

    /// re-resolve the peers and rebuild the ring view (token table) — the
    /// token table is rebuilt only when the MEMBERSHIP actually changed
    fn refresh(&mut self) {
        self.cached = self.config.resolve();
        self.cached_at = Instant::now();
        let m = self.members();
        if self.ring.members() != m.as_slice() {
            self.ring = Arc::new(Ring::new(m));
        }
    }

    /// the member list (self + resolved peers), sorted by address
    fn members(&self) -> Vec<SocketAddr> {
        let mut m = self.cached.clone();
        m.push(self.config.announce);
        m.sort();
        m.dedup();
        m
    }

    /// peer socket list (cloned; self not included)
    pub fn peers(&mut self) -> Vec<SocketAddr> {
        if self.cached_at.elapsed() >= LIST_TTL {
            self.refresh();
        }
        self.cached.clone()
    }

    /// the ring view: members + token table. Rebuilt when the membership
    /// changes (join/set_peers invalidate) or the peer list is re-resolved.
    pub fn ring(&mut self) -> Arc<Ring> {
        if self.cached_at.elapsed() >= LIST_TTL {
            self.refresh();
        }
        self.ring.clone()
    }

    /// the ring member owning `id` (index + address)
    pub fn owner_of(&mut self, id: &str) -> (usize, SocketAddr) {
        let ring = self.ring();
        let idx = ring
            .owner_of_id(id)
            .expect("ring cannot be empty: self is always a member");
        (idx, ring.members()[idx])
    }

    /// the ring member owning an unid'd doc by its content
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

/// serialized stats envelope (wire + stored per peer); df is ABSOLUTE doc
/// frequencies per (field id, term), Arc-shared so envelope BUILDS are O(1)
/// (was: one "fid:term" String per vocabulary term per push — O(vocab) allocs
/// on the ingest critical path in cluster mode)
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct StatsEnvelope {
    /// live doc count of the reporting shard
    pub docs: u32,
    /// per-field token sums for avg_len computation
    pub sums: Vec<u64>,
    /// df change set: empty = use as full replacement for this peer's map
    pub df: Arc<HashMap<u32, HashMap<String, u32>>>,
    /// true when `df` is the peer's whole map (full resync)
    pub full: bool,
}

/// per-peer absolute stats; df kept OWNED so gossip delta folds are O(changed)
/// (an Arc'd df would force an O(vocab) rematerialization per delta apply)
#[derive(Default, Clone)]
pub struct PeerStats {
    pub docs: u32,
    pub sums: Vec<u64>,
    pub df: HashMap<u32, HashMap<String, u32>>,
}

/// per-peer absolute stats + the merged global snapshot a query scores with
#[derive(Default)]
pub struct ClusterView {
    /// peer address -> last known absolute stats of that shard
    peers: HashMap<SocketAddr, PeerStats>,
    /// this node's own contribution, rewritten at every local commit (Arc df:
    /// O(1) set — the envelope build is an Arc clone)
    own: StatsEnvelope,
    /// cached merged view (invalidated by any update)
    global: Option<GlobalStats>,
}

/// merged cluster stats fed into the engine's scoring ctx
#[derive(Clone)]
pub struct GlobalStats {
    pub total_docs: u32,
    pub sums: Vec<u64>,
    /// field id -> (term -> df); Arc so the per-query clone is O(1)
    pub df: Arc<HashMap<u32, HashMap<String, u32>>>,
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
            self.peers.insert(
                peer,
                PeerStats {
                    docs: env.docs,
                    sums: env.sums,
                    df: (*env.df).clone(),
                },
            );
            self.global = None;
            return;
        }
        let cur = self.peers.entry(peer).or_default();
        cur.docs = env.docs;
        cur.sums = env.sums.clone();
        for (fid, m) in env.df.iter() {
            let per = cur.df.entry(*fid).or_default();
            for (term, v) in m.iter() {
                per.insert(term.clone(), *v);
            }
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
            // own: Arc'd map, read-only fold
            for (fid, m) in self.own.df.iter() {
                let per = df.entry(*fid).or_default();
                for (term, v) in m.iter() {
                    *per.entry(term.clone()).or_insert(0) += v;
                }
            }
            // peers: owned maps, read-only fold
            for p in self.peers.values() {
                for (fid, m) in p.df.iter() {
                    let per = df.entry(*fid).or_default();
                    for (term, v) in m.iter() {
                        *per.entry(term.clone()).or_insert(0) += v;
                    }
                }
            }
            self.global = Some(GlobalStats {
                total_docs: docs,
                sums,
                df: Arc::new(df),
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

    fn members(n: u16) -> Vec<SocketAddr> {
        (0..n)
            .map(|i| SocketAddr::from(([10, 0, 0, i as u8], 8080)))
            .collect()
    }

    fn owners(ring: &Ring, h: u64) -> Vec<SocketAddr> {
        ring.holder_indices(h)
            .iter()
            .map(|&i| ring.members()[i])
            .collect()
    }

    /// the keyspace moves when a member JOINS: the new member claims
    /// ~1/(N+1) of the keys and the REST keep their owner. The old
    /// `slot % N` layout re-homed ~N/(N+1) of the keys on every N -> N+1
    /// change (a near-full reshuffle: 67% at 2->3, 75% at 3->4).
    #[test]
    fn token_ring_churn() {
        let n_ids = 20_000;
        for n in 1..=6u16 {
            let before = Ring::new(members(n));
            let after = Ring::new(members(n + 1));
            let mut moved = 0usize;
            let mut holder_moved = 0usize;
            for i in 0..n_ids {
                let h = key_hash(&format!("doc-{i}"));
                let ob = before.owner_index(h).expect("ring not empty");
                let oa = after.owner_index(h).expect("ring not empty");
                if before.members()[ob] != after.members()[oa] {
                    moved += 1;
                }
                if owners(&before, h) != owners(&after, h) {
                    holder_moved += 1;
                }
            }
            let churn = moved as f64 / n_ids as f64;
            // the new member's share is the FLOOR of the churn: the ring must
            // not move anything else
            let bound = 1.0 / (n as f64 + 1.0) + 0.10;
            assert!(
                churn < bound,
                "N={n} -> {}: owner churn {churn:.3} exceeds {bound:.3}",
                n + 1
            );
            // replicas are successors: their churn is the owner's plus the
            // new member's share as a successor — still far below the old
            // layout's ~N/(N+1)
            let hbound = 2.0 / (n as f64 + 1.0) + 0.16;
            let hchurn = holder_moved as f64 / n_ids as f64;
            assert!(
                hchurn < hbound,
                "N={n} -> {}: holder churn {hchurn:.3} exceeds {hbound:.3}",
                n + 1
            );
        }
    }

    /// every member owns ~1/N of the keyspace (no hot member, no dead member).
    /// The placement skew is bounded by the vnode count (~1/sqrt(VNODES)).
    #[test]
    fn token_ring_balance() {
        let n = 5u16;
        let ring = Ring::new(members(n));
        let n_ids = 50_000;
        let mut counts = vec![0usize; n as usize];
        for i in 0..n_ids {
            let h = key_hash(&format!("doc-{i}"));
            counts[ring.owner_index(h).expect("ring not empty")] += 1;
        }
        let expect = n_ids as f64 / n as f64;
        for c in &counts {
            assert!(
                (*c as f64 - expect).abs() < expect * 0.10,
                "skew: {counts:?} (expected ~{expect:.0})"
            );
        }
    }

    /// the same membership always places a key the same way (the ring view
    /// must agree across nodes, or the copies scatter)
    #[test]
    fn token_ring_deterministic() {
        let a = Ring::new(members(4));
        let b = Ring::new(members(4));
        for i in 0..1000 {
            let id = format!("doc-{i}");
            assert_eq!(a.owner_of_id(&id), b.owner_of_id(&id));
            assert_eq!(a.holders_of_id(&id), b.holders_of_id(&id));
        }
    }

    /// R=2 replication: a 2-member ring holds every doc on BOTH members, a
    /// 1-member ring holds it on the single member
    #[test]
    fn token_ring_replication() {
        let one = Ring::new(members(1));
        let two = Ring::new(members(2));
        let h = key_hash("doc-1");
        assert_eq!(one.holder_indices(h).len(), 1);
        assert_eq!(two.holder_indices(h).len(), 2);
        let three = Ring::new(members(3));
        assert_eq!(three.holder_indices(h).len(), 2);
    }
}
