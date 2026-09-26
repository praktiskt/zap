//! dashboard event bus: push notifications for the SSE stream. Subscribers
//! register (mpsc); dead SSE sockets get reaped on the next broadcast.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Mutex, OnceLock};

/// event name the SSE clients listen for
pub const EVENT_STATS: &str = "stats";

/// registered dashboard streams (id -> sender)
fn clients() -> &'static Mutex<HashMap<u64, Sender<&'static str>>> {
    static CLIENTS: OnceLock<Mutex<HashMap<u64, Sender<&'static str>>>> = OnceLock::new();
    CLIENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// register a dashboard stream; returns (id, receiver)
pub fn subscribe() -> (u64, Receiver<&'static str>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    clients().lock().unwrap().insert(id, tx);
    (id, rx)
}

/// drop one subscriber (called when its stream closes)
pub fn unsubscribe(id: u64) {
    clients().lock().unwrap().remove(&id);
}

/// broadcast an event name; subscribers with closed receivers fall out
pub fn notify(event: &'static str) {
    let map = clients().lock().unwrap();
    if map.is_empty() {
        return;
    }
    let alive: Vec<u64> = map
        .iter()
        .filter_map(|(id, tx)| tx.send(event).ok().map(|_| *id))
        .collect();
    drop(map);
    let mut all = clients().lock().unwrap();
    all.retain(|id, _| alive.contains(id));
}

/// notify the dashboard that counters/stats changed
pub fn notify_stats() {
    notify(EVENT_STATS);
}
