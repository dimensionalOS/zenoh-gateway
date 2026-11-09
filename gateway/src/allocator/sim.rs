//! A bottleneck link with a queue, for driving a [`LinkModel`](super::LinkModel) the way a network drives one.
//!
//! # Why this exists
//!
//! Every bandwidth bug this crate has had needs a **standing queue** to appear, and until this module no test could
//! produce one. The older helpers (`run`, `page`) model a link as a *rate* and nothing else: a round's send is clipped
//! to `capacity` and the round trip is a constant handed in by the caller (`sample(.., 20.0)`). Sending more than the
//! link carries therefore has no consequence in a test — no queue forms, so the round trip never grows, so nothing
//! that reacts to a growing round trip can be exercised. Bugs of that shape were being found on a robot, by hand, and
//! only after they had shipped.
//!
//! The one that prompted this: `window` picks between a latency-first and a throughput-first mode, and crossing into
//! the second *raises* the ceiling (6.7x at the robot's bandwidth), which deepens the queue that chose it. Deciding
//! that on a round trip our own queue inflates makes it a one-way door, and the cost measured on the robot was 31% of
//! delivery, permanently, from one crossing. A control loop whose input its own output corrupts is the recurring shape
//! here, and a rate-only link cannot express it, because the corruption *is* the queue.
//!
//! # What it models, and what it deliberately does not
//!
//! A fluid bottleneck: one queue drained at `capacity`, a fixed `propagation_ms` under it, and a finite
//! `queue_limit_bytes` past which bytes are dropped. The round trip a sender sees is
//! `propagation_ms + queue / capacity` — the standing-queue delay that makes bufferbloat what it is. Per round it then
//! reports back exactly what [`crate::pacing`] reports from a real page: the delivery rates, the least round trip over
//! `RTT_WINDOW`, the median over `TYPICAL_RTT_WINDOW`, and the "quiet" median standing in for frames sent with almost
//! nothing of ours unacked.
//!
//! It is a *fluid* model on purpose: no per-packet timing, no reordering, no ack compression, and in-flight is
//! approximated as what sits in the queue plus what is on the wire. Those omissions cost nothing for the questions it
//! is here to answer, which are about where a control loop settles over seconds. It is not a substitute for measuring
//! on a robot, and a result from it should never be reported as one.

use super::{LinkModel, SLOW_LINK};
use crate::pacing::RoundSamples;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// `crate::pacing::RTT_WINDOW`.
const RTT_WINDOW: Duration = Duration::from_secs(10);
/// `crate::pacing::TYPICAL_RTT_WINDOW`.
const TYPICAL_RTT_WINDOW: Duration = Duration::from_secs(2);

/// What one round of a simulated link did, for a test to assert on.
///
/// Every field is part of the model's observable surface whether or not a test reads it today: these are what a
/// failure message needs to be diagnosable, and `sweep_schedules` prints most of them. Dropping the ones no assertion
/// currently touches would make the next investigation start by adding them back.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct Round {
    pub at: Duration,
    /// what the sender was allowed (bandwidth x gain), and the in-flight ceiling `window` gave it
    pub budget_bytes_per_sec: f64,
    pub window_bytes: usize,
    /// what it actually put on the link, and what came off the far end
    pub offered_bytes_per_sec: f64,
    pub delivered_bytes_per_sec: f64,
    /// bytes the queue held at the end of the round, and what it dropped for being full
    pub queue_bytes: f64,
    pub dropped_bytes: f64,
    /// the round trip a frame sent at the end of this round would see
    pub rtt_ms: f64,
    /// the model's own view, after being told about this round
    pub estimate_bytes_per_sec: f64,
    pub latency_first: bool,
    /// the model's pacing gain and whether it is still starting up, which decide the two above
    pub gain: f64,
    pub starting_up: bool,
    /// what the link was carrying this round
    pub carrying_bytes_per_sec: f64,
    /// which limit held the sender back
    pub window_waited: bool,
    pub link_limited: bool,
}

/// A fluid bottleneck: `capacity_bytes_per_sec` drained through one queue of at most `queue_limit_bytes`, with
/// `propagation_ms` of delay under it.
///
/// # A constant rate cannot build a standing queue, and that is not a flaw in the model
///
/// BBR paces at its estimate with a steady-state gain of 1, and its estimate is what the link delivered — so on a link
/// that always carries the same rate the sender converges to exactly that rate and the queue stays empty, by design.
/// A standing queue needs the **estimate to exceed the rate the link is carrying right now**, and on a real link that
/// is the normal condition rather than a rare one: the bandwidth is a *max filter* over about ten rounds, so a link
/// that varies (Wi-Fi, a tunnel, a shared uplink) is driven at its recent best while it is going through its recent
/// worst. On the robot the estimate sat at 2.8 MB/s against a p50 delivery of 1.28 MB/s with peaks of 4.27 — not an
/// error, just a max filter doing its job on a variable path.
///
/// So [`with_capacity`](Self::with_capacity) is what makes this model able to ask the interesting questions, and a test
/// using a constant rate is testing a link that does not exist outside a lab.
pub struct Link {
    pub capacity_bytes_per_sec: f64,
    /// what the link carries at a given point into the run, when it is not simply `capacity_bytes_per_sec`
    #[allow(clippy::type_complexity)]
    varying: Option<Box<dyn Fn(Duration) -> f64>>,
    pub propagation_ms: f64,
    /// the path's own delay at a given point into the run, when it is not simply `propagation_ms`
    #[allow(clippy::type_complexity)]
    wobbling: Option<Box<dyn Fn(Duration) -> f64>>,
    pub queue_limit_bytes: f64,
    /// a small stream that sends every round whatever the window says (see `with_trickle`)
    trickle_bytes_per_sec: f64,
    queue_bytes: f64,
    /// (when, round trip) a sender would have seen, for the three filters pacing reports
    rtts: VecDeque<(Instant, f64)>,
    /// ... and the subset standing in for frames sent with almost nothing of ours unacked (see `round`)
    quiet_rtts: VecDeque<(Instant, f64)>,
    /// when the first round ran, so a `Round` can say how far into the run it is
    start: Option<Instant>,
}

impl Link {
    /// A link that carries `capacity_bytes_per_sec`, `propagation_ms` round trip when its queue is empty, and a buffer
    /// one `propagation` worth of bandwidth deep unless `with_queue_limit` says otherwise (a shallow default would hide
    /// bufferbloat, and an unbounded one would never drop).
    pub fn new(capacity_bytes_per_sec: f64, propagation_ms: f64) -> Self {
        Self {
            capacity_bytes_per_sec,
            propagation_ms,
            // a second of bandwidth: deep enough for a standing queue to form and be felt, as a consumer router's is
            queue_limit_bytes: capacity_bytes_per_sec,
            varying: None,
            wobbling: None,
            trickle_bytes_per_sec: 0.0,
            queue_bytes: 0.0,
            rtts: VecDeque::new(),
            quiet_rtts: VecDeque::new(),
            start: None,
        }
    }

    pub fn with_queue_limit(mut self, bytes: f64) -> Self {
        self.queue_limit_bytes = bytes;
        self
    }

    /// A link whose rate changes as the run goes on, which is the only kind that can hold a standing queue (see the
    /// note on [`Link`]). `capacity_bytes_per_sec` stays the nominal rate the buffer is sized from.
    pub fn with_capacity(mut self, carrying: impl Fn(Duration) -> f64 + 'static) -> Self {
        self.varying = Some(Box::new(carrying));
        self
    }

    /// A low-rate stream alongside the bulk one — `tf` at 5 KB/s, odometry at 2 — which is what actually samples the
    /// path on a congested link, and without which a model cannot produce this class of bug at all.
    ///
    /// A bulk sender pinned against the window stops contributing "quiet" round trips exactly when the queue is
    /// deepest, so a single-stream model measures the path only during the gaps and reads it far too low (a sweep here
    /// topped out at 197 ms against a real 569). A small stream always has its next message ready, is never the thing
    /// the window holds back, and pays the standing queue like everything else — so it keeps reporting a round trip
    /// taken with almost nothing of ours unacked, throughout the congestion. That is the sample that told the old code
    /// the *path* was slow when it was our own queue.
    pub fn with_trickle(mut self, bytes_per_sec: f64) -> Self {
        self.trickle_bytes_per_sec = bytes_per_sec;
        self
    }

    /// What the link is carrying `elapsed` into the run.
    fn carrying(&self, elapsed: Duration) -> f64 {
        self.varying.as_ref().map_or(self.capacity_bytes_per_sec, |carrying| carrying(elapsed))
    }

    /// A path whose **own delay** moves, independently of any queue — which is the trigger this model could not
    /// express before, and the one that actually fired on the robot.
    ///
    /// The capture shows it plainly: two polls had a least round trip over the 300 ms target, and both had
    /// `inFlightBytes` of **25 KB** — a near-empty window. At 21.5 s the least was **494 ms** while the queue-blind
    /// sample read **72.5 ms**, so the path went slow with nothing of ours waiting anywhere. One poll later the mode
    /// flipped, and from then on our own queue kept it flipped. **The latch did not need a queue to start; it needed
    /// one bad second of Wi-Fi.** A model with a fixed propagation delay can never produce that, so it can never show
    /// that a fix survives it.
    pub fn with_path_delay(mut self, wobbling: impl Fn(Duration) -> f64 + 'static) -> Self {
        self.wobbling = Some(Box::new(wobbling));
        self
    }

    /// The path's own delay `elapsed` into the run.
    fn path_delay_ms(&self, elapsed: Duration) -> f64 {
        self.wobbling.as_ref().map_or(self.propagation_ms, |wobbling| wobbling(elapsed))
    }

    /// The round trip through the queue as it stands: the path itself plus how long what is waiting takes to drain at
    /// the rate the link is carrying.
    fn rtt_ms(&self, carrying: f64, path_ms: f64) -> f64 {
        path_ms + self.queue_bytes / carrying * 1000.0
    }

    /// Bytes sent and not yet acked: what is still queued, plus what the link holds while it propagates.
    fn in_flight_bytes(&self, carrying: f64, path_ms: f64) -> f64 {
        self.queue_bytes + carrying * path_ms / 1000.0
    }

    fn filtered(&self, now: Instant) -> (Option<f64>, Option<f64>, Option<f64>) {
        let least = self.rtts.iter().map(|(_, rtt)| *rtt).fold(f64::INFINITY, f64::min);
        let median = |mut taken: Vec<f64>| {
            taken.sort_by(f64::total_cmp);
            taken.get(taken.len() / 2).copied()
        };
        let typical =
            median(self.rtts.iter().filter(|(at, _)| now.duration_since(*at) <= TYPICAL_RTT_WINDOW).map(|(_, rtt)| *rtt).collect());
        let quiet = median(self.quiet_rtts.iter().map(|(_, rtt)| *rtt).collect());
        (least.is_finite().then_some(least), typical, quiet)
    }

    /// Carries one round: the sender offers `demand_bytes_per_sec`, held to what the model's budget and window allow,
    /// and the model is told what came back. Returns what happened, for assertions.
    pub fn round(
        &mut self,
        model: &mut LinkModel,
        now: Instant,
        secs: f64,
        demand_bytes_per_sec: f64,
        latency_target: Option<Duration>,
    ) -> Round {
        let elapsed = self.start.map_or(Duration::ZERO, |start| now.saturating_duration_since(start));
        let carrying = self.carrying(elapsed);
        let path_ms = self.path_delay_ms(elapsed);
        let (budget, _) = model.pacing(latency_target);
        let (window, target_capped) = model.window(latency_target);
        // A window is per round trip, not per round: a sender allowed `window` bytes unacked sustains `window / rtt`,
        // and a round fits `secs / rtt` of those. (Capping it at `window` per round instead understates the sender by
        // the ratio of the two — a 12 ms path on 250 ms rounds trickles at a twentieth of the rate, no queue ever
        // forms, and the model under test is never asked a hard question.)
        let window_rate = window as f64 / (self.rtt_ms(carrying, path_ms) / 1000.0);
        let wanted = demand_bytes_per_sec * secs;
        let allowed = wanted.min(budget * secs);
        let offered = allowed.min(window_rate * secs);
        // Materially, not by a hair: the window is `bandwidth x rtt x gain` truncated to whole bytes, so `window / rtt`
        // lands a rounding error under `bandwidth x gain` and a bare `<` calls every round window-held. That mattered —
        // it routed every sample to `latency_limited_rate`, which by design does not raise the bandwidth, so the
        // estimate froze at the first guess and the whole run was a trickle.
        let window_waited = offered < allowed - allowed.abs().max(1.0) * 1e-6;
        // Held by the window when the target did not set it is the link's own bandwidth-delay product holding us, which
        // is a true sample of the link; held by the target's cap is not (the sender wanted more and the link may carry
        // it), and held by our own pacing is neither.
        let link_limited = window_waited && !target_capped;
        let latency_limited = window_waited && target_capped;

        // the small stream goes out regardless: it is never what the window holds back
        let trickle = self.trickle_bytes_per_sec * secs;
        let offered = offered + trickle;
        let dropped = (self.queue_bytes + offered - self.queue_limit_bytes).max(0.0).min(offered);
        self.queue_bytes += offered - dropped;
        let drained = self.queue_bytes.min(carrying * secs);
        self.queue_bytes -= drained;

        let in_flight = self.in_flight_bytes(carrying, path_ms);
        let rtt = self.rtt_ms(carrying, path_ms);
        while self.rtts.front().is_some_and(|(at, _)| now.duration_since(*at) > RTT_WINDOW) {
            self.rtts.pop_front();
        }
        while self.quiet_rtts.front().is_some_and(|(at, _)| now.duration_since(*at) > RTT_WINDOW) {
            self.quiet_rtts.pop_front();
        }
        self.rtts.push_back((now, rtt));
        // A quiet sample is a frame sent while at most `pacing::QUIET_BYTES` (8 KB) of ours was unacked, and the
        // subtlety that matters
        // is that **unacked is not the same as queued**: the bytes we are waiting on are mostly sitting in a router,
        // and our own unacked count sweeps between nearly nothing and the window within a round as acks come back.
        // A fluid model reports one average per round, and here that average is a whole propagation's worth
        // (3.5 MB/s x 12 ms = 42 KB, already over that bar), so taking the in-flight test literally means
        // a quiet sample is never recorded at all — which is why this model first "passed" against the very code it
        // was written to indict.
        //
        // The real ones come from the *small* channels (tf at 5 KB/s, odom at 2): a 2 KB message sent in a gap sees
        // almost nothing of ours unacked and still waits behind the whole standing queue, which is exactly how a
        // "queue-free" sample comes back reading 569 ms. A round has such a gap whenever the sender was not pinned
        // against the window for all of it, so that is the condition, and the sample pays the queue like any other.
        if self.trickle_bytes_per_sec > 0.0 || !window_waited {
            self.quiet_rtts.push_back((now, rtt));
        }
        let (min_rtt_ms, typical_rtt_ms, path_rtt_ms) = self.filtered(now);

        let delivered = drained / secs;
        let samples = RoundSamples {
            link_limited_rate: link_limited.then_some(delivered),
            app_limited_rate: (!window_waited).then_some(delivered),
            latency_limited_rate: latency_limited.then_some(delivered),
            min_rtt_ms,
            typical_rtt_ms,
            path_rtt_ms,
            extra_acked: 0.0,
            drained: in_flight <= window as f64,
            window_waited,
        };
        model.update(now, secs, samples, 0.0, false, delivered);

        let start = *self.start.get_or_insert(now);
        Round {
            at: now.saturating_duration_since(start),
            budget_bytes_per_sec: budget,
            window_bytes: window,
            offered_bytes_per_sec: offered / secs,
            delivered_bytes_per_sec: delivered,
            queue_bytes: self.queue_bytes,
            dropped_bytes: dropped,
            rtt_ms: rtt,
            estimate_bytes_per_sec: model.bandwidth_bytes_per_sec,
            gain: model.gain,
            starting_up: model.starting_up,
            carrying_bytes_per_sec: carrying,
            window_waited,
            link_limited,
            latency_first: model
                .path_floor_ms()
                .is_some_and(|floor| latency_target.is_some_and(|target| floor < target.as_secs_f64() * 1000.0)),
        }
    }

    /// `rounds` rounds of `secs` each at a steady `demand_bytes_per_sec`, from `start`.
    pub fn run(
        &mut self,
        model: &mut LinkModel,
        start: Instant,
        rounds: u32,
        secs: f64,
        demand_bytes_per_sec: f64,
        latency_target: Option<Duration>,
    ) -> Vec<Round> {
        (0..rounds)
            .map(|round| {
                let now = start + Duration::from_secs_f64(secs * round as f64);
                self.round(model, now, secs, demand_bytes_per_sec, latency_target)
            })
            .collect()
    }
}

/// The ceiling the throughput-first mode would allow at this bandwidth, for a test or a sweep to show how far above
/// the target it is: `target x sqrt(bandwidth / SLOW_LINK)`.
#[allow(dead_code)]
pub fn throughput_first_limit_ms(bandwidth_bytes_per_sec: f64, target: Duration) -> f64 {
    target.as_secs_f64() * 1000.0 * (bandwidth_bytes_per_sec / SLOW_LINK).sqrt()
}

/// The largest a field reached over a whole run. The right statistic for a latch: the crossing that closes one is a
/// transient, and a median hides it.
pub fn peak(rounds: &[Round], field: impl Fn(&Round) -> f64) -> f64 {
    rounds.iter().map(field).fold(f64::NEG_INFINITY, f64::max)
}

/// A convenience for assertions: the median of a field over the second half of a run, where a control loop has
/// settled (the first half is startup and probing, which is a different question).
pub fn settled(rounds: &[Round], field: impl Fn(&Round) -> f64) -> f64 {
    let mut tail: Vec<f64> = rounds[rounds.len() / 2..].iter().map(field).collect();
    tail.sort_by(f64::total_cmp);
    tail[tail.len() / 2]
}

/// A named network scenario, so a change to the control loop is scored against every shape of link that has ever
/// caused trouble here rather than against one.
///
/// This is the part that makes the harness a regression suite instead of a single experiment: production RTC stacks
/// test a controller against a library of profiles (good Wi-Fi, a congested shared uplink, a path whose delay swings,
/// a link that drops out and returns) and assert on *outcomes* — what fraction of the link was delivered, how deep the
/// standing queue got, how long recovery took — rather than on internal fields. Asserting on outcomes is what lets the
/// mechanism be rewritten without rewriting the tests, which is exactly the freedom this crate has been missing.
pub struct Profile {
    pub name: &'static str,
    /// the link's nominal rate, what it carries over time, and the path's own delay over time
    pub nominal_bytes_per_sec: f64,
    pub carrying: Option<Schedule>,
    pub path_delay_ms: Option<Schedule>,
    pub propagation_ms: f64,
    /// the small frequent topics that sample the path even while the bulk stream is window-bound
    pub trickle_bytes_per_sec: f64,
    /// what the page is asking for
    pub demand_bytes_per_sec: f64,
}

/// What a profile's run came to, in the terms a reviewer should be arguing about.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    /// share of **what was both wanted and available** that arrived: `min(demand, capacity)` summed over the run, not
    /// the link's capacity, which a page asking for less than the link can never reach
    pub delivered_fraction: f64,
    /// the deepest standing queue, as the delay it added (ms), and the peak window it allowed
    pub peak_queue_ms: f64,
    pub peak_window_bytes: f64,
    /// the largest the window was *allowed* to be in latency-first mode, taken per round. It has to be per round: the
    /// bound is `bandwidth x gain x target` and both move, and comparing a peak window against the bandwidth and gain
    /// the run happened to *end* on marks a perfectly bounded run a failure (startup's gain of 2.89 is long gone by
    /// then). My first version of this did exactly that, on three profiles.
    pub peak_allowed_window_bytes: f64,
    /// rounds that left latency-first while the path was not in fact slow — the latch, counted
    pub rounds_latched: usize,
    pub rounds: usize,
    /// **Ground truth, from the profile and not from the model**: was the path's own delay at or over the target for
    /// the whole run? This is what says whether relaxing the window was legitimate.
    ///
    /// It must come from the simulator, and the reason is a mistake this file made: the first version asked the model
    /// for `path_floor_ms()` instead. Against the fixed code that reads 12 ms and the assertions run — but against the
    /// *old* code it reads 620 ms, because our own queue had corrupted it, so the test called the run "a genuinely
    /// slow path", excused it, and **passed on the very bug it exists to catch**. A test may never take its oracle
    /// from the thing under test.
    pub path_was_slow: bool,
    /// the least the path's own delay actually was, from the profile — the number `path_was_slow` is read from
    pub true_path_floor_ms: f64,
}

/// A value over time (a link's rate, or a path's delay). Shared rather than borrowed so a `Profile` can hand one to a
/// `Link` that outlives the call without a lifetime parameter spreading through the model.
pub type Schedule = std::sync::Arc<dyn Fn(Duration) -> f64 + Send + Sync>;

/// A schedule from a closure.
pub fn schedule(over_time: impl Fn(Duration) -> f64 + Send + Sync + 'static) -> Schedule {
    std::sync::Arc::new(over_time)
}

/// A schedule that never changes.
pub fn steady(value: f64) -> Schedule {
    schedule(move |_| value)
}

impl Profile {
    fn link(&self) -> Link {
        let mut link = Link::new(self.nominal_bytes_per_sec, self.propagation_ms).with_trickle(self.trickle_bytes_per_sec);
        if let Some(carrying) = self.carrying.clone() {
            link = link.with_capacity(move |elapsed| carrying(elapsed));
        }
        if let Some(delay) = self.path_delay_ms.clone() {
            link = link.with_path_delay(move |elapsed| delay(elapsed));
        }
        link
    }

    /// Runs the profile for `rounds` rounds of `secs` and scores it.
    pub fn score(&self, rounds: u32, secs: f64, latency_target: Option<Duration>) -> (Outcome, LinkModel) {
        let mut link = self.link();
        let mut model = LinkModel::default();
        let carried: Vec<f64> = (0..rounds).map(|round| link.carrying(Duration::from_secs_f64(secs * round as f64))).collect();
        let played = link.run(&mut model, Instant::now(), rounds, secs, self.demand_bytes_per_sec, latency_target);

        // what was wanted *and* there to be had: a page asking 2.5 MB/s of a 3.5 MB/s link can only ever get 71% of
        // the link, so scoring against the link alone would mark a perfect run a failure
        let could = carried.iter().map(|carrying| carrying.min(self.demand_bytes_per_sec + self.trickle_bytes_per_sec)).sum::<f64>() * secs;
        let did = played.iter().map(|round| round.delivered_bytes_per_sec).sum::<f64>() * secs;
        let target_ms = latency_target.map_or(f64::INFINITY, |target| target.as_secs_f64() * 1000.0);
        // the path's own delay over the run, as the profile defines it — never as the model perceives it
        let truly = |elapsed: Duration| self.path_delay_ms.as_ref().map_or(self.propagation_ms, |delay| delay(elapsed));
        let true_path_floor_ms = (0..rounds).map(|round| truly(Duration::from_secs_f64(secs * round as f64))).fold(f64::INFINITY, f64::min);
        let path_was_slow = true_path_floor_ms >= target_ms;
        (
            Outcome {
                delivered_fraction: if could > 0.0 { (did / could).min(1.0) } else { 0.0 },
                peak_queue_ms: played
                    .iter()
                    .zip(&carried)
                    .map(|(round, carrying)| round.queue_bytes / carrying * 1000.0)
                    .fold(0.0, f64::max),
                peak_window_bytes: peak(&played, |round| round.window_bytes as f64),
                peak_allowed_window_bytes: peak(&played, |round| round.estimate_bytes_per_sec * round.gain.max(1.0) * target_ms / 1000.0),
                // the floor being under the target is what makes leaving the mode a latch rather than a slow path
                rounds_latched: played.iter().filter(|round| !round.latency_first).count(),
                rounds: played.len(),
                path_was_slow,
                true_path_floor_ms,
            },
            model,
        )
    }
}

/// Every shape of link that has caused trouble here, or that a reviewer would reasonably ask about.
pub fn profiles() -> Vec<Profile> {
    vec![
        Profile {
            name: "a good link, nothing moving",
            nominal_bytes_per_sec: 3_500_000.0,
            carrying: Some(steady(3_500_000.0)),
            path_delay_ms: None,
            propagation_ms: 12.0,
            trickle_bytes_per_sec: 7_000.0,
            demand_bytes_per_sec: 2_500_000.0,
        },
        Profile {
            // the robot's tunnel over Wi-Fi: the max filter holds the good phase while the bad one is being carried
            name: "a varying link, 3.5 MB/s then 1.2",
            nominal_bytes_per_sec: 3_500_000.0,
            carrying: Some(schedule(|elapsed| if elapsed.as_secs_f64() % 10.0 < 3.0 { 3_500_000.0 } else { 1_200_000.0 })),
            path_delay_ms: None,
            propagation_ms: 12.0,
            trickle_bytes_per_sec: 7_000.0,
            demand_bytes_per_sec: 2_500_000.0,
        },
        Profile {
            // THE ONE THAT HAPPENED: a 12 ms path that spikes to 494 ms for a second, with an empty window
            name: "a one-second path spike on a quiet link",
            nominal_bytes_per_sec: 2_500_000.0,
            carrying: Some(steady(2_500_000.0)),
            path_delay_ms: Some(schedule(|elapsed| if (20.0..21.0).contains(&elapsed.as_secs_f64()) { 494.0 } else { 12.0 })),
            propagation_ms: 12.0,
            trickle_bytes_per_sec: 7_000.0,
            demand_bytes_per_sec: 2_000_000.0,
        },
        Profile {
            // ... and the same spike while the link is also varying, which is the robot's actual combination
            name: "a path spike while the link varies",
            nominal_bytes_per_sec: 3_500_000.0,
            carrying: Some(schedule(|elapsed| if elapsed.as_secs_f64() % 10.0 < 3.0 { 3_500_000.0 } else { 1_200_000.0 })),
            path_delay_ms: Some(schedule(|elapsed| if (20.0..21.0).contains(&elapsed.as_secs_f64()) { 494.0 } else { 12.0 })),
            propagation_ms: 12.0,
            trickle_bytes_per_sec: 7_000.0,
            demand_bytes_per_sec: 2_500_000.0,
        },
        Profile {
            // a genuinely slow path, where relaxing the ceiling is the RIGHT answer and a cap would starve it
            name: "a genuinely slow path, 400 ms throughout",
            nominal_bytes_per_sec: 2_500_000.0,
            carrying: Some(steady(2_500_000.0)),
            path_delay_ms: Some(steady(400.0)),
            propagation_ms: 400.0,
            trickle_bytes_per_sec: 7_000.0,
            demand_bytes_per_sec: 2_000_000.0,
        },
        Profile {
            // the link drops to almost nothing and comes back: does delivery return?
            name: "a link that drops out for 5 s and returns",
            nominal_bytes_per_sec: 2_500_000.0,
            carrying: Some(schedule(|elapsed| if (20.0..25.0).contains(&elapsed.as_secs_f64()) { 20_000.0 } else { 2_500_000.0 })),
            path_delay_ms: None,
            propagation_ms: 12.0,
            trickle_bytes_per_sec: 7_000.0,
            demand_bytes_per_sec: 2_000_000.0,
        },
    ]
}
