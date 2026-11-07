//! Measures a gateway's video stream without a browser: connect as a client, subscribe to one camera on one channel,
//! and print frame rate, latency, drops and keyframe spacing.
//!
//! A robot's camera usually has one consumer, so opening a browser to look at it takes the stream away from whatever
//! else is watching, and a browser tells you little about *why* a stream is slow. This says how many frames arrived,
//! how far behind they were and what was lost, which is what a change to the gateway has to move.
//!
//! ```text
//! cargo run --release --example video_probe --features client -- \
//!     --url http://m20-gos:7447 --key camera/front --encoding ros2_compressed_video \
//!     --channel video-h265 --seconds 20
//! ```
//!
//! `--channel video-h265` takes the publisher's H.265 straight through (what a Chrome or Safari does); `video-h264`
//! makes the gateway decode and re-encode it (what a Firefox does). Running both against the same camera is the
//! comparison the two paths exist to settle.
//!
//! **Latency.** Two numbers, because they answer different questions:
//! - *gateway* — how long the gateway held the sample, from `receivedMs` (when its sample arrived) to `writtenMs` (when
//!   the frame went to the track). Both are the gateway's own clock, so no clock comparison is involved and this number
//!   is exact. It is what decode-and-re-encode costs and what passthrough is meant to remove.
//! - *end to end* — from `receivedMs` to this process receiving the frame's last packet, corrected by the clock offset
//!   the client measures over the control channel. Only as good as that offset (the probe prints it), so read it as
//!   approximate unless the probe and the gateway are on one machine.
//!
//! **Drops** come from `seq` on the per-frame metadata, which numbers messages per subscription: a gap is a frame the
//! gateway sent that never arrived whole, and for a passthrough stream each one costs the picture until the publisher's
//! next keyframe, so the keyframe spacing below says how long that is.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use zenoh_gateway::client::{Client, ClientOptions, Message, SubscribeOptions};

/// What the command line asked for.
struct Options {
    url: String,
    key: String,
    encoding: Option<String>,
    channel: String,
    seconds: f64,
    /// print a line per frame as well as the summary
    verbose: bool,
    /// bearer token (app's zenohGatewayToken) for a gateway with an authorize hook
    token: Option<String>,
}

fn usage() -> String {
    "video_probe --url <gateway url> --key <camera key> [--encoding <name>] [--channel video-h265] [--seconds 20] [--verbose]".to_owned()
}

fn options() -> Result<Options> {
    let mut options = Options { url: String::new(), key: String::new(), encoding: None, channel: "video-h264".to_owned(), seconds: 20.0, verbose: false, token: None };
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        let mut value = || args.next().with_context(|| format!("{argument} needs a value\n{}", usage()));
        match argument.as_str() {
            "--url" => options.url = value()?,
            "--key" => options.key = value()?,
            "--encoding" => options.encoding = Some(value()?),
            "--channel" => options.channel = value()?,
            "--seconds" => options.seconds = value()?.parse().context("--seconds takes a number")?,
            "--verbose" => options.verbose = true,
            "--token" => options.token = Some(value()?),
            "--help" | "-h" => bail!("{}", usage()),
            other => bail!("unknown argument {other}\n{}", usage()),
        }
    }
    if options.url.is_empty() || options.key.is_empty() {
        bail!("--url and --key are needed\n{}", usage());
    }
    Ok(options)
}

/// One frame, as the probe saw it: its metadata and when its last packet arrived here.
#[derive(Default)]
struct Frame {
    /// the gateway's clock: when it got the sample
    received_ms: Option<f64>,
    /// the gateway's clock: when the frame went to the track
    written_ms: Option<f64>,
    /// this process's clock: when its last packet arrived (absent if only the metadata came)
    arrived: Option<Instant>,
    keyframe: bool,
    bytes: u32,
    size: (u32, u32),
}

/// The ms at `share` through `values` (which it sorts), or `None` when there are none.
fn percentile(values: &mut [f64], share: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let at = ((values.len() - 1) as f64 * share).round() as usize;
    Some(values[at])
}

/// `name: p50 / p95 / max ms` over `values`, or that there were none.
fn spread(name: &str, values: &mut [f64]) -> String {
    match (percentile(values, 0.5), percentile(values, 0.95), percentile(values, 1.0)) {
        (Some(p50), Some(p95), Some(max)) => format!("{name}: p50 {p50:.0} ms, p95 {p95:.0} ms, max {max:.0} ms"),
        _ => format!("{name}: no frames"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = options()?;
    println!("connecting to {} ...", options.url);
    let client = Client::connect(&options.url, ClientOptions { token: options.token.clone(), ..ClientOptions::default() }).await?;
    let offset = client.clock_offset_ms();
    println!(
        "connected: rtt {}, clock offset {}",
        client.rtt_ms().map_or("unknown".to_owned(), |rtt| format!("{rtt:.0} ms")),
        offset.map_or("unknown (end-to-end latency will be guesswork)".to_owned(), |offset| format!("{offset:.0} ms")),
    );
    if let Some(encoding) = &options.encoding
        && !client.encodings().iter().any(|known| known.name == *encoding)
    {
        bail!("this gateway has no encoding {encoding:?} (it has: {})", client.encodings().iter().map(|known| known.name.clone()).collect::<Vec<_>>().join(", "));
    }

    // metadata: true so every frame carries its own receivedMs/writtenMs, which is where the latency comes from
    let subscribe = SubscribeOptions { encoding: options.encoding.clone(), channel: Some(options.channel.clone()), metadata: Some(true), ..Default::default() };
    let mut subscription = client.subscribe(&options.key, subscribe).await.with_context(|| format!("subscribing to {} on {}", options.key, options.channel))?;
    println!("subscribed to {} on {} for {:.0}s\n", options.key, options.channel, options.seconds);

    // a frame's track data and its metadata travel apart and either may come first, so they are joined on seq
    let mut frames: HashMap<u32, Frame> = HashMap::new();
    let (mut first_seq, mut last_seq) = (None::<u32>, None::<u32>);
    let mut decode_errors = 0usize;
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(options.seconds);
    let mut next_stats = started + Duration::from_secs(5);
    while Instant::now() < deadline {
        if Instant::now() >= next_stats {
            next_stats += Duration::from_secs(5);
            println!("[t+{:.0}s stats]\n{}", started.elapsed().as_secs_f64(), subscription_stats(&client, &options.key).await);
        }
        let Ok(Some(message)) = tokio::time::timeout_at(deadline.into(), subscription.recv()).await else { break };
        match message {
            Message::VideoInfo(info) => {
                let frame = frames.entry(info.seq).or_default();
                frame.received_ms = info.received_ms;
                frame.written_ms = info.written_ms;
                frame.keyframe = info.keyframe;
                frame.bytes = info.encoded_bytes;
                frame.size = (info.width, info.height);
                first_seq = Some(first_seq.map_or(info.seq, |first| first.min(info.seq)));
                last_seq = Some(last_seq.map_or(info.seq, |last| last.max(info.seq)));
                if options.verbose {
                    println!("seq {} {}x{} {} {} bytes", info.seq, info.width, info.height, if info.keyframe { "key" } else { "   " }, info.encoded_bytes);
                }
            }
            // the track's own frames: matched to metadata by RTP timestamp would be exact, but the metadata carries the
            // RTP timestamp only from 0.5.20 on, so the arrival of the newest unmatched frame is used
            Message::Video(video) => {
                if let Some(seq) = last_seq
                    && let Some(frame) = frames.get_mut(&seq)
                {
                    frame.arrived.get_or_insert(video.received_at);
                }
            }
            Message::Data(_) | Message::Audio(_) => decode_errors += 1,
        }
    }
    let elapsed = started.elapsed().as_secs_f64();

    let (Some(first_seq), Some(last_seq)) = (first_seq, last_seq) else {
        // the gateway's own account of the stream says far more than a guess from out here: whether samples arrived at
        // all, whether any were encoded, and the last error if there was one
        println!("\nno frames in {elapsed:.1}s. what the gateway says about this subscription:\n{}", subscription_stats(&client, &options.key).await);
        bail!("no frames in {elapsed:.1}s on {} (see the gateway's stats above)", options.channel);
    };
    let sent = (last_seq.wrapping_sub(first_seq) as usize) + 1;
    let arrived = frames.len();
    let keyframes: Vec<u32> = {
        let mut keyframes: Vec<u32> = frames.iter().filter(|(_, frame)| frame.keyframe).map(|(seq, _)| *seq).collect();
        keyframes.sort_unstable();
        keyframes
    };
    let mut gateway_ms: Vec<f64> = frames.values().filter_map(|frame| Some(frame.written_ms? - frame.received_ms?)).collect();
    let mut end_to_end_ms: Vec<f64> = match offset {
        // arrival on this clock, moved onto the gateway's, less when the gateway got the sample
        Some(offset) => frames
            .values()
            .filter_map(|frame| {
                let arrived = frame.arrived?;
                let arrived_ms = unix_ms_at(started, arrived);
                Some(arrived_ms + offset - frame.received_ms?)
            })
            .collect(),
        None => Vec::new(),
    };
    let bytes: u64 = frames.values().map(|frame| frame.bytes as u64).sum();
    let size = frames.values().map(|frame| frame.size).max_by_key(|(width, _)| *width).unwrap_or_default();

    println!("\n--- {} on {} over {elapsed:.1}s ---", options.key, options.channel);
    println!("picture:   {}x{}", size.0, size.1);
    println!("frames:    {arrived} arrived of {sent} sent = {:.1} fps", arrived as f64 / elapsed);
    println!("lost:      {} ({:.1}/s)", sent - arrived, (sent - arrived) as f64 / elapsed);
    println!("bitrate:   {:.2} Mbit/s", bytes as f64 * 8.0 / elapsed / 1e6);
    println!("{}", spread("latency (gateway held)", &mut gateway_ms));
    if end_to_end_ms.is_empty() {
        println!("latency (end to end): needs the clock offset, which this connection did not give");
    } else {
        println!("{}  [approximate: uses the measured clock offset]", spread("latency (end to end)", &mut end_to_end_ms));
    }
    match keyframes.len() {
        0 => println!("keyframes: none — on a passthrough channel nothing can start until one arrives"),
        1 => println!("keyframes: 1 in {elapsed:.1}s"),
        _ => {
            let mut spacing: Vec<f64> = keyframes.windows(2).map(|pair| pair[1].wrapping_sub(pair[0]) as f64).collect();
            println!("keyframes: {} ({:.2}/s, every {:.0} frames) — a lost frame costs the picture until the next one", keyframes.len(), keyframes.len() as f64 / elapsed, percentile(&mut spacing, 0.5).unwrap_or_default());
        }
    }
    if decode_errors > 0 {
        println!("note:      {decode_errors} messages arrived that were not video — is {} really a video channel?", options.channel);
    }
    // the gateway's own counters whenever the stream is not healthy: fewer frames than it sent, or any at all lost
    if arrived < sent || arrived as f64 / elapsed < 1.0 {
        println!("\nthe stream is not healthy; what the gateway says:\n{}", subscription_stats(&client, &options.key).await);
    }
    Ok(())
}

/// The gateway's counters for the subscription on `key`, as `client.stats()` reports them.
///
/// Worth asking for by name: a probe out here can only say "no frames", while the gateway knows whether it received
/// samples, how many it encoded, what it last failed at and how often a stream decoder had to start over. A silent
/// `received=2337, encodes=0, lastEncodingError=null` says "the encode stage never ran", which is a different bug from
/// "the encoder is broken" and takes a different hour to find.
async fn subscription_stats(client: &Client, key: &str) -> String {
    let Ok(stats) = client.stats().await else { return "  (the gateway did not answer a stats request)".to_owned() };
    let Some(channels) = stats["channels"].as_array() else { return "  (no channels in the gateway's stats)".to_owned() };
    let mine = channels.iter().find(|channel| channel["key"].as_str() == Some(key));
    let Some(channel) = mine.or_else(|| channels.first()) else { return "  (the gateway has no subscriptions)".to_owned() };
    let stat = &channel["stats"];
    let field = |name: &str| stat[name].to_string();
    [
        format!("  samples received: {}   sent: {}   encoded: {}", field("received"), field("sent"), field("encodes")),
        format!("  dropped from the queue: {}   by age: {}   stream decoder restarts: {}", field("droppedQueue"), field("droppedAge"), field("streamResyncs")),
        format!("  encoding errors: {}   last: {}", field("encodingErrors"), field("lastEncodingError")),
        format!("  picture: {}x{}   decode {} ms   encode {} ms", field("videoWidth"), field("videoHeight"), field("decodeMs"), field("encodeMs")),
        format!("  the allocator granted: {} bytes/s at {} Hz", channel["allocation"]["budgetBytesPerSec"], channel["allocation"]["hz"]),
        format!("  cpuScaleCap: {}   quality: {}", stat["cpuScaleCap"], stat["quality"]),
        format!("  pacerWaits: {}   pacedMs: {}   droppedQueue: {}", stat["pacerWaits"], stat["pacedMs"], stat["droppedQueue"]),
        // what the send loop was away doing when a sample had to be dropped: a dropped sample costs a stream-decoded
        // stream every picture until the publisher's next keyframe, so this is the number that explains a low rate
        format!("  the send loop was away up to {} ms when a sample was dropped", stat["worstConsumerGapMs"]),
        // pictures decoded to keep the stream whole and not encoded: the frames a loop behind its publisher gives up
        // instead of giving up the stream until the next keyframe
        format!("  caught up (decoded, not encoded): {}", stat["catchUpSkips"]),
    ]
    .join("\n")
}

/// `at` as unix ms, from `started` being now-ish: the probe's own clock, to be moved onto the gateway's.
fn unix_ms_at(started: Instant, at: Instant) -> f64 {
    let now_unix_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |since| since.as_secs_f64() * 1000.0);
    now_unix_ms - started.elapsed().as_secs_f64() * 1000.0 + at.duration_since(started).as_secs_f64() * 1000.0
}
