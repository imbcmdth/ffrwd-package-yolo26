//! Instance segmentation as a matte: every frame leaves as a grayscale mask
//! of the instances the model found - 255 where an instance owns the pixel,
//! 0 everywhere else - optionally narrowed to one class name.
//!
//! The graph is YOLO26n-seg's end-to-end export, run through `wasi:nn`. The
//! model is NMS-free: what comes back is a short per-image box list, one row
//! per object with 32 mask coefficients beside it, plus the prototype planes
//! the coefficients weigh. A pixel is inside an instance where the weighted
//! prototype sum crosses zero - sigmoid rises with its argument and a half is
//! where it crosses zero, so no sigmoid is computed at all. The module never
//! opens a file - the host binds the graph to a name with
//! `-nn segment_mask=<path>` and this module asks for that name and nothing
//! else.
//!
//! The matte keeps the instance's own geometry and pixel format, so it feeds
//! straight into whatever reads a mask beside the picture: in yuv420p the
//! mask is the luma with neutral chroma, in rgba the same value in red, green
//! and blue, opaque.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:yolo26-segment/segment-mask",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};
use yolo26_common::{class_index, frame_box, le_f32s, to_input, Letterbox, PixFmt, Taps, SIDE};

/// The name the host binds the graph to. `-nn segment_mask=<path>`.
const MODEL: &str = "segment_mask";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "images";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// Mask coefficients per detection, and prototype planes to spend them on.
const COEFFICIENTS: usize = 32;

/// One returned row: two box corners, a score, a class index, and the
/// coefficients.
const ROW_CHANNELS: usize = 6 + COEFFICIENTS;

/// What a matte pixel inside an instance carries.
const KEEP: u8 = 255;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25}},"additionalProperties":false}"#;

fn default_conf() -> f64 {
    0.25
}

#[derive(Clone, Deserialize)]
// The schema says these two and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    /// One COCO class name to keep, or None for every class.
    #[serde(default)]
    class: Option<String>,
    #[serde(default = "default_conf")]
    conf: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            class: None,
            conf: default_conf(),
        }
    }
}

/// What the params settled once checked: the class as the graph's own index.
#[derive(Clone, Copy, Debug)]
struct Settled {
    class: Option<usize>,
    conf: f32,
}

/// One object out of the graph's box list, in the square's coordinates.
#[derive(Clone)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    coefficients: Vec<f32>,
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    letterbox: Letterbox,
    settled: Settled,
    /// What the graph calls its input, settled by the first call that works.
    input_name: Cell<&'static str>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

fn parse_params(params: &str) -> Result<Settled, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed)
            .map_err(|e| format!("segment_mask cannot read its params: {e}"))?
    };
    if !parsed.conf.is_finite() || !(0.0..=1.0).contains(&parsed.conf) {
        return Err(format!(
            "segment_mask needs conf between 0 and 1, got {}",
            parsed.conf
        ));
    }
    let class = match parsed.class.as_deref() {
        None | Some("") => None,
        Some(name) => Some(class_index(name).ok_or_else(|| {
            format!(
                "segment_mask does not know the class '{name}'; the model was \
                 trained on the 80 COCO classes, 'person' through 'toothbrush'"
            )
        })?),
    };
    Ok(Settled {
        class,
        conf: parsed.conf as f32,
    })
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("segment_mask: {what}: {code} ({})", error.data())
}

/// Which returned tensor is the box list and which the prototypes, by shape:
/// the prototypes are the rank-4 one carrying the 32 planes, and the box list
/// the rank-3 one with a row's channels last. Names are not read, so an
/// export that spells them differently still resolves. A dense grid -
/// channels first, thousands of anchors - is refused by name: that is the
/// export this module does not decode.
fn outputs(shapes: &[Vec<u32>]) -> Result<(usize, usize), String> {
    let rows = shapes.iter().position(
        |dimensions| matches!(dimensions.as_slice(), [_, _, channels] if *channels as usize == ROW_CHANNELS),
    );
    let prototypes = shapes.iter().position(|dimensions| {
        matches!(dimensions.as_slice(), [_, planes, _, _]
            if *planes as usize == COEFFICIENTS)
    });
    match (rows, prototypes) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(format!(
            "segment_mask: the graph returned {shapes:?}, and this module wants \
             the end-to-end export's [1, rows, {ROW_CHANNELS}] box list beside \
             [1, {COEFFICIENTS}, height, width]"
        )),
    }
}

/// The box list thresholded and narrowed to the class the params keep. Each
/// row is `x1, y1, x2, y2, score, class`, then the coefficients; the list is
/// padded, and the padding scores zero, so the threshold drops it.
fn decode(values: &[f32], settled: Settled) -> Vec<Detection> {
    let mut found = Vec::new();
    for row in values.chunks_exact(ROW_CHANNELS) {
        if row[4] < settled.conf {
            continue;
        }
        if let Some(wanted) = settled.class {
            if row[5].max(0.0) as usize != wanted {
                continue;
            }
        }
        found.push(Detection {
            x1: row[0],
            y1: row[1],
            x2: row[2],
            y2: row[3],
            coefficients: row[6..ROW_CHANNELS].to_vec(),
        });
    }
    found
}

/// One prototype row weighted by a detection's coefficients. Each plane
/// contributes one contiguous run, so the 32-term reduction is 32 walks of the
/// row rather than a gather per pixel.
fn weigh_row(
    prototypes: &[f32],
    plane: usize,
    width: usize,
    coefficients: &[f32],
    row: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    for (index, weight) in coefficients.iter().enumerate() {
        let start = index * plane + row * width;
        let source = &prototypes[start..start + width];
        for (slot, value) in out.iter_mut().zip(source) {
            *slot += weight * value;
        }
    }
}

/// One row of weighted prototypes resized to a run of frame columns.
fn resize_row(source: &[f32], columns: &Taps, out: &mut [f32]) {
    for (((sample, low), high), fraction) in out
        .iter_mut()
        .zip(&columns.low)
        .zip(&columns.high)
        .zip(&columns.fraction)
    {
        let (a, b) = (source[*low], source[*high]);
        *sample = a + (b - a) * fraction;
    }
}

/// The combined matte: `KEEP` where any surviving instance owns the pixel.
///
/// A mask is `sigmoid(coefficients . prototypes)` thresholded at a half,
/// clipped to its own box. Sigmoid rises with its argument and a half is
/// where it crosses zero, so the weighted sum is compared against zero
/// directly. The resize is separable: a prototype row is weighted and
/// stretched to the box's columns once, and the two frame rows reading it mix
/// the same numbers, which leaves the per-pixel work one mix, one compare and
/// one store along a contiguous run.
fn matte(
    detections: &[Detection],
    prototypes: &[f32],
    proto: (usize, usize),
    frame: (usize, usize),
    letterbox: Letterbox,
) -> Vec<u8> {
    let (proto_w, proto_h) = proto;
    let (width, height) = frame;
    let plane = proto_w * proto_h;

    let mut map = vec![0u8; width * height];
    let mut weighted = vec![0f32; proto_w];
    // Where a square coordinate lands on the prototype grid, each axis by its
    // own ratio because the grid is the square scaled down.
    let per_square_x = proto_w as f32 / SIDE as f32;
    let per_square_y = proto_h as f32 / SIDE as f32;

    for detection in detections {
        let (x0, y0, x1, y1) = frame_box(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            letterbox,
            width,
            height,
        );
        if x0 == x1 || y0 == y1 {
            continue;
        }
        let run = x1 - x0;

        let columns = Taps::build(run, proto_w, |step| {
            let square = letterbox.offset_x as f32 + (x0 + step) as f32 * letterbox.scale;
            (square + 0.5) * per_square_x - 0.5
        });
        let rows = Taps::build(y1 - y0, proto_h, |step| {
            let square = letterbox.offset_y as f32 + (y0 + step) as f32 * letterbox.scale;
            (square + 0.5) * per_square_y - 0.5
        });

        let mut resized = [vec![0f32; run], vec![0f32; run]];
        let mut held: [Option<usize>; 2] = [None, None];

        for step in 0..rows.low.len() {
            for row in [rows.low[step], rows.high[step]] {
                let slot = row % 2;
                if held[slot] != Some(row) {
                    weigh_row(
                        prototypes,
                        plane,
                        proto_w,
                        &detection.coefficients,
                        row,
                        &mut weighted,
                    );
                    resize_row(&weighted, &columns, &mut resized[slot]);
                    held[slot] = Some(row);
                }
            }
            let (top, bottom) = (&resized[rows.low[step] % 2], &resized[rows.high[step] % 2]);
            let ty = rows.fraction[step];

            let at = (y0 + step) * width + x0;
            let target = &mut map[at..at + run];
            for ((slot, a), b) in target.iter_mut().zip(top).zip(bottom) {
                let logit = a + (b - a) * ty;
                if logit > 0.0 {
                    *slot = KEEP;
                }
            }
        }
    }
    map
}

/// A matte written as a frame of the instance's own format: the luma plane
/// with neutral chroma, or the same value in red, green and blue.
fn to_frame(map: &[u8], pix_fmt: PixFmt, width: usize, height: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    match pix_fmt {
        PixFmt::Yuv420p => {
            out[..width * height].copy_from_slice(map);
            // 128 in both chroma planes is no colour at all.
            out[width * height..].fill(128);
        }
        PixFmt::Rgba => {
            for (pixel, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(map) {
                *pixel = [*value, *value, *value, 255];
            }
        }
    }
    out
}

/// One frame through the graph, however the graph names its input.
fn compute(opened: &Opened, input: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    let dimensions = [1, 3, SIDE as u32, SIDE as u32];
    let name = opened.input_name.get();
    let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
    match opened.context.compute(vec![(name.to_string(), tensor)]) {
        Ok(returned) => Ok(returned),
        // An export whose input is not called what this one calls it. The
        // host takes a position where it takes a name, so the retry names none,
        // and the name that worked is kept for every frame after this one.
        Err(_) if name == INPUT_NAME => {
            opened.input_name.set(INPUT_INDEX);
            let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
            opened
                .context
                .compute(vec![(INPUT_INDEX.to_string(), tensor)])
                .map_err(|e| failed("compute", &e))
        }
        Err(e) => Err(failed("compute", &e)),
    }
}

/// One frame in, its matte out.
fn run(opened: &Opened, frame: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let input = to_input(
        frame,
        opened.pix_fmt,
        opened.width,
        opened.height,
        opened.letterbox,
    );
    let returned = compute(opened, &input)?;
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let (rows, prototypes) = outputs(&shapes)?;

    let proto_h = shapes[prototypes][2] as usize;
    let proto_w = shapes[prototypes][3] as usize;

    let found = decode(&le_f32s(&tensors[rows].data()), opened.settled);
    let map = matte(
        &found,
        &le_f32s(&tensors[prototypes].data()),
        (proto_w, proto_h),
        (opened.width, opened.height),
        opened.letterbox,
    );
    Ok(to_frame(
        &map,
        opened.pix_fmt,
        opened.width,
        opened.height,
        len,
    ))
}

struct SegmentMask;

impl Guest for SegmentMask {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "segment_mask".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("segment_mask reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = PixFmt::parse(&video.pix_fmt, "segment_mask")?;
        let settled = parse_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first frame is what a provider picks its kernels on, and every
        // frame after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                letterbox: Letterbox::new(video.width as usize, video.height as usize),
                settled,
                input_name: Cell::new(INPUT_NAME),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let settled = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.settled = settled;
            }
        });
        Ok(())
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no frame
        // is ever left over.
        let mut out = Vec::with_capacity(frames.len());
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed
                .as_ref()
                .expect("init loads the graph before any frame arrives");
            for frame in &frames {
                match run(opened, &frame.frame, frame.frame.len()) {
                    Ok(map) => out.push(OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::New(map),
                        rows: vec![],
                    }),
                    // `process` has no way to say no, so a graph that failed
                    // mid-stream stops the run rather than passing a frame off
                    // as a matte.
                    Err(message) => panic!("{message}"),
                }
            }
        });
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(SegmentMask);

#[cfg(test)]
mod tests {
    use super::*;

    const PROTO: usize = 160;

    fn settled(class: Option<usize>, conf: f32) -> Settled {
        Settled { class, conf }
    }

    /// One row of the box list: a box, a score, a class, and coefficients
    /// that spend everything on the first prototype plane.
    fn row(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: f32, weight: f32) -> Vec<f32> {
        let mut values = vec![x1, y1, x2, y2, score, class];
        values.push(weight);
        values.extend([0.0; COEFFICIENTS - 1]);
        values
    }

    /// Prototypes whose first plane is all ones: a coefficient of one is then
    /// a positive logit everywhere, and a negative one a negative logit.
    fn flat_prototypes() -> Vec<f32> {
        let mut planes = vec![0f32; COEFFICIENTS * PROTO * PROTO];
        planes[..PROTO * PROTO].fill(1.0);
        planes
    }

    #[test]
    fn the_threshold_and_the_class_both_narrow_the_list() {
        // A person, a bus, and a padding row.
        let mut values = row(0.0, 0.0, 100.0, 100.0, 0.9, 0.0, 1.0);
        values.extend(row(200.0, 200.0, 300.0, 300.0, 0.6, 5.0, 1.0));
        values.extend(vec![0.0; ROW_CHANNELS]);

        assert_eq!(decode(&values, settled(None, 0.25)).len(), 2);
        assert_eq!(
            decode(&values, settled(Some(0), 0.25)).len(),
            1,
            "narrowed to person, the bus goes"
        );
        assert_eq!(
            decode(&values, settled(Some(5), 0.25)).len(),
            1,
            "narrowed to bus, the person goes"
        );
        assert_eq!(
            decode(&values, settled(None, 0.7)).len(),
            1,
            "the threshold drops the bus on its own"
        );
    }

    #[test]
    fn the_coefficients_ride_their_own_row() {
        let values = row(0.0, 0.0, 100.0, 100.0, 0.9, 0.0, 0.5);
        let found = decode(&values, settled(None, 0.25));
        assert_eq!(found[0].coefficients.len(), COEFFICIENTS);
        assert_eq!(found[0].coefficients[0], 0.5);
    }

    #[test]
    fn the_matte_paints_an_instance_over_its_own_box() {
        let found = decode(
            &row(100.0, 100.0, 300.0, 300.0, 0.9, 0.0, 1.0),
            settled(None, 0.25),
        );
        let map = matte(
            &found,
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert_eq!(map[200 * SIDE + 200], KEEP, "inside the box is the instance");
        assert_eq!(map[50 * SIDE + 50], 0, "outside it is background");
        assert_eq!(map[400 * SIDE + 400], 0);
    }

    #[test]
    fn a_negative_mask_paints_nothing_even_inside_its_box() {
        let found = decode(
            &row(100.0, 100.0, 300.0, 300.0, 0.9, 0.0, -1.0),
            settled(None, 0.25),
        );
        let map = matte(
            &found,
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert!(map.iter().all(|value| *value == 0));
    }

    #[test]
    fn two_instances_paint_one_combined_matte() {
        let mut values = row(100.0, 100.0, 200.0, 200.0, 0.9, 0.0, 1.0);
        values.extend(row(400.0, 400.0, 500.0, 500.0, 0.8, 0.0, 1.0));
        let found = decode(&values, settled(None, 0.25));
        let map = matte(
            &found,
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert_eq!(map[150 * SIDE + 150], KEEP);
        assert_eq!(map[450 * SIDE + 450], KEEP);
        assert_eq!(map[300 * SIDE + 300], 0, "between them is background");
    }

    #[test]
    fn a_matte_writes_neutral_chroma_and_opaque_alpha() {
        let map = vec![KEEP; 4 * 4];
        let yuv = to_frame(&map, PixFmt::Yuv420p, 4, 4, 4 * 4 + 2 * 2 * 2);
        assert!(yuv[..16].iter().all(|v| *v == KEEP), "the luma is the matte");
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&map, PixFmt::Rgba, 4, 4, 4 * 4 * 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(
                *pixel,
                [KEEP, KEEP, KEEP, 255],
                "equal in every channel, and opaque"
            );
        }
    }

    #[test]
    fn the_two_returned_tensors_are_told_apart_by_shape() {
        assert_eq!(
            outputs(&[vec![1, 300, 38], vec![1, 32, 160, 160]]).expect("both found"),
            (0, 1)
        );
        // The same two, in the other order.
        assert_eq!(
            outputs(&[vec![1, 32, 160, 160], vec![1, 300, 38]]).expect("both found"),
            (1, 0)
        );
    }

    #[test]
    fn a_dense_grid_is_refused_by_name() {
        let error =
            outputs(&[vec![1, 116, 8400], vec![1, 32, 160, 160]]).expect_err("not end-to-end");
        assert!(error.starts_with("segment_mask: "), "{error}");
    }

    #[test]
    fn params_default_to_every_class_at_the_published_threshold() {
        let parsed = parse_params("").expect("empty is the defaults");
        assert_eq!(parsed.class, None);
        assert_eq!(parsed.conf, 0.25);
        let named = parse_params(r#"{"class":"person"}"#).expect("a known class");
        assert_eq!(named.class, Some(0));
    }

    #[test]
    fn a_class_the_model_was_not_trained_on_is_refused_by_name() {
        let error = parse_params(r#"{"class":"warp core"}"#).expect_err("not a COCO class");
        assert!(error.contains("warp core"), "{error}");
        assert!(
            parse_params(r#"{"conf":1.5}"#).is_err(),
            "and so is a threshold outside 0..1"
        );
    }
}
