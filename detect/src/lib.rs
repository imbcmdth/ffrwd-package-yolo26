//! Object detection: every frame passes through untouched, with one row per
//! object beside it - the class as COCO label text, the confidence, and the
//! box in the frame's own pixels.
//!
//! The graph is YOLO26n's end-to-end export, run through `wasi:nn`. The
//! model is NMS-free: what comes back is a short per-image box list, already
//! one row per object, so decoding is a threshold and a coordinate map and
//! nothing else. The module never opens a file - the host binds the graph to
//! a name with `-nn detect=<path>` and this module asks for that name and
//! nothing else.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:yolo26-detect/detect",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};
use yolo26_common::{class_name, frame_box, le_f32s, to_input, Letterbox, PixFmt, SIDE};

/// The name the host binds the graph to. `-nn detect=<path>`.
const MODEL: &str = "detect";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "images";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// One returned row: two box corners, a score, and a class index.
const ROW_CHANNELS: usize = 6;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25}},"additionalProperties":false}"#;

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number"},"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"required":["class","conf","x","y","w","h"],"additionalProperties":false}"#;

fn default_conf() -> f64 {
    0.25
}

#[derive(Clone, Copy, Debug, Deserialize)]
// The schema says this one and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "default_conf")]
    conf: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            conf: default_conf(),
        }
    }
}

/// One row per detection, the box in the frame's own pixels.
#[derive(Serialize)]
struct Row {
    class: String,
    conf: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// One object out of the graph's box list, still on the frame's own axes.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Found {
    class: usize,
    conf: f32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    letterbox: Letterbox,
    params: Params,
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

fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("detect cannot read its params: {e}"))?
    };
    if !parsed.conf.is_finite() || !(0.0..=1.0).contains(&parsed.conf) {
        return Err(format!(
            "detect needs conf between 0 and 1, got {}",
            parsed.conf
        ));
    }
    Ok(parsed)
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
    format!("detect: {what}: {code} ({})", error.data())
}

/// Which returned tensor is the box list, by shape: rank 3 with six channels
/// per row. Names are not read, so an export that spells them differently
/// still resolves. A dense grid - channels first, thousands of anchors - is
/// refused by name: that is the export this module does not decode.
fn rows_tensor(shapes: &[Vec<u32>]) -> Result<usize, String> {
    let found = shapes.iter().position(
        |dimensions| matches!(dimensions.as_slice(), [_, _, channels] if *channels as usize == ROW_CHANNELS),
    );
    found.ok_or_else(|| {
        format!(
            "detect: the graph returned {shapes:?}, and this module wants the \
             end-to-end export's [1, rows, {ROW_CHANNELS}] box list"
        )
    })
}

/// The box list thresholded and brought back onto the frame. Each row is
/// `x1, y1, x2, y2, score, class` in the square's own pixels; the list is
/// padded, and the padding scores zero, so the threshold drops it.
fn decode(
    values: &[f32],
    conf: f32,
    letterbox: Letterbox,
    width: usize,
    height: usize,
) -> Vec<Found> {
    let mut found = Vec::new();
    for row in values.chunks_exact(ROW_CHANNELS) {
        let score = row[4];
        if score < conf {
            continue;
        }
        let (x0, y0, x1, y1) = frame_box((row[0], row[1], row[2], row[3]), letterbox, width, height);
        if x0 == x1 || y0 == y1 {
            continue;
        }
        found.push(Found {
            class: row[5].max(0.0) as usize,
            conf: score,
            x: x0 as u32,
            y: y0 as u32,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        });
    }
    found
}

/// One detection's row, as the NDJSON line that rides its frame.
fn to_row(found: &Found) -> String {
    serde_json::to_string(&Row {
        class: class_name(found.class),
        // To four places: the graph's own precision is nowhere near the
        // sixteen digits an f32 widened to an f64 prints.
        conf: (f64::from(found.conf) * 10_000.0).round() / 10_000.0,
        x: found.x,
        y: found.y,
        w: found.w,
        h: found.h,
    })
    .expect("row serializes")
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

/// One frame in, its rows out.
fn run(opened: &Opened, frame: &[u8]) -> Result<Vec<String>, String> {
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
    let rows = rows_tensor(&shapes)?;
    let found = decode(
        &le_f32s(&tensors[rows].data()),
        opened.params.conf as f32,
        opened.letterbox,
        opened.width,
        opened.height,
    );
    Ok(found.iter().map(to_row).collect())
}

struct Detect;

impl Guest for Detect {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "detect".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
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
            // The rows leaving are this module's own detections.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("detect reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = PixFmt::parse(&video.pix_fmt, "detect")?;
        let parsed = parse_params(&params)?;

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
                params: parsed,
                input_name: Cell::new(INPUT_NAME),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.params = parsed;
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
                match run(opened, &frame.frame) {
                    Ok(rows) => out.push(OutFrame {
                        pts: frame.pts,
                        // The picture leaves untouched; the rows are the work.
                        frame: FramePayload::Same,
                        rows,
                    }),
                    // `process` has no way to say no, so a graph that failed
                    // mid-stream stops the run rather than dropping detections
                    // on the floor.
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

export!(Detect);

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows the real export returned for one 720x576 frame, captured from a
    /// run of the pinned graph: `x1, y1, x2, y2, score, class` in the
    /// square's own pixels, three people, a chair, and a marginal fifth
    /// person. The fixture is the decode's contract, not the model's.
    const CAPTURED: [[f32; 6]; 5] = [
        [79.0, 141.0, 585.0, 573.0, 0.908, 0.0],
        [312.0, 121.0, 426.0, 492.0, 0.860, 0.0],
        [447.0, 246.0, 635.0, 571.0, 0.739, 0.0],
        [410.0, 290.0, 578.0, 566.0, 0.336, 56.0],
        [494.0, 217.0, 542.0, 269.0, 0.297, 0.0],
    ];

    fn captured() -> Vec<f32> {
        CAPTURED.iter().flatten().copied().collect()
    }

    #[test]
    fn the_captured_tensor_decodes_to_one_row_per_object() {
        // 720x576 fills the square's width; the bars are 64 rows top and
        // bottom, and every box comes back through them.
        let letterbox = Letterbox::new(720, 576);
        let found = decode(&captured(), 0.25, letterbox, 720, 576);
        assert_eq!(found.len(), 5, "every captured row clears 0.25");
        assert_eq!(found[0].class, 0, "the surest object is a person");
        assert_eq!(found[3].class, 56, "and the fourth is the chair");
        for object in &found {
            assert!(object.x + object.w <= 720, "boxes stay on the picture");
            assert!(object.y + object.h <= 576);
        }
        // The square's y=141 is the frame's (141-64)/0.888..: the bars are
        // subtracted before the scale comes off.
        assert_eq!(found[0].y, 86);
    }

    #[test]
    fn the_threshold_drops_what_scores_under_it() {
        let letterbox = Letterbox::new(720, 576);
        let found = decode(&captured(), 0.5, letterbox, 720, 576);
        assert_eq!(found.len(), 3, "the chair and the marginal person go");
        assert!(found.iter().all(|object| object.conf >= 0.5));
    }

    #[test]
    fn padding_rows_score_zero_and_never_decode() {
        // The export pads its list to a fixed length; the pad rows are all
        // zeros, and a zero score is under every threshold.
        let mut values = captured();
        values.extend([0.0; ROW_CHANNELS]);
        values.extend([0.0; ROW_CHANNELS]);
        let letterbox = Letterbox::new(720, 576);
        assert_eq!(decode(&values, 0.25, letterbox, 720, 576).len(), 5);
    }

    #[test]
    fn a_box_entirely_in_the_letterbox_padding_is_dropped() {
        // A box inside the top bar names no pixel of the picture.
        let values = [100.0, 10.0, 200.0, 50.0, 0.9, 0.0];
        let letterbox = Letterbox::new(720, 576);
        assert!(decode(&values, 0.25, letterbox, 720, 576).is_empty());
    }

    #[test]
    fn a_row_spells_the_class_as_text_and_rounds_the_confidence() {
        assert_eq!(
            to_row(&Found {
                class: 0,
                conf: 0.90785,
                x: 89,
                y: 86,
                w: 569,
                h: 486,
            }),
            r#"{"class":"person","conf":0.9079,"x":89,"y":86,"w":569,"h":486}"#
        );
    }

    #[test]
    fn the_box_list_is_found_by_shape_and_a_dense_grid_is_refused() {
        assert_eq!(
            rows_tensor(&[vec![1, 300, 6]]).expect("the end-to-end list"),
            0
        );
        let error = rows_tensor(&[vec![1, 84, 8400]]).expect_err("a dense grid");
        assert!(error.starts_with("detect: "), "{error}");
    }

    #[test]
    fn params_default_to_the_threshold_the_schema_publishes() {
        let parsed = parse_params("").expect("empty is the defaults");
        assert_eq!(parsed.conf, 0.25);
        let braces = parse_params("{}").expect("and so is an empty object");
        assert_eq!(braces.conf, 0.25);
    }

    #[test]
    fn params_outside_zero_to_one_are_refused_by_name() {
        for bad in [r#"{"conf":1.5}"#, r#"{"conf":-0.1}"#] {
            let error = parse_params(bad).expect_err(bad);
            assert!(error.starts_with("detect "), "{error}");
        }
        assert!(
            parse_params(r#"{"radius":3}"#).is_err(),
            "and so is a param this module has none of"
        );
    }
}
