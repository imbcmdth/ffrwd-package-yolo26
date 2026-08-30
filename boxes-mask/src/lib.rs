//! Boxes to matte: the annotation rows arriving with each frame are
//! rasterized into a grayscale mask - 255 inside each box, 0 everywhere
//! else. `grow` pads every box outward in pixels; `feather` softens the edge
//! over that many pixels, falling linearly from the box's edge to nothing.
//!
//! The picture itself is never read - only its geometry matters - so the
//! matte can be composed against the original stream by whatever consumes a
//! mask beside it. A row that is not a box - one an upstream module emitted
//! for something else - is skipped rather than refused.

wit_bindgen::generate!({
    path: "wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"grow":{"type":"number","minimum":0,"maximum":4096,"default":0},"feather":{"type":"number","minimum":0,"maximum":4096,"default":0}},"additionalProperties":false}"#;

#[derive(Clone, Copy, Deserialize)]
// The schema says these two and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default)]
    grow: f64,
    #[serde(default)]
    feather: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            grow: 0.0,
            feather: 0.0,
        }
    }
}

/// One box to rasterize, as an upstream detector reports it. Extra keys are
/// ignored: a row carrying a class and a confidence beside the four
/// coordinates is still a box.
#[derive(Deserialize)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// The pixel format the host chose at `init`, fixed for the instance's life.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

/// What `init` settled.
#[derive(Clone, Copy)]
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    params: Params,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed)
            .map_err(|e| format!("boxes_mask cannot read its params: {e}"))?
    };
    for (name, value) in [("grow", parsed.grow), ("feather", parsed.feather)] {
        if !value.is_finite() || !(0.0..=4096.0).contains(&value) {
            return Err(format!(
                "boxes_mask needs {name} between 0 and 4096, got {value}"
            ));
        }
    }
    Ok(parsed)
}

/// The alpha of one axis at a pixel centre `p`: full inside `[edge0, edge1]`,
/// falling linearly to nothing `feather` pixels outside either edge.
fn axis_alpha(p: f64, edge0: f64, edge1: f64, feather: f64) -> f64 {
    let outside = (edge0 - p).max(p - edge1).max(0.0);
    if outside <= 0.0 {
        return 1.0;
    }
    if feather <= 0.0 {
        return 0.0;
    }
    (1.0 - outside / feather).max(0.0)
}

/// One box painted into the matte, combined by max so overlapping boxes keep
/// each other's coverage. The horizontal ramp is computed once per box; each
/// row then scales it by its own vertical alpha, one multiply and one store
/// along a contiguous run.
fn paint(map: &mut [u8], width: usize, height: usize, rect: &Rect, params: Params) {
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return;
    }
    let (grow, feather) = (params.grow, params.feather);
    let x0 = rect.x - grow;
    let x1 = rect.x + rect.w + grow;
    let y0 = rect.y - grow;
    let y1 = rect.y + rect.h + grow;

    let first_x = (x0 - feather).floor().max(0.0) as usize;
    let last_x = ((x1 + feather).ceil().min(width as f64) as usize).max(first_x);
    let first_y = (y0 - feather).floor().max(0.0) as usize;
    let last_y = ((y1 + feather).ceil().min(height as f64) as usize).max(first_y);
    if first_x == last_x || first_y == last_y {
        return;
    }

    // The horizontal ramp, in eight bits so the row loop stays integer.
    let ramp: Vec<u16> = (first_x..last_x)
        .map(|x| (axis_alpha(x as f64 + 0.5, x0, x1, feather) * 255.0).round() as u16)
        .collect();

    for y in first_y..last_y {
        let ay = (axis_alpha(y as f64 + 0.5, y0, y1, feather) * 255.0).round() as u16;
        if ay == 0 {
            continue;
        }
        let row = &mut map[y * width + first_x..y * width + last_x];
        for (slot, ax) in row.iter_mut().zip(&ramp) {
            let value = ((ax * ay + 127) / 255) as u8;
            if value > *slot {
                *slot = value;
            }
        }
    }
}

/// The matte the rows rasterize to, one byte a pixel.
fn rasterize(rows: &[String], width: usize, height: usize, params: Params) -> Vec<u8> {
    let mut map = vec![0u8; width * height];
    for row in rows {
        let Ok(rect) = serde_json::from_str::<Rect>(row) else {
            continue;
        };
        paint(&mut map, width, height, &rect, params);
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

struct BoxesMask;

impl Guest for BoxesMask {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "boxes_mask".to_string(),
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
            // The rows are the boxes this module rasterizes.
            reads_rows: true,
            // And they are consumed by it: what leaves is the matte alone.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("boxes_mask reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("boxes_mask does not accept pixel format {other}")),
        };
        let parsed = parse_params(&params)?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                params: parsed,
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
        let opened = OPENED
            .with(|o| *o.borrow())
            .expect("init settles the geometry before any frame arrives");

        let out = frames
            .iter()
            .map(|frame| {
                let map = rasterize(&frame.rows, opened.width, opened.height, opened.params);
                OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::New(to_frame(
                        &map,
                        opened.pix_fmt,
                        opened.width,
                        opened.height,
                        frame.frame.len(),
                    )),
                    // The boxes were the rows' whole purpose; none travel on.
                    rows: vec![],
                }
            })
            .collect();
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(BoxesMask);

#[cfg(test)]
mod tests {
    use super::*;

    /// What a matte pixel fully inside a box carries.
    const KEEP: u8 = 255;

    const W: usize = 64;
    const H: usize = 48;

    fn params(grow: f64, feather: f64) -> Params {
        Params { grow, feather }
    }

    fn rows(items: &[&str]) -> Vec<String> {
        items.iter().map(|r| r.to_string()).collect()
    }

    #[test]
    fn a_box_paints_hard_edges_when_nothing_feathers() {
        let map = rasterize(
            &rows(&[r#"{"class":"person","conf":0.9,"x":10,"y":10,"w":20,"h":10}"#]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert_eq!(map[15 * W + 15], KEEP, "inside the box");
        assert_eq!(map[15 * W + 10], KEEP, "the left edge is inside");
        assert_eq!(map[15 * W + 29], KEEP, "and so is the last column");
        assert_eq!(map[15 * W + 30], 0, "one past it is not");
        assert_eq!(map[9 * W + 15], 0, "above the box is background");
    }

    #[test]
    fn grow_pads_the_box_outward() {
        let map = rasterize(
            &rows(&[r#"{"x":10,"y":10,"w":20,"h":10}"#]),
            W,
            H,
            params(4.0, 0.0),
        );
        assert_eq!(map[15 * W + 7], KEEP, "four pixels left of the box");
        assert_eq!(map[7 * W + 15], KEEP, "and four above it");
        assert_eq!(map[15 * W + 5], 0, "five is past the growth");
    }

    #[test]
    fn feather_ramps_from_full_at_the_edge_to_nothing() {
        let map = rasterize(
            &rows(&[r#"{"x":16,"y":16,"w":16,"h":16}"#]),
            W,
            H,
            params(0.0, 8.0),
        );
        assert_eq!(map[20 * W + 20], KEEP, "inside stays full");
        let mid = map[20 * W + 12]; // four pixels outside a left edge at 16
        assert!(
            (100..=160).contains(&mid),
            "half way out is about half, got {mid}"
        );
        assert_eq!(map[20 * W + 4], 0, "past the feather is background");
        let corner = map[12 * W + 12]; // four outside on both axes
        assert!(
            corner < mid,
            "a corner takes both ramps: {corner} < {mid}"
        );
    }

    #[test]
    fn overlapping_boxes_keep_the_stronger_coverage() {
        let map = rasterize(
            &rows(&[
                r#"{"x":10,"y":10,"w":10,"h":10}"#,
                r#"{"x":15,"y":10,"w":10,"h":10}"#,
            ]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert_eq!(map[15 * W + 17], KEEP, "the overlap is full, not doubled");
        assert_eq!(map[15 * W + 12], KEEP);
        assert_eq!(map[15 * W + 23], KEEP);
    }

    #[test]
    fn a_box_running_off_the_frame_is_clipped_not_refused() {
        let map = rasterize(
            &rows(&[r#"{"x":-10,"y":-10,"w":30,"h":30}"#]),
            W,
            H,
            params(0.0, 4.0),
        );
        assert_eq!(map[0], KEEP, "the corner the box covers is painted");
        assert_eq!(map[25 * W + 25], 0, "past its clipped extent is not");
    }

    #[test]
    fn rows_that_are_not_boxes_are_skipped_rather_than_refused() {
        let map = rasterize(
            &rows(&[
                r#"{"shot":4}"#,
                "not json at all",
                r#"{"x":10,"y":10,"w":5,"h":5}"#,
            ]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert_eq!(map[12 * W + 12], KEEP);
    }

    #[test]
    fn no_rows_at_all_is_an_entirely_black_matte() {
        let map = rasterize(&[], W, H, params(8.0, 8.0));
        assert!(map.iter().all(|v| *v == 0));
    }

    #[test]
    fn a_degenerate_box_paints_nothing() {
        let map = rasterize(
            &rows(&[r#"{"x":10,"y":10,"w":0,"h":10}"#]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert!(map.iter().all(|v| *v == 0));
    }

    #[test]
    fn the_matte_frame_keeps_neutral_chroma_and_opaque_alpha() {
        let map = vec![KEEP; 4 * 4];
        let yuv = to_frame(&map, PixFmt::Yuv420p, 4, 4, 4 * 4 + 2 * 2 * 2);
        assert!(yuv[..16].iter().all(|v| *v == KEEP));
        assert!(yuv[16..].iter().all(|v| *v == 128));

        let rgba = to_frame(&map, PixFmt::Rgba, 4, 4, 4 * 4 * 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(*pixel, [KEEP, KEEP, KEEP, 255]);
        }
    }

    #[test]
    fn params_outside_the_schema_are_refused_by_name() {
        assert!(parse_params(r#"{"grow":-1}"#).is_err());
        assert!(parse_params(r#"{"feather":5000}"#).is_err());
        assert!(parse_params(r#"{"radius":3}"#).is_err());
        let parsed = parse_params("").expect("empty is the defaults");
        assert_eq!((parsed.grow, parsed.feather), (0.0, 0.0));
    }
}
