//! Boxes on the picture: the annotation rows arriving with each frame are
//! drawn as green rectangle outlines, `thickness` pixels wide, growing
//! inward from each box's edge so a drawn box never spills past what the
//! detector reported. A row that is not a box - one an upstream module
//! emitted for something else - is skipped rather than refused.

wit_bindgen::generate!({
    path: "wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"thickness":{"type":"integer","minimum":1,"maximum":64,"default":2}},"additionalProperties":false}"#;

/// The outline's colour: green, as each format spells it. The yuv triple is
/// pure RGB green through the full-range BT.601 forward transform.
const GREEN_RGB: [u8; 3] = [0, 255, 0];
const GREEN_Y: u8 = 150;
const GREEN_U: u8 = 43;
const GREEN_V: u8 = 21;

fn default_thickness() -> u32 {
    2
}

#[derive(Clone, Copy, Deserialize)]
// The schema says this one and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "default_thickness")]
    thickness: u32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            thickness: default_thickness(),
        }
    }
}

/// One box to draw, as an upstream detector reports it. Extra keys are
/// ignored: a row carrying a class and a confidence beside the four
/// coordinates is still a box.
#[derive(Deserialize)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

/// A rectangle cut down to the frame: `x0..x1` by `y0..y1`, exclusive.
#[derive(Clone, Copy)]
struct Region {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// A rectangle inside the frame, or None when it falls outside it entirely.
fn clamp(rect: &Rect, width: usize, height: usize) -> Option<Region> {
    let w = width as i64;
    let h = height as i64;
    let x0 = rect.x.clamp(0, w);
    let y0 = rect.y.clamp(0, h);
    let x1 = rect.x.saturating_add(rect.w).clamp(0, w);
    let y1 = rect.y.saturating_add(rect.h).clamp(0, h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Region {
        x0: x0 as usize,
        y0: y0 as usize,
        x1: x1 as usize,
        y1: y1 as usize,
    })
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
            .map_err(|e| format!("draw_boxes cannot read its params: {e}"))?
    };
    if !(1..=64).contains(&parsed.thickness) {
        return Err(format!(
            "draw_boxes needs thickness between 1 and 64, got {}",
            parsed.thickness
        ));
    }
    Ok(parsed)
}

/// One filled band of the picture painted green, in whichever format the
/// instance was opened for. The band is already inside the frame.
fn fill(frame: &mut [u8], opened: &Opened, band: Region) {
    let (width, height) = (opened.width, opened.height);
    match opened.pix_fmt {
        PixFmt::Rgba => {
            for y in band.y0..band.y1 {
                let row = &mut frame[(y * width + band.x0) * 4..(y * width + band.x1) * 4];
                for pixel in row.as_chunks_mut::<4>().0 {
                    pixel[0] = GREEN_RGB[0];
                    pixel[1] = GREEN_RGB[1];
                    pixel[2] = GREEN_RGB[2];
                }
            }
        }
        PixFmt::Yuv420p => {
            for y in band.y0..band.y1 {
                frame[y * width + band.x0..y * width + band.x1].fill(GREEN_Y);
            }
            // Chroma is half resolution both ways; the band rounds outward so
            // a drawn edge never keeps the picture's own colour.
            let (cw, ch) = (width / 2, height / 2);
            let pixels = width * height;
            let cx0 = band.x0 / 2;
            let cx1 = (band.x1.div_ceil(2)).min(cw);
            let cy0 = band.y0 / 2;
            let cy1 = (band.y1.div_ceil(2)).min(ch);
            let (u, v) = frame[pixels..].split_at_mut(cw * ch);
            for cy in cy0..cy1 {
                u[cy * cw + cx0..cy * cw + cx1].fill(GREEN_U);
                v[cy * cw + cx0..cy * cw + cx1].fill(GREEN_V);
            }
        }
    }
}

/// One box's outline: four bands growing inward from its edges, so a
/// thickness wider than the box fills it and never spills.
fn outline(frame: &mut [u8], opened: &Opened, region: Region, thickness: usize) {
    let top = (region.y0 + thickness).min(region.y1);
    let bottom = region.y1.saturating_sub(thickness).max(top);
    for band in [
        // Top and bottom, full width.
        Region { y1: top, ..region },
        Region { y0: bottom, ..region },
        // Left and right, between them.
        Region {
            y0: top,
            y1: bottom,
            x1: (region.x0 + thickness).min(region.x1),
            ..region
        },
        Region {
            y0: top,
            y1: bottom,
            x0: region.x1.saturating_sub(thickness).max(region.x0),
            ..region
        },
    ] {
        if band.x0 < band.x1 && band.y0 < band.y1 {
            fill(frame, opened, band);
        }
    }
}

/// Every row's box drawn on one frame.
fn draw(frame: &mut [u8], opened: &Opened, rows: &[String]) -> bool {
    let mut drew = false;
    for row in rows {
        let Ok(rect) = serde_json::from_str::<Rect>(row) else {
            continue;
        };
        let Some(region) = clamp(&rect, opened.width, opened.height) else {
            continue;
        };
        outline(frame, opened, region, opened.params.thickness as usize);
        drew = true;
    }
    drew
}

struct DrawBoxes;

impl Guest for DrawBoxes {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "draw_boxes".to_string(),
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
            // The rows are the boxes this module draws.
            reads_rows: true,
            // And they are consumed by it: what leaves is the picture alone.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("draw_boxes reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("draw_boxes does not accept pixel format {other}")),
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
            .into_iter()
            .map(|frame| {
                let mut picture = frame.frame;
                let drew = draw(&mut picture, &opened, &frame.rows);
                OutFrame {
                    pts: frame.pts,
                    // A frame with nothing to draw passes through uncopied.
                    frame: if drew {
                        FramePayload::New(picture)
                    } else {
                        FramePayload::Same
                    },
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

export!(DrawBoxes);

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 32;
    const H: usize = 24;

    fn opened(pix_fmt: PixFmt, thickness: u32) -> Opened {
        Opened {
            width: W,
            height: H,
            pix_fmt,
            params: Params { thickness },
        }
    }

    fn grey_yuv() -> Vec<u8> {
        vec![100u8; W * H + 2 * (W / 2) * (H / 2)]
    }

    fn rows(items: &[&str]) -> Vec<String> {
        items.iter().map(|r| r.to_string()).collect()
    }

    #[test]
    fn an_outline_paints_the_edges_and_leaves_the_interior() {
        let mut frame = grey_yuv();
        let drew = draw(
            &mut frame,
            &opened(PixFmt::Yuv420p, 2),
            &rows(&[r#"{"class":"person","conf":0.9,"x":4,"y":4,"w":16,"h":12}"#]),
        );
        assert!(drew);
        assert_eq!(frame[4 * W + 10], GREEN_Y, "the top edge is drawn");
        assert_eq!(frame[5 * W + 10], GREEN_Y, "two rows thick");
        assert_eq!(frame[10 * W + 4], GREEN_Y, "the left edge is drawn");
        assert_eq!(frame[10 * W + 19], GREEN_Y, "and the right");
        assert_eq!(frame[10 * W + 10], 100, "the interior is untouched");
        assert_eq!(frame[2 * W + 10], 100, "and so is outside the box");
    }

    #[test]
    fn the_outline_colours_the_chroma_under_its_edges() {
        let mut frame = grey_yuv();
        draw(
            &mut frame,
            &opened(PixFmt::Yuv420p, 2),
            &rows(&[r#"{"x":4,"y":4,"w":16,"h":12}"#]),
        );
        let pixels = W * H;
        let (cw, ch) = (W / 2, H / 2);
        assert_eq!(frame[pixels + 2 * cw + 5], GREEN_U, "U under the top edge");
        assert_eq!(
            frame[pixels + cw * ch + 2 * cw + 5],
            GREEN_V,
            "V under the top edge"
        );
        assert_eq!(
            frame[pixels + 4 * cw + 5],
            100,
            "chroma inside the box is untouched"
        );
    }

    #[test]
    fn a_thickness_wider_than_the_box_fills_it_without_spilling() {
        let mut frame = grey_yuv();
        draw(
            &mut frame,
            &opened(PixFmt::Yuv420p, 8),
            &rows(&[r#"{"x":10,"y":10,"w":6,"h":6}"#]),
        );
        for y in 10..16 {
            for x in 10..16 {
                assert_eq!(frame[y * W + x], GREEN_Y, "({x},{y}) is filled");
            }
        }
        assert_eq!(frame[9 * W + 12], 100, "nothing above the box");
        assert_eq!(frame[12 * W + 16], 100, "nothing right of it");
    }

    #[test]
    fn a_box_running_off_the_frame_is_clipped_and_one_outside_is_skipped() {
        let mut frame = grey_yuv();
        let drew = draw(
            &mut frame,
            &opened(PixFmt::Yuv420p, 2),
            &rows(&[
                r#"{"x":-4,"y":-4,"w":10,"h":10}"#,
                r#"{"x":100,"y":100,"w":10,"h":10}"#,
            ]),
        );
        assert!(drew, "the clipped one still draws");
        assert_eq!(frame[0], GREEN_Y, "its visible corner is painted");
    }

    #[test]
    fn rgba_paints_green_and_keeps_alpha() {
        let mut frame = vec![100u8; W * H * 4];
        draw(
            &mut frame,
            &opened(PixFmt::Rgba, 1),
            &rows(&[r#"{"x":4,"y":4,"w":8,"h":8}"#]),
        );
        let at = (4 * W + 6) * 4;
        assert_eq!(&frame[at..at + 4], &[0, 255, 0, 100], "green, alpha kept");
        let inside = (8 * W + 8) * 4;
        assert_eq!(frame[inside], 100, "the interior is untouched");
    }

    #[test]
    fn rows_that_are_not_boxes_draw_nothing() {
        let mut frame = grey_yuv();
        let drew = draw(
            &mut frame,
            &opened(PixFmt::Yuv420p, 2),
            &rows(&[r#"{"shot":4}"#, "not json at all"]),
        );
        assert!(!drew, "and the frame passes through uncopied");
        assert!(frame[..W * H].iter().all(|v| *v == 100));
    }

    #[test]
    fn params_outside_the_schema_are_refused_by_name() {
        assert!(parse_params(r#"{"thickness":0}"#).is_err());
        assert!(parse_params(r#"{"thickness":65}"#).is_err());
        assert!(parse_params(r#"{"radius":3}"#).is_err());
        assert_eq!(parse_params("").expect("defaults").thickness, 2);
    }
}
