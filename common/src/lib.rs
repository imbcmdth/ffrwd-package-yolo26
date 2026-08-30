//! What the yolo26 modules share: the letterbox into the model's square, the
//! frame-to-tensor preprocessing, the COCO class names, and the box math that
//! brings a model coordinate back onto the frame. Nothing here touches
//! `wasi:nn` or the wit bindings, so every module compiles it in as plain
//! Rust and its tests run on the host.

/// The square the graphs are run at. The exports are static at this size.
pub const SIDE: usize = 640;

/// What the letterbox pads with. The model was trained and is evaluated
/// against this grey, and a marginal detection at the edge of the picture
/// turns on it.
pub const PAD: f32 = 114.0 / 255.0;

/// The classes the graphs were trained on, in their own order.
pub const COCO: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// The name of a class by the graph's own index. An export trained on
/// something else is named by number rather than guessed at.
pub fn class_name(index: usize) -> String {
    COCO.get(index)
        .map_or_else(|| index.to_string(), |name| (*name).to_string())
}

/// The graph's index for a class name, or None for a name it was not
/// trained on.
pub fn class_index(name: &str) -> Option<usize> {
    COCO.iter().position(|known| *known == name)
}

/// The pixel format an instance was opened for, fixed for its life.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PixFmt {
    Yuv420p,
    Rgba,
}

impl PixFmt {
    /// The format the host named, or an error naming what it was.
    pub fn parse(named: &str, module: &str) -> Result<PixFmt, String> {
        match named {
            "yuv420p" => Ok(PixFmt::Yuv420p),
            "rgba" => Ok(PixFmt::Rgba),
            other => Err(format!("{module} does not accept pixel format {other}")),
        }
    }
}

/// Where the frame sits inside the square the graph is run at, once it has
/// been scaled to fit with its shape kept.
#[derive(Clone, Copy)]
pub struct Letterbox {
    /// Square pixels per frame pixel.
    pub scale: f32,
    /// Where the scaled frame starts inside the square.
    pub offset_x: usize,
    pub offset_y: usize,
    /// How much of the square the scaled frame covers.
    pub width: usize,
    pub height: usize,
}

impl Letterbox {
    pub fn new(width: usize, height: usize) -> Letterbox {
        let scale = (SIDE as f32 / width as f32).min(SIDE as f32 / height as f32);
        let scaled_w = ((width as f32 * scale).round() as usize).clamp(1, SIDE);
        let scaled_h = ((height as f32 * scale).round() as usize).clamp(1, SIDE);
        Letterbox {
            scale,
            offset_x: (SIDE - scaled_w) / 2,
            offset_y: (SIDE - scaled_h) / 2,
            width: scaled_w,
            height: scaled_h,
        }
    }

    /// A square coordinate back on the frame's own horizontal axis.
    pub fn frame_x(self, square: f32) -> f32 {
        (square - self.offset_x as f32) / self.scale
    }

    /// A square coordinate back on the frame's own vertical axis.
    pub fn frame_y(self, square: f32) -> f32 {
        (square - self.offset_y as f32) / self.scale
    }
}

/// A box in the square's own coordinates, brought back onto the frame and
/// clipped to the picture: `(x0, y0, x1, y1)` in frame pixels, exclusive on
/// the right and bottom. Empty - `x0 == x1` or `y0 == y1` - when the box
/// falls entirely in the letterbox padding.
pub fn frame_box(
    square: (f32, f32, f32, f32),
    letterbox: Letterbox,
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    let clip = |value: f32, limit: usize| value.clamp(0.0, limit as f32) as usize;
    let x0 = clip(letterbox.frame_x(square.0).floor(), width);
    let x1 = clip(letterbox.frame_x(square.2).ceil(), width);
    let y0 = clip(letterbox.frame_y(square.1).floor(), height);
    let y1 = clip(letterbox.frame_y(square.3).ceil(), height);
    (x0, y0, x1.max(x0), y1.max(y0))
}

/// Where each of `count` output steps reads from along one axis: the two
/// samples it falls between, and how far along it sits. Every row of a resize
/// reads the same columns, so a column map is built once rather than per row.
pub struct Taps {
    pub low: Vec<usize>,
    pub high: Vec<usize>,
    pub fraction: Vec<f32>,
}

impl Taps {
    /// `at` gives the source coordinate of each step, on a source `extent`
    /// samples long.
    pub fn build(count: usize, extent: usize, at: impl Fn(usize) -> f32) -> Taps {
        let mut map = Taps {
            low: Vec::with_capacity(count),
            high: Vec::with_capacity(count),
            fraction: Vec::with_capacity(count),
        };
        for step in 0..count {
            let f = at(step).clamp(0.0, (extent - 1) as f32);
            let base = f.floor() as usize;
            map.low.push(base);
            map.high.push((base + 1).min(extent - 1));
            map.fraction.push(f - base as f32);
        }
        map
    }
}

/// One frame row as red, green and blue, a channel at a time so each is
/// contiguous.
fn row_to_rgb(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize, y: usize, out: &mut [f32]) {
    let (red, rest) = out.split_at_mut(width);
    let (green, blue) = rest.split_at_mut(width);
    match pix_fmt {
        PixFmt::Rgba => {
            for (x, pixel) in frame[y * width * 4..(y + 1) * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                red[x] = f32::from(pixel[0]);
                green[x] = f32::from(pixel[1]);
                blue[x] = f32::from(pixel[2]);
            }
        }
        PixFmt::Yuv420p => {
            let pixels = width * height;
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let chroma = cw * ch;
            let luma = &frame[y * width..(y + 1) * width];
            let crow = (y / 2).min(ch - 1) * cw;
            for x in 0..width {
                let l = f32::from(luma[x]);
                let ci = crow + (x / 2).min(cw - 1);
                let u = f32::from(frame[pixels + ci]) - 128.0;
                let v = f32::from(frame[pixels + chroma + ci]) - 128.0;
                // The usual BT.601 inverse, in full range: the frames a module
                // is handed are what the host decoded, not studio-swing video.
                red[x] = l + 1.402 * v;
                green[x] = l - 0.344_136 * u - 0.714_136 * v;
                blue[x] = l + 1.772 * u;
            }
        }
    }
}

/// One frame row resized to the square's columns, channel by channel.
fn resize_rgb_row(rgb: &[f32], columns: &Taps, width: usize, out: &mut [f32]) {
    let count = columns.low.len();
    for channel in 0..3 {
        let source = &rgb[channel * width..(channel + 1) * width];
        let target = &mut out[channel * count..(channel + 1) * count];
        for (((sample, low), high), fraction) in target
            .iter_mut()
            .zip(&columns.low)
            .zip(&columns.high)
            .zip(&columns.fraction)
        {
            let (a, b) = (source[*low], source[*high]);
            *sample = a + (b - a) * fraction;
        }
    }
}

/// The frame scaled into the square the graph takes and laid out as the planar
/// fp32 tensor it expects: red, green and blue in turn, each rescaled to 0..1.
///
/// The resize is separable, so each frame row is turned into the square's
/// columns once and the two square rows that read it mix the same numbers.
/// Slots go by parity, and a square row mixes frame rows `y` and `y + 1`,
/// which never share one.
pub fn to_input(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    height: usize,
    letterbox: Letterbox,
) -> Vec<u8> {
    let plane = SIDE * SIDE;
    let mut planes = vec![PAD; plane * 3];

    let columns = Taps::build(letterbox.width, width, |sx| {
        (sx as f32 + 0.5) / letterbox.scale - 0.5
    });
    let rows = Taps::build(letterbox.height, height, |sy| {
        (sy as f32 + 0.5) / letterbox.scale - 0.5
    });
    let mut rgb = vec![0f32; width * 3];
    let mut resized = [
        vec![0f32; letterbox.width * 3],
        vec![0f32; letterbox.width * 3],
    ];
    let mut held: [Option<usize>; 2] = [None, None];

    for sy in 0..letterbox.height {
        for y in [rows.low[sy], rows.high[sy]] {
            let slot = y % 2;
            if held[slot] != Some(y) {
                row_to_rgb(frame, pix_fmt, width, height, y, &mut rgb);
                resize_rgb_row(&rgb, &columns, width, &mut resized[slot]);
                held[slot] = Some(y);
            }
        }
        let (top_row, bottom_row) = (&resized[rows.low[sy] % 2], &resized[rows.high[sy] % 2]);
        let ty = rows.fraction[sy];

        let at = (letterbox.offset_y + sy) * SIDE + letterbox.offset_x;
        for channel in 0..3 {
            let top = &top_row[channel * letterbox.width..(channel + 1) * letterbox.width];
            let bottom = &bottom_row[channel * letterbox.width..(channel + 1) * letterbox.width];
            let target =
                &mut planes[channel * plane + at..channel * plane + at + letterbox.width];
            for ((sample, a), b) in target.iter_mut().zip(top).zip(bottom) {
                *sample = (a + (b - a) * ty).clamp(0.0, 255.0) / 255.0;
            }
        }
    }

    let mut bytes = vec![0u8; planes.len() * 4];
    let (words, _) = bytes.as_chunks_mut::<4>();
    for (word, value) in words.iter_mut().zip(&planes) {
        *word = value.to_le_bytes();
    }
    bytes
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
pub fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_frame_is_letterboxed_with_bars_above_and_below() {
        let letterbox = Letterbox::new(1280, 640);
        assert_eq!(letterbox.width, SIDE, "the wide side fills the square");
        assert_eq!(letterbox.height, SIDE / 2, "and the other is scaled to fit");
        assert_eq!(letterbox.offset_x, 0);
        assert_eq!(letterbox.offset_y, SIDE / 4, "the bars are above and below");
    }

    #[test]
    fn a_tall_frame_is_letterboxed_with_bars_left_and_right() {
        let letterbox = Letterbox::new(810, 1080);
        assert_eq!(letterbox.height, SIDE);
        assert_eq!(letterbox.width, 480);
        assert_eq!(letterbox.offset_x, 80, "80 columns of padding either side");
        assert_eq!(letterbox.offset_y, 0);
    }

    #[test]
    fn a_square_frame_fills_the_square() {
        let letterbox = Letterbox::new(256, 256);
        assert_eq!((letterbox.width, letterbox.height), (SIDE, SIDE));
        assert_eq!((letterbox.offset_x, letterbox.offset_y), (0, 0));
    }

    #[test]
    fn a_square_coordinate_comes_back_to_the_frame_it_came_from() {
        let letterbox = Letterbox::new(810, 1080);
        assert!(letterbox.frame_x(letterbox.offset_x as f32).abs() < 0.01);
        assert!(
            (letterbox.frame_x((letterbox.offset_x + letterbox.width) as f32) - 810.0).abs()
                < 0.01
        );
        assert!(letterbox.frame_y(0.0).abs() < 0.01);
        assert!((letterbox.frame_y(SIDE as f32) - 1080.0).abs() < 0.01);
    }

    #[test]
    fn a_square_box_lands_on_the_frame_clipped_to_the_picture() {
        // A 720x576 frame fills the square's width; the bars are above and
        // below, 32 square pixels each (576 * 640/720 = 512).
        let letterbox = Letterbox::new(720, 576);
        let (x0, y0, x1, y1) = frame_box((0.0, 64.0, 320.0, 576.0), letterbox, 720, 576);
        assert_eq!((x0, y0), (0, 0), "the top-left corner maps to the frame's");
        assert_eq!(x1, 360, "half the square is half the frame");
        assert_eq!(y1, 576, "and the square's last picture row is the frame's");
    }

    #[test]
    fn a_box_entirely_in_the_padding_is_empty() {
        let letterbox = Letterbox::new(720, 576);
        let (x0, y0, x1, y1) = frame_box((0.0, 0.0, 640.0, 30.0), letterbox, 720, 576);
        assert_eq!(y0, y1, "the top bar holds no picture");
        assert!(x0 <= x1);
    }

    #[test]
    fn a_class_is_named_by_the_graphs_own_index() {
        assert_eq!(class_name(0), "person");
        assert_eq!(class_name(5), "bus");
        assert_eq!(class_name(79), "toothbrush");
        assert_eq!(
            class_name(80),
            "80",
            "an export trained on something else is numbered, not guessed at"
        );
    }

    #[test]
    fn a_class_name_finds_the_index_it_names() {
        assert_eq!(class_index("person"), Some(0));
        assert_eq!(class_index("skateboard"), Some(36));
        assert_eq!(class_index("warp core"), None);
    }

    #[test]
    fn the_padding_of_a_letterboxed_input_is_the_grey_the_model_was_trained_with() {
        // A tall frame: the padding is the columns either side of it.
        let (width, height) = (32usize, 64usize);
        let letterbox = Letterbox::new(width, height);
        let frame = vec![0u8; width * height * 4];
        let bytes = to_input(&frame, PixFmt::Rgba, width, height, letterbox);
        let (words, _) = bytes.as_chunks::<4>();
        let value = |channel: usize, x: usize, y: usize| {
            f32::from_le_bytes(words[channel * SIDE * SIDE + y * SIDE + x])
        };
        assert!((value(0, 0, 0) - PAD).abs() < 1e-6, "the padding is grey");
        assert!(
            value(1, SIDE / 2, SIDE / 2).abs() < 1e-6,
            "and the black frame inside it is black"
        );
    }
}
