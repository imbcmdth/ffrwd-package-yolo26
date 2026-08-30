# ffrwd/yolo26

YOLO26 detection and segmentation, hosted in wasm. One model family,
one record shape, then composition: `detect` produces rows,
`segment_mask` produces a matte, the utilities turn rows into mattes
or drawn overlays, and everything downstream is native ffmpeg.

## License

This package is **AGPL-3.0**. The model weights are Ultralytics'
YOLO26, released under AGPL-3.0, and the package follows them. Using
the models over a network in a commercial product needs [Ultralytics'
commercial license](https://www.ultralytics.com/license); nothing here
changes that.

The weights are not in the archive: the manifest pins them - exact
repo, revision, file and sha256 - and `ffrwd install` fetches and
verifies them. Both are the medium end-to-end (NMS-free) ONNX exports
at 640x640, about 176 MB together, run through `wasi:nn` on the
machine's own ONNX Runtime.

## Model exports

- `detect(v, conf DEFAULT 0.25)` returns
  `STRUCT(v video_stream, boxes STRUCT(class text, conf number, x number, y number, w number, h number)[])` -
  the picture untouched, one row per object per frame, boxes in the
  frame's own pixels, classes as COCO label text.
- `segment_mask(v, class DEFAULT NULL, conf DEFAULT 0.25)` returns the
  found instances as one grayscale matte, optionally narrowed to one
  class name.

Narrow `detect`'s rows at run time with the gather spelling:

```sql
ARRAY(SELECT r FROM unnest(ffrwd.yolo26.detect(v).boxes) r
      WHERE r.class = 'person' AND r.conf >= 0.5)
```

## Utilities

- `boxes_mask(v, boxes, grow DEFAULT 0, feather DEFAULT 0)` - the rows
  rasterized into a matte; `grow` pads each box in pixels, `feather`
  softens the edge.
- `draw_boxes(v, boxes, thickness DEFAULT 2)` - the boxes drawn on the
  picture.

## Composition

The composition layer lives in `ffrwd/mask_tools` - `blur_where`,
`mosaic_where`, `spotlight`, `cutout` and the `masked` spelling they
share - because it is model-agnostic: any grayscale matte beside any
stream, all native ffmpeg. This package's recipes call it as
`ffrwd.mask_tools.blur_where(...)` and so on, feeding it the mattes
`segment_mask` and `boxes_mask` produce.

## Recipes

`blur-people`, `mosaic-people`, `spotlight`, `replace-background`,
`draw`, `detections` - run `ffrwd list` for each one's variables, or
read the header of the recipe file.

```
ffrwd ffrwd.yolo26.blur-people -v source=street.mp4 -v dest=blurred.mp4
```

## Building

The modules build against the wit from the installed `ffrwd/wasm`
package:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```

The weights are fetched at `ffrwd install` time by whoever installs
the published package; a development checkout runs the recipes only
after placing the pinned models beside the built modules.
