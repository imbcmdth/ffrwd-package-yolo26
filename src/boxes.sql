-- The row consumers: pure Rust modules, no model. Both take exactly the
-- record `detect` returns.
--
-- `boxes_mask` rasterizes the boxes into a grayscale matte - `grow` pads
-- each box outward in pixels, `feather` softens the edge over that many.
-- `draw_boxes` draws the boxes on the picture as green outlines.
CREATE FUNCTION boxes_mask(v video_stream,
                           boxes STRUCT(class text, conf number,
                                        x number, y number, w number, h number)[],
                           grow number DEFAULT 0, feather number DEFAULT 0)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/boxes_mask.wasm', 'boxes_mask' LANGUAGE wasm;

CREATE FUNCTION draw_boxes(v video_stream,
                           boxes STRUCT(class text, conf number,
                                        x number, y number, w number, h number)[],
                           thickness number DEFAULT 2)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/draw_boxes.wasm', 'draw_boxes' LANGUAGE wasm;
