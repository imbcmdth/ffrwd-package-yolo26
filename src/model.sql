-- The two model exports, hosted as wasm modules the package ships. The
-- weights themselves are pinned in the manifest and land beside each module
-- at install.
--
-- `detect` returns the picture untouched with a row per object beside it -
-- the class as COCO label text, the confidence, and the box in the frame's
-- own pixels. `segment_mask` returns the found instances as one grayscale
-- matte, optionally narrowed to one class name, ready for maskedmerge and
-- everything else that reads a mask beside the picture.
CREATE FUNCTION detect(v video_stream, conf number DEFAULT 0.25)
RETURNS STRUCT(v video_stream, boxes STRUCT(class text, conf number,
                                            x number, y number, w number, h number)[])
  AS 'target/wasm32-wasip2/release/detect.wasm', 'detect' LANGUAGE wasm;

CREATE FUNCTION segment_mask(v video_stream, class text DEFAULT NULL,
                             conf number DEFAULT 0.25)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/segment_mask.wasm', 'segment_mask' LANGUAGE wasm;
