-- Darken everything but one class of object: the matte keeps its own
-- brightness and the rest of the picture dims.
-- variables: source (input media path), class (a COCO class name, e.g. person), dim (how much the rest darkens, 0 to 1, defaults to 0.6), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/yolo26/recipes/spotlight.sql -v source=street.mp4 -v class=person -v dest=spotlit.mp4
COPY (
  SELECT ffrwd.mask_tools.spotlight(v, ffrwd.yolo26.segment_mask(v, :'class'), :dim), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
