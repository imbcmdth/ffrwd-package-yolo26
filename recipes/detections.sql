-- Every detection as NDJSON, one line per object per frame - class,
-- confidence, and the box in pixels. No video is written.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), track (video track index, defaults to the first), dest (output path, a .ndjson file)
-- example: ffrwd compile -f packages/ffrwd/yolo26/recipes/detections.sql -v source=street.mp4 -v dest=objects.ndjson
COPY (
  SELECT ffrwd.yolo26.detect(v, :conf).boxes
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest'
