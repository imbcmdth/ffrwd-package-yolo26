-- Draw a box around everything the detector finds, on the first video
-- track, or the one `track` names.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/yolo26/recipes/draw.sql -v source=street.mp4 -v dest=boxed.mp4
COPY (
  SELECT ffrwd.yolo26.draw_boxes(ffrwd.yolo26.detect(v, :conf)), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
