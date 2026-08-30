-- Mosaic every person the segmentation model finds, on the first video
-- track, or the one `track` names.
-- variables: source (input media path), size (mosaic block size, defaults to 16), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/yolo26/recipes/mosaic-people.sql -v source=street.mp4 -v dest=mosaic.mp4
COPY (
  SELECT ffrwd.mask_tools.mosaic_where(v, ffrwd.yolo26.segment_mask(v, 'person'), :size), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
