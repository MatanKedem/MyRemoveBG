# MyRemoveBG

## Local processing modes

The local engine accepts an optional `mode` multipart parameter:

- `auto` (default): runs both ONNX models and enables the flat white-artwork preservation pass when the image is detected as sparse artwork.
- `graphics`: explicitly enables the white-canvas artwork pass for logos, text, and illustrations. All near-white pixels are made transparent, including enclosed white areas inside borders, while non-white strokes and lettering are preserved. It also runs both ONNX models on overlapping 900px tiles (180px minimum overlap) for more source detail on images up to 4 megapixels. Edge tiles are letterboxed to preserve their aspect ratio, and weighted blending avoids uncovered seams. Larger images safely fall back to full-image inference to limit memory use.
- `photo`: uses the neural masks without the flat-artwork white-canvas pass.

The mode only affects the local provider. External providers continue to use their own processing pipelines.
