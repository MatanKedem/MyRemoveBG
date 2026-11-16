# MyRemoveBG

## Local processing modes

The local engine accepts an optional `mode` multipart parameter:

- `auto` (default): runs both ONNX models and enables the flat white-artwork preservation pass when the image is detected as sparse artwork.
- `graphics`: explicitly enables the white-canvas artwork pass for logos, text, and illustrations. Use this for graphics on a mostly white background.
- `photo`: uses the neural masks without the flat-artwork white-canvas pass.

The mode only affects the local provider. External providers continue to use their own processing pipelines.
