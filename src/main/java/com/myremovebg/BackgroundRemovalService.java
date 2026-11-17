package com.myremovebg;

import ai.onnxruntime.*;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Arrays;

@Service
public class BackgroundRemovalService {

    private static final int U2_SIZE = 320;
    private static final int BIREF_SIZE = 384;

    private OrtEnvironment env;
    private OrtSession u2Session;
    private OrtSession birefSession;

    private String u2Input, u2Output;
    private String birefInput, birefOutput;

    @PostConstruct
    public void init() throws Exception {
        env = OrtEnvironment.getEnvironment();

        u2Session = env.createSession("models/u2netp.onnx");
        birefSession = env.createSession("models/birefnet-lite.onnx");

        u2Input = u2Session.getInputNames().iterator().next();
        u2Output = u2Session.getOutputNames().iterator().next();
        birefInput = birefSession.getInputNames().iterator().next();
        birefOutput = birefSession.getOutputNames().iterator().next();

        System.out.println("U2NET AI READY");
        System.out.println("BIREFNET AI READY");

        System.out.println("FUSION ENGINE READY");
        System.out.println("GRAPHICS PRESERVATION ENGINE READY");
    }

    private float[][] runModel(
            OrtSession session,
            String inputName,
            String outputName,
            BufferedImage original,
            int size) throws Exception {

        BufferedImage resized = new BufferedImage(
                size, size, BufferedImage.TYPE_INT_RGB);

        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(original, 0, 0, size, size, null);
        } finally {
            g.dispose();
        }

        float[] input = new float[3 * size * size];
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int rgb = resized.getRGB(x, y);
                float[] channels = {
                    ((rgb >> 16) & 255) / 255f,
                    ((rgb >> 8) & 255) / 255f,
                    (rgb & 255) / 255f
                };

                for (int c = 0; c < 3; c++) {
                    input[c * size * size + y * size + x] =
                            (channels[c] - mean[c]) / std[c];
                }
            }
        }

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(input),
                    new long[]{1, 3, size, size});
             OrtSession.Result result = session.run(
                    Collections.singletonMap(inputName, tensor))) {

            float[][][][] prediction = (float[][][][])
                    ((OnnxTensor) result.get(outputName)
                    .orElseThrow()).getValue();

            return prediction[0][0];
        }
    }

    /**
     * Runs inference on overlapping 900px tiles. This gives the fixed-size
     * ONNX models more source detail while feather-blending overlaps to avoid
     * visible seams. Used only for Graphics mode and capped to control memory.
     */
    private float[][] runModelTiled(
            OrtSession session,
            String inputName,
            String outputName,
            BufferedImage original,
            int modelSize) throws Exception {

        final int tileSize = 900;
        final int overlap = 180;
        final int feather = 180;
        int width = original.getWidth();
        int height = original.getHeight();

        int[] xStarts = tileStarts(width, tileSize, overlap);
        int[] yStarts = tileStarts(height, tileSize, overlap);
        // Accumulate directly into the returned 2D mask to avoid a third
        // full-resolution float buffer on top of the weight map.
        float[][] result = new float[height][width];
        float[] totalWeight = new float[width * height];

        for (int y0 : yStarts) {
            int tileHeight = Math.min(tileSize, height - y0);
            for (int x0 : xStarts) {
                int tileWidth = Math.min(tileSize, width - x0);
                BufferedImage sourceTile = original.getSubimage(x0, y0, tileWidth, tileHeight);

                // Keep the source aspect ratio. Stretching narrow edge tiles to
                // a square distorts lettering and symbols near image boundaries.
                BufferedImage tile = new BufferedImage(
                        tileSize, tileSize, BufferedImage.TYPE_INT_RGB);
                Graphics2D tileGraphics = tile.createGraphics();
                int scaledWidth = Math.max(1, (int) Math.round(
                        Math.min(tileSize / (double) tileWidth,
                                 tileSize / (double) tileHeight) * tileWidth));
                int scaledHeight = Math.max(1, (int) Math.round(
                        Math.min(tileSize / (double) tileWidth,
                                 tileSize / (double) tileHeight) * tileHeight));
                int padLeft = (tileSize - scaledWidth) / 2;
                int padTop = (tileSize - scaledHeight) / 2;
                try {
                    tileGraphics.setColor(Color.WHITE);
                    tileGraphics.fillRect(0, 0, tileSize, tileSize);
                    tileGraphics.setRenderingHint(
                            RenderingHints.KEY_INTERPOLATION,
                            RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                    tileGraphics.drawImage(sourceTile, padLeft, padTop,
                            scaledWidth, scaledHeight, null);
                } finally {
                    tileGraphics.dispose();
                }

                float[][] prediction = runModel(
                        session, inputName, outputName, tile, modelSize);

                for (int ty = 0; ty < tileHeight; ty++) {
                    float ny = (padTop + (ty + 0.5f) * scaledHeight / tileHeight)
                            / tileSize;
                    float wy = 1f;
                    if (y0 > 0) {
                        wy = Math.min(wy, featherWeight(ty + 1, feather));
                    }
                    if (y0 + tileHeight < height) {
                        wy = Math.min(wy, featherWeight(tileHeight - ty, feather));
                    }

                    for (int tx = 0; tx < tileWidth; tx++) {
                        float nx = (padLeft + (tx + 0.5f) * scaledWidth / tileWidth)
                                / tileSize;
                        float wx = 1f;
                        if (x0 > 0) {
                            wx = Math.min(wx, featherWeight(tx + 1, feather));
                        }
                        if (x0 + tileWidth < width) {
                            wx = Math.min(wx, featherWeight(tileWidth - tx, feather));
                        }

                        float weight = Math.max(0.0001f, wx * wy);
                        int index = (y0 + ty) * width + x0 + tx;
                        result[y0 + ty][x0 + tx] += sample(prediction, nx, ny) * weight;
                        totalWeight[index] += weight;
                    }
                }
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                result[y][x] = totalWeight[index] > 0f
                        ? result[y][x] / totalWeight[index] : 0f;
            }
        }
        return result;
    }

    static int[] tileStarts(int length, int tileSize, int overlap) {
        if (length <= tileSize) return new int[]{0};
        int stride = tileSize - overlap;
        java.util.ArrayList<Integer> starts = new java.util.ArrayList<>();
        for (int start = 0; start + tileSize < length; start += stride) {
            starts.add(start);
        }
        int last = length - tileSize;
        // Always append the final tile instead of moving an earlier tile.
        // Replacing the previous start can leave an uncovered strip.
        if (starts.isEmpty() || starts.get(starts.size() - 1) != last) {
            starts.add(last);
        }
        return starts.stream().mapToInt(Integer::intValue).toArray();
    }

    static float featherWeight(int distance, int feather) {
        float t = Math.max(0f, Math.min(1f, distance / (float) feather));
        return t * t * (3f - 2f * t);
    }

    /**
     * Converts model output to probabilities without stretching each image's
     * values independently. BiRefNet exports may expose logits, while some
     * exports already include sigmoid; detect the range and handle both.
     */
    private float[][] toProbabilityMask(float[][] mask) {
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;

        for (float[] row : mask) {
            for (float value : row) {
                if (!Float.isFinite(value)) continue;
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }

        boolean logits = min < -0.0001f || max > 1.0001f;
        float[][] result = new float[mask.length][mask[0].length];

        for (int y = 0; y < mask.length; y++) {
            for (int x = 0; x < mask[y].length; x++) {
                float value = mask[y][x];
                if (!Float.isFinite(value)) {
                    result[y][x] = 0f;
                } else if (logits) {
                    // Numerically stable sigmoid.
                    if (value >= 0f) {
                        result[y][x] = 1f / (1f + (float) Math.exp(-Math.min(value, 80f)));
                    } else {
                        float exp = (float) Math.exp(Math.max(value, -80f));
                        result[y][x] = exp / (1f + exp);
                    }
                } else {
                    result[y][x] = clamp(value);
                }
            }
        }

        return result;
    }

    private float clamp(float value) {
        return Math.max(0f, Math.min(1f, value));
    }

    private float sample(float[][] mask, float x, float y) {
        float px = x * (mask[0].length - 1);
        float py = y * (mask.length - 1);

        int x0 = (int) px;
        int y0 = (int) py;
        int x1 = Math.min(x0 + 1, mask[0].length - 1);
        int y1 = Math.min(y0 + 1, mask.length - 1);

        float fx = px - x0;
        float fy = py - y0;

        float top = mask[y0][x0] * (1 - fx)
                + mask[y0][x1] * fx;
        float bottom = mask[y1][x0] * (1 - fx)
                + mask[y1][x1] * fx;

        return top * (1 - fy) + bottom * fy;
    }

    private boolean isNearWhiteBackgroundPixel(int rgb) {
        int alpha = (rgb >>> 24) & 255;
        int red = (rgb >>> 16) & 255;
        int green = (rgb >>> 8) & 255;
        int blue = rgb & 255;

        return alpha > 32
                && red >= 225 && green >= 225 && blue >= 225
                && (765 - red - green - blue) <= 75
                && (Math.max(red, Math.max(green, blue))
                    - Math.min(red, Math.min(green, blue))) <= 28;
    }

    /**
     * Limits the white-canvas cleanup to sparse, flat artwork.
     * This avoids color-keying ordinary photographs with textured subjects.
     */
    private boolean looksLikeFlatWhiteArtwork(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int step = Math.max(1, Math.max(width, height) / 160);
        int[] histogram = new int[512];
        int samples = 0;
        int whiteSamples = 0;
        int foregroundSamples = 0;
        int distinctColors = 0;

        for (int y = 0; y < height; y += step) {
            for (int x = 0; x < width; x += step) {
                samples++;
                int rgb = image.getRGB(x, y);

                if (isNearWhiteBackgroundPixel(rgb)) {
                    whiteSamples++;
                    continue;
                }

                if (((rgb >>> 24) & 255) <= 32) {
                    continue;
                }

                foregroundSamples++;
                int red = (rgb >>> 16) & 255;
                int green = (rgb >>> 8) & 255;
                int blue = rgb & 255;
                int bucket = ((red >> 5) << 6)
                        | ((green >> 5) << 3)
                        | (blue >> 5);

                if (histogram[bucket] == 0) {
                    distinctColors++;
                }
                histogram[bucket]++;
            }
        }

        if (samples == 0
                || whiteSamples / (double) samples < 0.65d
                || foregroundSamples / (double) samples < 0.005d
                || foregroundSamples / (double) samples > 0.35d
                || distinctColors > 64) {
            return false;
        }

        Arrays.sort(histogram);
        int topColors = 0;
        for (int i = histogram.length - 1;
             i >= Math.max(0, histogram.length - 8); i--) {
            topColors += histogram[i];
        }

        return topColors / (double) foregroundSamples >= 0.70d;
    }

    /**
     * Finds near-white pixels connected to the image borders and marks them
     * as background. Large enclosed white regions are kept opaque, which
     * preserves things such as the white paper inside a bordered card.
     *
     * State values:
     * 0 = not classified, 1 = exterior background, 2 = small enclosed region,
     * 3 = large enclosed white foreground,
     * 4 = non-white artwork foreground (preserved exactly).
     */
    private byte[] classifyWhiteBorderRegions(BufferedImage image, boolean forceGraphicsMode) {
        int width = image.getWidth();
        int height = image.getHeight();
        long pixelCountLong = (long) width * height;

        // Avoid allocating a large flood-fill queue for unusually huge images.
        if (width < 3 || height < 3 || pixelCountLong > 8_000_000L) {
            return null;
        }

        int pixelCount = (int) pixelCountLong;
        long borderCount = 2L * width + 2L * height - 4L;
        long whiteBorderCount = 0;

        for (int x = 0; x < width; x++) {
            if (isNearWhiteBackgroundPixel(image.getRGB(x, 0))) whiteBorderCount++;
            if (isNearWhiteBackgroundPixel(image.getRGB(x, height - 1))) whiteBorderCount++;
        }
        for (int y = 1; y < height - 1; y++) {
            if (isNearWhiteBackgroundPixel(image.getRGB(0, y))) whiteBorderCount++;
            if (isNearWhiteBackgroundPixel(image.getRGB(width - 1, y))) whiteBorderCount++;
        }

        // Only activate this cleanup on sparse, flat artwork with a white canvas.
        if (whiteBorderCount < borderCount * 0.65d
                || (!forceGraphicsMode && !looksLikeFlatWhiteArtwork(image))) {
            return null;
        }

        byte[] state = new byte[pixelCount];
        int[] queue = new int[pixelCount];

        int head = 0;
        int tail = 0;

        // Seed the flood fill from all four edges.
        for (int x = 0; x < width; x++) {
            int top = x;
            int bottom = (height - 1) * width + x;
            if (state[top] == 0 && isNearWhiteBackgroundPixel(image.getRGB(x, 0))) {
                state[top] = 1;
                queue[tail++] = top;
            }
            if (state[bottom] == 0 && isNearWhiteBackgroundPixel(image.getRGB(x, height - 1))) {
                state[bottom] = 1;
                queue[tail++] = bottom;
            }
        }
        for (int y = 1; y < height - 1; y++) {
            int left = y * width;
            int right = left + width - 1;
            if (state[left] == 0 && isNearWhiteBackgroundPixel(image.getRGB(0, y))) {
                state[left] = 1;
                queue[tail++] = left;
            }
            if (state[right] == 0 && isNearWhiteBackgroundPixel(image.getRGB(width - 1, y))) {
                state[right] = 1;
                queue[tail++] = right;
            }
        }

        // Flood-fill the exterior near-white canvas (8-connected).
        while (head < tail) {
            int index = queue[head++];
            int x = index % width;
            int y = index / width;

            for (int dy = -1; dy <= 1; dy++) {
                int ny = y + dy;
                if (ny < 0 || ny >= height) continue;

                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;

                    int next = ny * width + nx;
                    if (state[next] == 0
                            && isNearWhiteBackgroundPixel(image.getRGB(nx, ny))) {
                        state[next] = 1;
                        queue[tail++] = next;
                    }
                }
            }
        }

        // On sparse flat artwork, every opaque non-white pixel is artwork.
        // Preserve these pixels directly instead of allowing a low-resolution
        // neural mask to erase thin lettering, arrows, and logo strokes.
        for (int index = 0; index < pixelCount; index++) {
            int x = index % width;
            int y = index / width;
            int rgb = image.getRGB(x, y);
            if (((rgb >>> 24) & 255) > 32
                    && !isNearWhiteBackgroundPixel(rgb)) {
                state[index] = 4;
            }
        }

        // Large enclosed white areas are part of the foreground (e.g. a white card).
        int minimumEnclosedArea = Math.max(250, pixelCount / 500);
        for (int start = 0; start < pixelCount; start++) {
            int sx = start % width;
            int sy = start / width;

            if (state[start] != 0
                    || !isNearWhiteBackgroundPixel(image.getRGB(sx, sy))) {
                continue;
            }

            head = 0;
            tail = 0;
            queue[tail++] = start;
            state[start] = 2;

            while (head < tail) {
                int index = queue[head++];
                int x = index % width;
                int y = index / width;

                for (int dy = -1; dy <= 1; dy++) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;

                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;

                        int next = ny * width + nx;
                        if (state[next] == 0
                                && isNearWhiteBackgroundPixel(image.getRGB(nx, ny))) {
                            state[next] = 2;
                            queue[tail++] = next;
                        }
                    }
                }
            }

            if (tail >= minimumEnclosedArea) {
                for (int i = 0; i < tail; i++) {
                    state[queue[i]] = 3;
                }
            }
        }

        return state;
    }

    public boolean isReady() {
        return u2Session != null && birefSession != null;
    }

    public byte[] removeBackground(byte[] bytes) throws Exception {
        return removeBackground(bytes, "auto");
    }

    public byte[] removeBackground(byte[] bytes, String mode) throws Exception {
        String processingMode = mode == null ? "auto" : mode.toLowerCase();
        if (!processingMode.equals("auto") && !processingMode.equals("graphics")
                && !processingMode.equals("photo")) {
            throw new IllegalArgumentException("Unknown local processing mode: " + processingMode);
        }
        BufferedImage original = ImageIO.read(new ByteArrayInputStream(bytes));
        if (original == null) {
            throw new IOException("Invalid image");
        }
        long pixels = (long) original.getWidth() * original.getHeight();
        if (pixels > 25_000_000L) {
            throw new IOException("Image is too large. Maximum supported size is 25 megapixels.");
        }

        // Three full-resolution float buffers are needed while blending tiles.
        // Keep this conservative so image processing does not exhaust the JVM heap.
        boolean tiledGraphics = processingMode.equals("graphics") && pixels <= 4_000_000L;
        System.out.println("Running U2NET" + (tiledGraphics ? " (overlapping tiles)..." : "..."));
        float[][] u2 = toProbabilityMask(tiledGraphics
                ? runModelTiled(u2Session, u2Input, u2Output, original, U2_SIZE)
                : runModel(u2Session, u2Input, u2Output, original, U2_SIZE));

        System.out.println("Running BiRefNet" + (tiledGraphics ? " (overlapping tiles)..." : "..."));
        float[][] biref = toProbabilityMask(tiledGraphics
                ? runModelTiled(birefSession, birefInput, birefOutput, original, BIREF_SIZE)
                : runModel(birefSession, birefInput, birefOutput, original, BIREF_SIZE));

        byte[] whiteRegions = processingMode.equals("photo")
                ? null : classifyWhiteBorderRegions(original, processingMode.equals("graphics"));
        if (tiledGraphics) {
            System.out.println("TILED GRAPHICS INFERENCE ENABLED (900px tiles, 180px overlap)");
        }
        if (whiteRegions != null) {
            System.out.println("WHITE BACKGROUND CLEANUP ENABLED (" + processingMode + ")");
        } else if (processingMode.equals("graphics")) {
            System.out.println("GRAPHICS MODE: image does not have a sufficiently white border; using AI mask");
        }

        int width = original.getWidth();
        int height = original.getHeight();
        BufferedImage output = new BufferedImage(
                width, height, BufferedImage.TYPE_INT_ARGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = width <= 1 ? 0 : x / (float) (width - 1);
                float ny = height <= 1 ? 0 : y / (float) (height - 1);

                float b = clamp(sample(biref, nx, ny));
                float u = clamp(sample(u2, nx, ny));

                // BiRefNet is the primary high-detail mask. U2NetP contributes
                // a small stabilizing vote without washing out fine transparent edges.
                float alpha = b * 0.90f + u * 0.10f;

                int rgb = original.getRGB(x, y);
                int sourceAlpha = (rgb >>> 24) & 255;
                int outputAlpha = Math.round(clamp(alpha) * sourceAlpha);

                if (whiteRegions != null) {
                    byte region = whiteRegions[y * width + x];
                    if (region == 1) {
                        outputAlpha = 0;
                    } else if (region == 3 || region == 4) {
                        outputAlpha = sourceAlpha;
                    }
                }

                output.setRGB(x, y,
                        (outputAlpha << 24) | (rgb & 0x00ffffff));
            }
        }

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        if (!ImageIO.write(output, "PNG", out)) {
            throw new IOException("PNG encoder is unavailable");
        }

        System.out.println("BACKGROUND REMOVAL COMPLETE");
        return out.toByteArray();
    }
}
