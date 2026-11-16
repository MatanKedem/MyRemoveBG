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

@Service
public class BackgroundRemovalService {

    private static final int U2_SIZE = 320;
    private static final int BIREF_SIZE = 384;
    private static final int ISNET_SIZE = 1024;

    private OrtEnvironment env;
    private OrtSession u2Session;
    private OrtSession birefSession;
    private OrtSession isnetSession;

    private String u2Input, u2Output;
    private String birefInput, birefOutput;
    private String isnetInput;
    private String[] isnetOutputs;

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

        isnetSession = env.createSession("models/isnet.onnx");
        isnetInput = isnetSession.getInputNames().iterator().next();
        isnetOutputs = isnetSession.getOutputNames().toArray(new String[0]);

        System.out.println("ISNET AI READY");
        for (String name : isnetOutputs) {
            System.out.println("ISNET OUTPUT AVAILABLE: " + name);
        }

        System.out.println("FUSION ENGINE READY");
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

    private float[][] normalize(float[][] mask) {
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;

        for (float[] row : mask) {
            for (float value : row) {
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }

        float range = Math.max(max - min, 0.000001f);
        float[][] result = new float[mask.length][mask[0].length];

        for (int y = 0; y < mask.length; y++) {
            for (int x = 0; x < mask[y].length; x++) {
                result[y][x] = clamp((mask[y][x] - min) / range);
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
                && red >= 235 && green >= 235 && blue >= 235
                && (765 - red - green - blue) <= 45;
    }

    /**
     * Finds near-white pixels connected to the image borders and marks them
     * as background. Large enclosed white regions are kept opaque, which
     * preserves things such as the white paper inside a bordered card.
     *
     * State values:
     * 0 = not classified, 1 = exterior background, 2 = small enclosed region,
     * 3 = large enclosed white foreground.
     */
    private byte[] classifyWhiteBorderRegions(BufferedImage image) {
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

        // Only activate this cleanup when the image clearly has a white canvas.
        if (whiteBorderCount < borderCount * 0.65d) {
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

    private void diagnoseIsnet(BufferedImage original)
            throws Exception {

        BufferedImage resized = new BufferedImage(
                ISNET_SIZE, ISNET_SIZE, BufferedImage.TYPE_INT_RGB);

        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(original, 0, 0, ISNET_SIZE, ISNET_SIZE, null);
        } finally {
            g.dispose();
        }

        float[] input = new float[3 * ISNET_SIZE * ISNET_SIZE];

        for (int y = 0; y < ISNET_SIZE; y++) {
            for (int x = 0; x < ISNET_SIZE; x++) {
                int rgb = resized.getRGB(x, y);
                input[y * ISNET_SIZE + x] =
                        (((rgb >> 16) & 255) / 255f) * 2f - 1f;
                input[ISNET_SIZE * ISNET_SIZE + y * ISNET_SIZE + x] =
                        (((rgb >> 8) & 255) / 255f) * 2f - 1f;
                input[2 * ISNET_SIZE * ISNET_SIZE + y * ISNET_SIZE + x] =
                        ((rgb & 255) / 255f) * 2f - 1f;
            }
        }

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(input),
                    new long[]{1, 3, ISNET_SIZE, ISNET_SIZE});
             OrtSession.Result result = isnetSession.run(
                    Collections.singletonMap(isnetInput, tensor))) {

            for (String name : isnetOutputs) {
                Object value = result.get(name).orElseThrow();
                if (!(value instanceof OnnxTensor outputTensor)) continue;

                float[][][][] mask = (float[][][][]) outputTensor.getValue();
                if (mask.length == 0 || mask[0].length == 0) continue;

                float min = Float.MAX_VALUE;
                float max = -Float.MAX_VALUE;
                double sum = 0;
                long count = 0;

                for (float[] row : mask[0][0]) {
                    for (float item : row) {
                        min = Math.min(min, item);
                        max = Math.max(max, item);
                        sum += item;
                        count++;
                    }
                }

                if (count > 0) {
                    System.out.printf(
                            "ISNET OUTPUT %s | MIN: %.5f | MAX: %.5f | AVG: %.5f%n",
                            name, min, max, sum / count);
                }
            }
        }
    }

    public byte[] removeBackground(byte[] bytes) throws Exception {
        BufferedImage original = ImageIO.read(new ByteArrayInputStream(bytes));
        if (original == null) {
            throw new IOException("Invalid image");
        }

        System.out.println("Running ISNET diagnostics...");
        diagnoseIsnet(original);

        System.out.println("Running U2NET...");
        float[][] u2 = normalize(runModel(
                u2Session, u2Input, u2Output, original, U2_SIZE));

        System.out.println("Running BiRefNet...");
        float[][] biref = runModel(
                birefSession, birefInput, birefOutput, original, BIREF_SIZE);

        byte[] whiteRegions = classifyWhiteBorderRegions(original);
        if (whiteRegions != null) {
            System.out.println("WHITE BACKGROUND CLEANUP ENABLED");
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

                // Keep the existing model fusion unchanged; cleanup is a separate final step.
                float alpha = b * 0.70f + u * 0.30f;
                if (u > 0.65f && b > 0.20f) {
                    alpha = Math.max(alpha, Math.min(u, b + 0.12f));
                }

                int rgb = original.getRGB(x, y);
                int sourceAlpha = (rgb >>> 24) & 255;
                int outputAlpha = Math.round(clamp(alpha) * sourceAlpha);

                if (whiteRegions != null) {
                    byte region = whiteRegions[y * width + x];
                    if (region == 1) {
                        outputAlpha = 0;
                    } else if (region == 3) {
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
