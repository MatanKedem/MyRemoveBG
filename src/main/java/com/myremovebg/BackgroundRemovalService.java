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

    private OrtEnvironment env;
    private OrtSession u2Session;
    private OrtSession birefSession;

    private String u2Input, u2Output;
    private String birefInput, birefOutput;

    private static final int ISNET_SIZE = 1024;
    private OrtSession isnetSession;
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

        isnetInput = isnetSession.getInputNames()
                .iterator().next();

        isnetOutputs = isnetSession.getOutputNames()
                .toArray(new String[0]);

        System.out.println("ISNET AI READY");

        for (String name : isnetOutputs) {
            System.out.println("ISNET MASK AVAILABLE: " + name);
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
        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, size, size, null);
        g.dispose();

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


    private void diagnoseIsnet(BufferedImage original)
            throws Exception {

        BufferedImage resized = new BufferedImage(
                ISNET_SIZE, ISNET_SIZE,
                BufferedImage.TYPE_INT_RGB);

        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, ISNET_SIZE, ISNET_SIZE, null);
        g.dispose();

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
                float[][][][] mask = (float[][][][])
                        ((OnnxTensor) result.get(name)
                        .orElseThrow()).getValue();

                float min = Float.MAX_VALUE;
                float max = -Float.MAX_VALUE;
                double sum = 0;
                long count = 0;

                for (float[] row : mask[0][0]) {
                    for (float value : row) {
                        min = Math.min(min, value);
                        max = Math.max(max, value);
                        sum += value;
                        count++;
                    }
                }

                System.out.printf(
                        "ISNET MASK %s | MIN: %.5f | MAX: %.5f | AVG: %.5f%n",
                        name, min, max, sum / count);
            }
        }
    }

    public byte[] removeBackground(byte[] bytes) throws Exception {

        BufferedImage original = ImageIO.read(
                new ByteArrayInputStream(bytes));

        if (original == null) {
            throw new IOException("Invalid image");
        }

        System.out.println("Running ISNET diagnostics...");
        diagnoseIsnet(original);

        System.out.println("Running U2NET...");
        float[][] u2 = normalize(runModel(
                u2Session, u2Input, u2Output,
                original, U2_SIZE));

        System.out.println("Running BiRefNet...");
        float[][] biref = runModel(
                birefSession, birefInput, birefOutput,
                original, BIREF_SIZE);

        BufferedImage output = new BufferedImage(
                original.getWidth(),
                original.getHeight(),
                BufferedImage.TYPE_INT_ARGB);

        for (int y = 0; y < original.getHeight(); y++) {
            for (int x = 0; x < original.getWidth(); x++) {

                float nx = original.getWidth() <= 1 ? 0
                        : x / (float)(original.getWidth() - 1);

                float ny = original.getHeight() <= 1 ? 0
                        : y / (float)(original.getHeight() - 1);

                float b = clamp(sample(biref, nx, ny));
                float u = clamp(sample(u2, nx, ny));

                // Weighted fusion: BiRefNet 70%, U2Net 30%
                float alpha = b * 0.70f + u * 0.30f;

                // Preserve details when both models detect the subject
                if (u > 0.65f && b > 0.20f) {
                    alpha = Math.max(
                            alpha,
                            Math.min(u, b + 0.12f));
                }

                alpha = clamp(alpha);

                int a = Math.round(alpha * 255);
                int rgb = original.getRGB(x, y);

                output.setRGB(x, y,
                        (a << 24) | (rgb & 0x00ffffff));
            }
        }

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(output, "PNG", out);

        System.out.println("FUSION COMPLETE");

        return out.toByteArray();
    }
}
