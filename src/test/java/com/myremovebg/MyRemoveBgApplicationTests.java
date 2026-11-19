package com.myremovebg;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class MyRemoveBgApplicationTests {

    @Autowired
    private BackgroundRemovalService backgroundRemovalService;

    @Test
    void contextLoads() {
        assertNotNull(backgroundRemovalService);
    }

    @Test
    void graphicsModeRemovesEnclosedWhiteAndLightGrayFringe() throws Exception {
        BufferedImage input = new BufferedImage(96, 96, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = input.createGraphics();
        try {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, input.getWidth(), input.getHeight());

            graphics.setColor(new Color(225, 225, 225));
            graphics.fillRect(4, 4, 88, 88);

            graphics.setColor(Color.BLACK);
            graphics.setStroke(new BasicStroke(5));
            graphics.drawRect(16, 16, 64, 64);

            graphics.setColor(new Color(30, 90, 210));
            graphics.fillOval(40, 40, 16, 16);
        } finally {
            graphics.dispose();
        }

        ByteArrayOutputStream source = new ByteArrayOutputStream();
        assertTrue(ImageIO.write(input, "PNG", source));

        byte[] resultBytes = backgroundRemovalService.removeBackground(
                source.toByteArray(), "graphics");
        BufferedImage result = ImageIO.read(new ByteArrayInputStream(resultBytes));

        assertNotNull(result);
        assertEquals(0, (result.getRGB(8, 48) >>> 24) & 255,
                "Light gray outside the artwork must be transparent");
        assertEquals(0, (result.getRGB(24, 24) >>> 24) & 255,
                "Enclosed white areas must be transparent in Graphics mode");
        assertTrue(((result.getRGB(16, 48) >>> 24) & 255) > 200,
                "Dark outline must remain visible");
        assertTrue(((result.getRGB(48, 48) >>> 24) & 255) > 200,
                "Colored artwork must remain visible");
    }

    @Test
    void removesOuterWhiteCanvasAndKeepsEnclosedWhiteArtwork() throws Exception {
        BufferedImage input = new BufferedImage(96, 96, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = input.createGraphics();
        try {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, input.getWidth(), input.getHeight());

            // A closed dark frame simulates a white printed card.
            graphics.setColor(Color.BLACK);
            graphics.setStroke(new BasicStroke(5));
            graphics.drawRect(16, 16, 64, 64);

            graphics.setColor(new Color(190, 35, 35));
            graphics.fillOval(40, 40, 16, 16);
        } finally {
            graphics.dispose();
        }

        ByteArrayOutputStream source = new ByteArrayOutputStream();
        assertTrue(ImageIO.write(input, "PNG", source));

        byte[] resultBytes = backgroundRemovalService.removeBackground(source.toByteArray());
        BufferedImage result = ImageIO.read(new ByteArrayInputStream(resultBytes));

        assertNotNull(result, "The result must be a readable PNG");
        assertEquals(input.getWidth(), result.getWidth());
        assertEquals(input.getHeight(), result.getHeight());
        assertTrue(result.getColorModel().hasAlpha(), "The output must contain alpha");

        assertEquals(0, (result.getRGB(8, 48) >>> 24) & 255,
                "Near-white canvas connected to the image edge must be transparent");
        assertEquals(255, (result.getRGB(24, 24) >>> 24) & 255,
                "Large enclosed white artwork must remain opaque");
    }
}
