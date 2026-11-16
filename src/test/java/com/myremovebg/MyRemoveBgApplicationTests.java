package com.myremovebg;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import javax.imageio.ImageIO;
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
    void removesBackgroundAndReturnsTransparentPng() throws Exception {
        BufferedImage input = new BufferedImage(96, 96, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = input.createGraphics();
        try {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, input.getWidth(), input.getHeight());
            graphics.setColor(new Color(190, 35, 35));
            graphics.fillOval(24, 12, 48, 72);
        } finally {
            graphics.dispose();
        }

        ByteArrayOutputStream source = new ByteArrayOutputStream();
        assertTrue(ImageIO.write(input, "PNG", source));

        byte[] resultBytes = backgroundRemovalService.removeBackground(source.toByteArray());

        assertNotNull(resultBytes);
        assertTrue(resultBytes.length > 8, "The result must contain PNG data");

        BufferedImage result = ImageIO.read(new ByteArrayInputStream(resultBytes));
        assertNotNull(result, "The result must be a readable image");
        assertEquals(input.getWidth(), result.getWidth());
        assertEquals(input.getHeight(), result.getHeight());
        assertTrue(result.getColorModel().hasAlpha(), "The output must preserve an alpha channel");
    }
}
