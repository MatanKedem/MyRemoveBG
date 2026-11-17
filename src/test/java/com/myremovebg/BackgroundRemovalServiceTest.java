package com.myremovebg;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BackgroundRemovalServiceTest {

    @Test
    void tileStartsCoverEveryPixelWithoutGaps() {
        int tileSize = 900;
        int overlap = 180;

        for (int length : new int[]{900, 901, 1532, 1600, 2000, 2500, 5000}) {
            int[] starts = BackgroundRemovalService.tileStarts(length, tileSize, overlap);

            assertTrue(starts.length > 0, "At least one tile is required");
            assertEquals(0, starts[0], "The first tile must start at zero");
            assertEquals(length - tileSize, starts[starts.length - 1],
                    "The last tile must reach the final pixel");

            int coveredUntil = 0;
            for (int start : starts) {
                assertTrue(start <= coveredUntil,
                        "Tile layout left an uncovered strip for length " + length);
                coveredUntil = Math.max(coveredUntil, start + tileSize);
            }
            assertTrue(coveredUntil >= length,
                    "Tiles must cover the full image for length " + length);
        }
    }

    @Test
    void tileStartsRemainOrderedAndOverlap() {
        int[] starts = BackgroundRemovalService.tileStarts(5000, 900, 180);

        for (int i = 1; i < starts.length; i++) {
            assertTrue(starts[i] > starts[i - 1]);
            assertTrue(starts[i] - starts[i - 1] <= 900 - 180,
                    "Adjacent tiles must retain at least the configured overlap");
        }
    }

    @Test
    void featherWeightsAreSmoothAndBounded() {
        assertEquals(0f, BackgroundRemovalService.featherWeight(0, 180), 0.0001f);
        assertEquals(1f, BackgroundRemovalService.featherWeight(180, 180), 0.0001f);
        assertEquals(0.5f, BackgroundRemovalService.featherWeight(90, 180), 0.0001f);
        assertTrue(BackgroundRemovalService.featherWeight(45, 180) > 0f);
        assertTrue(BackgroundRemovalService.featherWeight(45, 180) < 1f);
    }
}
