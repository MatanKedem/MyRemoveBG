package com.myremovebg;

import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class BackgroundController {
    private final BackgroundRemovalService service;

    public BackgroundController(BackgroundRemovalService service) {
        this.service = service;
    }

    @PostMapping(
        value = "/api/remove-background",
        consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public ResponseEntity<byte[]> remove(
            @RequestParam("image") MultipartFile image) throws Exception {

        if (image.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        byte[] result = service.removeBackground(image.getBytes());

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=background-removed.png")
                .body(result);
    }
}
