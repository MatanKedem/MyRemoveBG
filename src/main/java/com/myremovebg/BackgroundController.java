package com.myremovebg;

import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class BackgroundController {
    private final BackgroundRemovalService localService;
    private final ExternalBackgroundRemovalService externalService;

    public BackgroundController(
            BackgroundRemovalService localService,
            ExternalBackgroundRemovalService externalService) {
        this.localService = localService;
        this.externalService = externalService;
    }

    @PostMapping(
        value = "/api/remove-background",
        consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public ResponseEntity<byte[]> remove(
            @RequestParam("image") MultipartFile image,
            @RequestParam(defaultValue = "local") String provider,
            @RequestParam(defaultValue = "auto") String mode) throws Exception {

        if (image.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        byte[] input = image.getBytes();
        byte[] result = switch (provider.toLowerCase()) {
            case "local" -> localService.removeBackground(input, mode);
            case "removebg" -> externalService.removeWithRemoveBg(input);
            case "clipdrop" -> externalService.removeWithClipdrop(input);
            case "photoroom" -> externalService.removeWithPhotoRoom(input);
            default -> throw new IllegalArgumentException("Unknown provider: " + provider);
        };

        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "inline; filename=background-removed.png")
                .body(result);
    }

    @GetMapping("/api/providers")
    public Map<String, Object> providers() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("local", Map.of("configured", true, "label", "MyRemoveBG AI"));
        result.put("removebg", Map.of(
                "configured", externalService.isRemoveBgConfigured(),
                "label", "remove.bg API"));
        result.put("clipdrop", Map.of(
                "configured", externalService.isClipdropConfigured(),
                "label", "Clipdrop API"));
        result.put("photoroom", Map.of(
                "configured", externalService.isPhotoRoomConfigured(),
                "label", "PhotoRoom API"));
        return result;
    }
}
