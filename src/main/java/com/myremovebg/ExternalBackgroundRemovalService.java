package com.myremovebg;

import org.springframework.core.env.Environment;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.UUID;

@Service
public class ExternalBackgroundRemovalService {

    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(20))
            .build();

    private final Environment environment;

    public ExternalBackgroundRemovalService(Environment environment) {
        this.environment = environment;
    }

    private String apiKey(String envName, String propertyName, String... aliases) {
        String value = System.getenv(envName);
        if (value != null && !value.isBlank()) return value.trim();
        value = environment.getProperty(envName);
        if (value != null && !value.isBlank()) return value.trim();
        value = environment.getProperty(propertyName);
        if (value != null && !value.isBlank()) return value.trim();
        for (String alias : aliases) {
            value = System.getenv(alias);
            if (value != null && !value.isBlank()) return value.trim();
            value = environment.getProperty(alias);
            if (value != null && !value.isBlank()) return value.trim();
        }
        return "";
    }

    private String removeBgApiKey() {
        return apiKey("REMOVEBG_API_KEY", "removebg.api-key", "REMOVE_BG_API_KEY", "REMOVEBG_KEY");
    }

    private String clipdropApiKey() {
        return apiKey("CLIPDROP_API_KEY", "clipdrop.api-key", "CLIPDROP_KEY");
    }

    private String photoRoomApiKey() {
        return apiKey("PHOTOROOM_API_KEY", "photoroom.api-key", "PHOTOROOM_KEY");
    }

    public boolean isRemoveBgConfigured() {
        return !removeBgApiKey().isBlank();
    }

    public boolean isClipdropConfigured() {
        return !clipdropApiKey().isBlank();
    }

    public boolean isPhotoRoomConfigured() {
        return !photoRoomApiKey().isBlank();
    }

    public byte[] removeWithRemoveBg(byte[] image) throws Exception {
        requireKey(isRemoveBgConfigured(), "remove.bg");
        String boundary = "----MyRemoveBG" + UUID.randomUUID();
        byte[] body = multipart(boundary,
                new Part("image_file", "upload.png", "image/png", image),
                new Part("size", null, "text/plain", "preview".getBytes(StandardCharsets.UTF_8)),
                new Part("format", null, "text/plain", "png".getBytes(StandardCharsets.UTF_8)));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.remove.bg/v1.0/removebg"))
                .timeout(Duration.ofSeconds(90))
                .header("X-Api-Key", removeBgApiKey())
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .header("Accept", "image/png")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();

        return sendImageRequest(request, "remove.bg");
    }

    public byte[] removeWithClipdrop(byte[] image) throws Exception {
        requireKey(isClipdropConfigured(), "Clipdrop");
        String boundary = "----MyRemoveBG" + UUID.randomUUID();
        byte[] body = multipart(boundary,
                new Part("image_file", "upload.png", "image/png", image));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://clipdrop-api.co/remove-background/v1"))
                .timeout(Duration.ofSeconds(90))
                .header("x-api-key", clipdropApiKey())
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .header("Accept", "image/png")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();

        return sendImageRequest(request, "Clipdrop");
    }

    public byte[] removeWithPhotoRoom(byte[] image) throws Exception {
        requireKey(isPhotoRoomConfigured(), "PhotoRoom");
        String boundary = "----MyRemoveBG" + UUID.randomUUID();
        byte[] body = multipart(boundary,
                new Part("image_file", "upload.png", "image/png", image),
                new Part("format", null, "text/plain", "png".getBytes(StandardCharsets.UTF_8)));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://sdk.photoroom.com/v1/segment"))
                .timeout(Duration.ofSeconds(90))
                .header("x-api-key", photoRoomApiKey())
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .header("Accept", "image/png")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();

        return sendImageRequest(request, "PhotoRoom");
    }

    private byte[] sendImageRequest(HttpRequest request, String provider) throws Exception {
        HttpResponse<byte[]> response = client.send(
                request, HttpResponse.BodyHandlers.ofByteArray());

        String contentType = response.headers()
                .firstValue("content-type").orElse("");

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            String detail = new String(response.body(), StandardCharsets.UTF_8);
            if (detail.length() > 500) detail = detail.substring(0, 500);
            throw new IOException(provider + " API returned HTTP "
                    + response.statusCode() + ": " + detail);
        }

        if (!contentType.toLowerCase().startsWith("image/")) {
            throw new IOException(provider + " returned an unexpected response type: "
                    + contentType);
        }

        return response.body();
    }

    private void requireKey(boolean configured, String provider) {
        if (!configured) {
            throw new IllegalStateException(provider
                    + " is not configured. Add its API key as a server environment variable.");
        }
    }

    private byte[] multipart(String boundary, Part... parts) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        for (Part part : parts) {
            write(out, "--" + boundary + "\r\n");
            String disposition = "Content-Disposition: form-data; name=\"" + part.name + "\"";
            if (part.filename != null) disposition += "; filename=\"" + part.filename + "\"";
            write(out, disposition + "\r\n");
            write(out, "Content-Type: " + part.contentType + "\r\n\r\n");
            out.write(part.data);
            write(out, "\r\n");
        }

        write(out, "--" + boundary + "--\r\n");
        return out.toByteArray();
    }

    private void write(ByteArrayOutputStream out, String value) throws IOException {
        out.write(value.getBytes(StandardCharsets.UTF_8));
    }

    private record Part(String name, String filename, String contentType, byte[] data) {}
}
