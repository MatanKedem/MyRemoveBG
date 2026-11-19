package com.myremovebg;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Base64;

/**
 * Protects the whole web app with HTTP Basic authentication and applies a
 * conservative global quota to background-removal requests.
 *
 * Set MYREMOVEBG_ACCESS_PASSWORD in the server environment before starting
 * the app. The password is never stored in source control.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class AccessProtectionFilter extends OncePerRequestFilter {

    private static final String USERNAME = "myremovebg";
    private static final int MAX_PER_MINUTE = 5;
    private static final int MAX_PER_DAY = 30;

    private long minuteWindow = System.currentTimeMillis() / 60_000L;
    private long dayWindow = System.currentTimeMillis() / 86_400_000L;
    private int minuteCount;
    private int dayCount;

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        String configuredPassword = System.getenv("MYREMOVEBG_ACCESS_PASSWORD");
        if (configuredPassword == null || configuredPassword.length() < 16) {
            response.setStatus(HttpServletResponse.SC_SERVICE_UNAVAILABLE);
            response.setContentType("text/plain");
            response.setCharacterEncoding(StandardCharsets.UTF_8.name());
            response.getWriter().write(
                    "MyRemoveBG is locked: configure MYREMOVEBG_ACCESS_PASSWORD "
                    + "with at least 16 characters in the server environment.");
            return;
        }

        if (!isAuthorized(request, configuredPassword)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setHeader("WWW-Authenticate", "Basic realm=\"MyRemoveBG\", charset=\"UTF-8\"");
            response.setHeader("Cache-Control", "no-store");
            response.setContentType("text/plain");
            response.setCharacterEncoding(StandardCharsets.UTF_8.name());
            response.getWriter().write("Authentication required.");
            return;
        }

        if ("POST".equalsIgnoreCase(request.getMethod())
                && "/api/remove-background".equals(request.getRequestURI())
                && !allowProcessingRequest()) {
            response.setStatus(429);
            response.setHeader("Retry-After", "60");
            response.setHeader("Cache-Control", "no-store");
            response.setContentType("text/plain");
            response.setCharacterEncoding(StandardCharsets.UTF_8.name());
            response.getWriter().write(
                    "Usage limit reached. MyRemoveBG allows up to 5 requests per minute "
                    + "and 30 requests per day.");
            return;
        }

        filterChain.doFilter(request, response);
    }

    private boolean isAuthorized(HttpServletRequest request, String expectedPassword) {
        String header = request.getHeader("Authorization");
        if (header == null || !header.startsWith("Basic ")) {
            return false;
        }

        try {
            byte[] decoded = Base64.getDecoder().decode(header.substring(6).trim());
            String credentials = new String(decoded, StandardCharsets.UTF_8);
            int separator = credentials.indexOf(':');
            if (separator < 0) {
                return false;
            }

            byte[] suppliedUser = credentials.substring(0, separator)
                    .getBytes(StandardCharsets.UTF_8);
            byte[] expectedUser = USERNAME.getBytes(StandardCharsets.UTF_8);
            byte[] suppliedPassword = credentials.substring(separator + 1)
                    .getBytes(StandardCharsets.UTF_8);
            byte[] expectedPasswordBytes = expectedPassword.getBytes(StandardCharsets.UTF_8);

            return MessageDigest.isEqual(suppliedUser, expectedUser)
                    & MessageDigest.isEqual(suppliedPassword, expectedPasswordBytes);
        } catch (IllegalArgumentException ex) {
            return false;
        }
    }

    private synchronized boolean allowProcessingRequest() {
        long now = System.currentTimeMillis();
        long currentMinute = now / 60_000L;
        long currentDay = now / 86_400_000L;

        if (currentMinute != minuteWindow) {
            minuteWindow = currentMinute;
            minuteCount = 0;
        }
        if (currentDay != dayWindow) {
            dayWindow = currentDay;
            dayCount = 0;
        }

        if (minuteCount >= MAX_PER_MINUTE || dayCount >= MAX_PER_DAY) {
            return false;
        }

        minuteCount++;
        dayCount++;
        return true;
    }
}
