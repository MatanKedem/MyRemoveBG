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
 * Protects the whole web app with HTTP Basic authentication.
 *
 * Set MYREMOVEBG_ACCESS_PASSWORD in the server environment before starting
 * the app. The password is never stored in source control.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class AccessProtectionFilter extends OncePerRequestFilter {

    private static final String USERNAME = "myremovebg";
    private static final int PASSWORD_LENGTH = 6;

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        String configuredPassword = System.getenv("MYREMOVEBG_ACCESS_PASSWORD");
        if (configuredPassword == null || configuredPassword.length() != PASSWORD_LENGTH) {
            response.setStatus(HttpServletResponse.SC_SERVICE_UNAVAILABLE);
            response.setContentType("text/plain");
            response.setCharacterEncoding(StandardCharsets.UTF_8.name());
            response.getWriter().write(
                    "MyRemoveBG is locked: configure MYREMOVEBG_ACCESS_PASSWORD "
                    + "with exactly 6 characters in the server environment.");
            return;
        }

        if (!isAuthorized(request, configuredPassword)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setHeader("WWW-Authenticate", "Basic realm=\\"MyRemoveBG\\", charset=\\"UTF-8\\"");
            response.setHeader("Cache-Control", "no-store");
            response.setContentType("text/plain");
            response.setCharacterEncoding(StandardCharsets.UTF_8.name());
            response.getWriter().write("Authentication required.");
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
}
