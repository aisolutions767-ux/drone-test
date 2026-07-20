FROM mcr.microsoft.com/playwright:v1.40.0-jammy

# Install python3, pip, and ffmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries
RUN playwright install chromium

# Copy application files
COPY . .

# Set environment variables for headless webgl rendering
ENV DISPLAY=:99
ENV XVFB_WHD=1280x720x24
ENV PLAYWRIGHT_JSON_OUTPUT=true

# Expose port
EXPOSE 8765

# Start command with virtual framebuffer for headless WebGL support
CMD ["xvfb-run", "-a", "python3", "drone_server.py"]
