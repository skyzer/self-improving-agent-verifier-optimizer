FROM python:3.12-slim

WORKDIR /app

COPY README.md index.html ux-structure-options.html verifier_lab.py favicon.svg favicon.ico favicon-32x32.png apple-touch-icon.png favicon-512.png ./
COPY data ./data
COPY tests ./tests

# Build-time smoke checks: deterministic generator + unit tests.
RUN python3 verifier_lab.py --validate \
    && python3 -m unittest discover -s tests -v

EXPOSE 8787

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/index.html', timeout=2).read(100); print('ok')" || exit 1

CMD ["python3", "-m", "http.server", "8787", "--bind", "0.0.0.0"]
