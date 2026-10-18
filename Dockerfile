FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home appuser
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser load_db.py .
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser deploy/ deploy/
RUN mkdir -p /app/db /app/cache /app/dataset && python load_db.py --data data --db /app/dataset/app.sqlite && python deploy/prepare_analytics.py --db /app/dataset/app.sqlite && chmod 444 /app/dataset/app.sqlite && chown -R appuser:appuser /app/db /app/cache
ENV DB_PATH=/app/dataset/app.sqlite CACHE_DB=/app/cache/cache.sqlite DEMO_USAGE_DB=/app/cache/demo-usage.sqlite
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/ready', timeout=3)"
CMD ["python", "deploy/start.py"]
