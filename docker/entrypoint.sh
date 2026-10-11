#!/bin/sh
# Container entrypoint: "serve" starts the FastAPI API and the Streamlit UI;
# anything else is run as a normal command (training, tests, analysis).
set -e

if [ "$1" = "serve" ]; then
  python -m uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" &
  API_PID=$!
  python -m streamlit run app/demo_app.py \
      --server.address 0.0.0.0 --server.port "${PORT:-8501}" \
      --server.headless true --browser.gatherUsageStats false &
  UI_PID=$!
  # Exit (and let Docker restart us) if either process dies.
  trap 'kill $API_PID $UI_PID 2>/dev/null' TERM INT
  while kill -0 $API_PID 2>/dev/null && kill -0 $UI_PID 2>/dev/null; do
    sleep 5
  done
  kill $API_PID $UI_PID 2>/dev/null || true
  exit 1
fi

exec "$@"
