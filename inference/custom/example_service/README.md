# Custom Inference Service — Example

This is a minimal FastAPI service that implements the interface expected by
`CustomBackendClient` in `apps/api/src/polyglot_gateway/inference/custom.py`.

## Contract

The service must expose:

```
POST /translate
  Body:  {"text": str, "src": str, "tgt": str, "model": str}
  Reply: {"translation": str}

GET /health
  Reply: {"status": "ok"}
```

## Running locally

```bash
pip install fastapi uvicorn
uvicorn main:app --port 9000
```

## Wiring it in

1. Build and start the service (Docker or bare process).

2. Add an entry to `configs/models.yaml`:
   ```yaml
   backends:
     my-model:
       type: custom
       url: custom-service:9000   # Docker service name, or host:port
       model_name: my-model
   ```

3. Rebuild the gateway image and restart:
   ```bash
   make build-app && make api-up
   ```

4. Translate using your new backend:
   ```bash
   curl -X POST http://localhost:80/translate \
     -H "X-API-Key: changeme" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello", "source": "en", "target": "vi", "model": "my-model"}'
   ```

## Using Docker Compose

Add a service block to `infra/docker-compose.yml` (see the commented `custom-service` example
at the bottom of that file), then add it to the `depends_on` list of the `worker` service.
