# Whisper ASR with ROCm Support

Whisper automatic speech recognition with AMD GPU (ROCm) support.

## Features

- ✅ **AMD GPU support** via ROCm/PyTorch
- ✅ **10-20x faster** than CPU (for medium model)
- ✅ Compatible API with `onerahmet/openai-whisper-asr-webservice`
- ✅ Multi-language support
- ✅ Word-level timestamps

## Performance

| Device | 1 min audio | 10 min audio |
|--------|-------------|--------------|
| CPU | ~30-60s | ~5-10 min |
| **ROCm GPU** | **~3-10s** | **~30s-2min** |

## Build & Run

```bash
cd /home/michal/ai-code/whisper-rocm

# Build image
docker build -t whisper-rocm .

# Run
docker run -d \
  --name whisper-rocm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e ASR_MODEL=medium \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -p 9000:9000 \
  whisper-rocm

# Check logs
docker logs -f whisper-rocm

# Test
curl http://localhost:9000/health
```

## Docker Compose

Replace existing whisper service:

```yaml
whisper-rocm:
  build:
    context: ${AI_CODE_PATH}/whisper-rocm
  container_name: whisper
  environment:
    - ASR_MODEL=medium
    - ROCR_VISIBLE_DEVICES=0
    - HSA_OVERRIDE_GFX_VERSION=12.0.1
  devices:
    - /dev/kfd
    - /dev/dri
  group_add:
    - "485"  # video
    - "482"  # render
  ports:
    - "127.0.0.1:9000:9000"
  restart: unless-stopped
```

## API

### GET /health
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "device_name": "AMD Radeon RX 7900 XTX"
}
```

### POST /asr
Upload audio file for transcription.

**Compatible with original API!** No changes needed in doc-converter code.

## Models

Available models (via ASR_MODEL env):
- `tiny` - fastest, lowest quality
- `base`
- `small`
- `medium` - **recommended** (good balance)
- `large` - best quality, slower

## GPU Detection

The server auto-detects ROCm/CUDA:
```
🚀 Using CUDA/ROCm device: AMD Radeon RX 7900 XTX
✅ Model loaded successfully on cuda
```

If GPU not found:
```
⚠️  CUDA/ROCm not available, using CPU
```

## Troubleshooting

### GPU not detected
Check if ROCm sees GPU:
```bash
docker exec whisper rocm-smi
```

### Build errors
Make sure you have ROCm drivers installed on host.

### Slow transcription
Check if actually using GPU:
```bash
docker logs whisper | grep "device:"
# Should show: device: cuda
```
