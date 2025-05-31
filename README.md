# NSFW Playground

[![Build Status](https://img.shields.io/docker/build/username/nsfw-playground)][docker_hub]
[![License](https://img.shields.io/github/license/username/nsfw-playground)]

> HIGH-PRIORITY: Supabase configuration and secrets must be set in environment variables

## Requirements
- Python 3.9+
- Node.js 16.20.0
- Redis 6.2+
- H100 GPU instance for production

## Deployment Setup

### Environment Variables

```env
c:\Users\Administrator\nsfw-playground\.env
NSFW_API_KEY=your_text_generation_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_secret_key
REDIS_HOST=redis
```

> GPU users: Use `nvidia-docker` for hardware acceleration

[docker_hub]: https://hub.docker.com/r/username/nsfw-playground