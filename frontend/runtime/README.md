# Runtime Client

This folder contains the TypeScript-side contract for the Python runtime bridge.

Current scope:
- runtime state types
- event types
- legacy UI spec types
- a small fetch client for the localhost HTTP bridge

Expected bridge endpoints:
- `GET /health`
- `GET /bridge`
- `GET /capabilities`
- `GET /commands`
- `GET /state`
- `GET /events`
- `GET /workers`
- `GET /settings-schema`
- `GET /ui-spec`
- `GET /ui-spec-override`
- `GET /preview/current`
- `POST /commands/{commandName}`

The Python bridge is optional and can be enabled by:
- environment variable `CAPTION_RUNTIME_HTTP=1`
- or runtime command `bridge.start`
