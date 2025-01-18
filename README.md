# llmws
A Python-based server for exposing a large language model (LLM) through WebSocket, allowing real-time interaction with minimal latency. This project includes support for transformers models and is designed to be lightweight, efficient, and highly configurable.


Features:
* WebSocket-based communication: Enables bidirectional, real-time communication with clients.
* LLM integration: Supports loading models from Hugging Face's transformers.
* Streaming token generation: Sends tokens incrementally for a responsive user experience.
* CUDA acceleration: Utilizes GPU if available, with optional Flash Attention for faster inference.
* Customizable generation settings: Adjust tokens, temperature, and other parameters.
* Lightweight and extensible: Ready for integration with command-line and web clients.
