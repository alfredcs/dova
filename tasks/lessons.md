# Lessons

## LLM provider router (src/dova/config/providers.py)

- **Removing a provider from `LLM_PROVIDER_ORDER` does NOT disable it.**
  `_provider_priority()` returns `99` for any provider absent from the order list,
  and `_get_sorted_providers()` iterates **all enabled providers** regardless of the
  order — so an "excluded" provider is still constructed and runs at the *tail* of
  the fallback chain (where, if broken, it 404s). To truly disable a tier, gate its
  **construction** on `"<name>" in DEFAULT_PROVIDER_ORDER`, not just the env order.

- **Verify the gateway catalog before "fixing" a provider by swapping model IDs.**
  Mantle's `/anthropic/v1/messages` returns a structured `not_found_error` for
  missing models — *route exists ≠ model exists*. Guessing model-ID variants
  (Bedrock-style, native-dated, public aliases) burned several probe rounds;
  `GET {host}/v1/models` settled it in one call. The Mantle gateway serves **no
  Claude models** — Claude is only the Bedrock primary tier.
