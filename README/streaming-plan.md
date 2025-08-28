# Streaming Refactoring Plan for Kokoro TTS

**Andy Hertzfeld**

### Philosophy: Simpler is Better

The current `KokoroPipeline` has fragmented and overly complex logic for chunking text before synthesis. English is handled differently from other languages, and the default behavior relies on simple newline splits. This increases complexity and leads to inefficient memory usage for long inputs.

This plan outlines a refactoring to a unified, stream-based approach that processes text sentence-by-sentence. This will reduce the memory footprint, improve responsiveness (time-to-first-audio), and dramatically simplify the codebase.

---

### The Plan

The core idea is to introduce a single, robust sentence-splitting mechanism at the entry point of the pipeline. This becomes the primary method for chunking text for all languages, ensuring a consistent and efficient processing flow.

#### Step 1. Unify Text Splitting in `KokoroPipeline.__call__`

-   **Action:** Modify the `__call__` method in `mlx_audio/tts/models/kokoro/pipeline.py`.
-   **Details:**
    -   Replace the current `re.split(split_pattern, ...)` logic with a more intelligent sentence tokenizer. A more robust regex like `r'(?<=[.!?])\s+(?=[A-Z])|(?<=Dr\.)\s+|\n\n'` will be used to split text more reliably, handling abbreviations and using paragraph breaks as explicit chunks.
    -   This new splitter will become the default behavior. The `split_pattern` argument will be deprecated or removed to simplify the API.

#### Step 2. Simplify the Main Processing Loop

-   **Action:** Refactor the main `for` loop within `__call__`.
-   **Details:**
    -   The loop will now iterate over sentences yielded by the new, unified splitter.
    -   The separate, complex logic paths for English (`if self.lang_code in "ab":`) versus non-English languages will be removed. All sentences, regardless of language, will be passed through the same processing path (G2P -> phoneme conversion -> inference).

#### Step 3. Refactor and Simplify `en_tokenize`

-   **Action:** Adjust the responsibility of the `en_tokenize` method.
-   **Details:**
    -   The primary chunking logic based on the 510-phoneme limit will be removed from this method. Since the pipeline now processes one sentence at a time, `en_tokenize`'s role changes from a primary chunker to a **safeguard for pathological cases**.
    -   Its only responsibility will be to check if a *single sentence* is too long (e.g., > 400 phonemes). If it is, it will perform a "dumb split" at the nearest comma or a hard character limit and log a warning. This prevents stalls on rare, extremely long sentences.

#### Step 4. Remove Redundant Code

-   **Action:** Delete the now-obsolete chunking logic for non-English languages.
-   **Details:** The entire `else` block (lines ~405-436) that handles manual chunking for non-English text becomes redundant and will be removed. The new sentence splitter at the beginning of `__call__` handles this for all languages.

---

### Expected Outcomes

1.  **Reduced Memory Footprint:** Peak memory usage will be determined by the longest single sentence, not the length of the entire input text.
2.  **Improved Responsiveness:** The system will `yield` audio as soon as the first sentence is synthesized, providing a much faster "time to first sound."
3.  **Simplified Codebase:** A single, unified, and easy-to-understand logic path for text processing makes the code more maintainable and robust. It follows the "simpler is better" principle.