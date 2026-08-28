# Sentence Attention

Domain language for the sentence-attention experiment line: block-structured
attention over sentence blocks with gist-token summary channels, trained and
evaluated on the nanochat d12/10k-step protocol.

## Language

**Gist Token**:
A reserved token id placed past the real vocab, inserted between sentences by the
data pipeline; its position is the write-slot through which later sentence blocks
see a summary of an earlier sentence.
_Avoid_: summary token, EOS token, boundary token

**Strict Block-Causal Regime**:
The sentence-attention variant where a token attends only to its own sentence
block plus all earlier gist tokens in the document — gists are the sole
cross-sentence channel.
_Avoid_: nltk regime, pure sentence attention

**Windowed Alternation Regime**:
The sentence-attention variant where some layers use a short local attention
window, so cross-sentence information flows through both the window and any
gists; gists are not load-bearing here.
_Avoid_: alt regime, sliding-window arm

**Gist Slot**:
One of the K gist positions emitted at a single sentence boundary; slots are
distinguishable (each can carry a different facet of the sentence summary).
_Avoid_: gist copy, gist repeat

**Engram Gist Memory**:
A hashed bigram lookup table whose retrieved entries feed the Gist Hypernetwork's
key/value stream — stored n-gram associations, as opposed to content recomputed
from the context window.
_Avoid_: bigram hash embeddings (that names the token-level residual-stream variant), conditional memory

**Gist Hypernetwork**:
A learned encoder that maps a completed sentence's tokens to the gist
embedding(s) emitted at that sentence's boundary, replacing the fixed learned
gist embedding rows.
_Avoid_: gist head (that names the per-position value/output projection variant), gist init (that names static initialization schemes)
