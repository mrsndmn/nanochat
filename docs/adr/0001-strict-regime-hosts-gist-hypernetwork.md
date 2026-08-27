# Gist Hypernetwork is tested in the strict block-causal regime, not the near-baseline alt regime

The windowed-alternation regime scores far better (BPB 0.8028 vs 0.8191 for strict
NLTK K=8, full-causal baseline 0.8022), so testing new gist mechanisms there looks
attractive. We deliberately host the Gist Hypernetwork experiment in the *worse*
strict regime: there the gist channel is the only cross-sentence path, so gist
content quality has maximum leverage, while in the alt regime the local window
carries most context and a null result would be uninformative about the mechanism.
The alt-regime code also lives only on unmerged sibling branches
(`experiment/sentence-attention-tuning-2`), so this choice keeps the diff on-branch.
A win in strict (≥0.003 BPB vs the fixed-gist control) justifies porting the
mechanism into alt as a follow-up; the reverse order risks burning the budget on a
predictable null.
