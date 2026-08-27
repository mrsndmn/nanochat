# The hypernetwork round pays for two arms — gated residual AND forced-replace — to make a null interpretable

Every prior content-flavored gist variant (eos_clone, real_mean, hash buckets,
gist-head projections) came back neutral-to-worse, so there is a live hypothesis
that gist *input* content is simply ignored by the trunk. A single gated arm
(embedding = fixed gist row + α·h(sentence), α zero-init) cannot distinguish
"content is useless" from "the gate stayed shut". We therefore also train a
forced-replace arm (embedding = h(sentence), no fallback, no gate): if gated nulls
at α≈0 and forced is worse than the fixed-gist control, the content channel is
genuinely useless and the whole direction — including the deferred test-time
training follow-up — is cleanly falsified; if forced beats the control, content
matters and the gate was too conservative. One extra 10k-step training job is the
price of not having to re-run the round to interpret its outcome.
