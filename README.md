# HybridAgent latent probe

A small, cheap probe of **[HybridAgent](https://github.com/togethercomputer/HybridAgent)**
([Zou et al., 2026, arXiv:2610.18861](https://arxiv.org/abs/2610.18861)). HybridAgent is an LLM agent that switches
between *latent mode*, where it feeds its own hidden state back in as the next input, and
*text mode* for tool calls and answers. A small adapter called HybridLink maps the hidden
state back into input-embedding space. The paper says each latent thought can be read back
as "pseudo-text" by taking the nearest vocabulary token at every step.

The question here is simple: **when you actually read those pseudo-texts, what do they show,
and what do they hide?**

Everything ran on Hugging Face Jobs, one L4 GPU, using the released
`HybridAgent-8B-Agentic` checkpoint. Eleven questions in total: 3 maths, 8 classic
intuition traps. All the GPU time, including two failed setup runs, came to well under £1.

> Status: exploratory notes from n = 11 single samples. Treat these as observations, not results.

## Scoreboard

| Question | Trap answer | Model | Latent share |
|---|---|---|---|
| Trailing zeros of 100! | – | **24** ✅ | 39% |
| Circle x²+y²=25 reflected in (4,1) | – | **(1,−16,−4,43)** ✅ | 32% |
| Bat and ball | 10p | **5p** ✅ | 25% |
| 5 machines, 5 widgets, 5 min | 100 min | **5** ✅ | 32% |
| Lily pads double, 48 days | 24 | **47** ✅ | 24% |
| Horse bought and sold twice | £10 | **£20** ✅ | 24% |
| Months with 28 days | 1 | **1** ❌ (correct: 12) | 27% |
| Sally's sisters | 2 | **1** ✅ | 28% |
| Overtake 2nd place | 1st | **2nd** ✅ | 33% |
| 1-mile train, 1-mile tunnel | 1 min | **2** ✅ | 31% |
| Is 3599 prime? | yes | **No (59×61)** ✅ | 32% |

10/11 correct. The one miss is the only *language* trap; the arithmetic-flavoured traps
didn't catch it. These puzzles are famous, though, so contamination is likely.

## What the pseudo-text readout shows

**1. Sometimes it shows the model's mistakes as they happen.** In the circle problem, a
latent span reads *"the original circle is x²+y²=25, so x²+y²=25"*. That mixes up the
reflected coordinates with the original ones. The text that follows makes the same
mistake, briefly derives a straight line, and then recovers.

**2. Once, it shows the right answer and the text then abandons it.** In the
28-days question, a latent span reads:

> *"…the only month that has 28 days in a common year is February. but I a leap year, February has 29 days. So, the number would be 1 2 ,"*

Qwen tokenises digits one at a time, so "1 2" plausibly means **12**, the correct answer. The very
next text span says *"even the shortest months are 30 days long. So the only month with 28
days is February"* (false), and the final answer is 1. This is one sample, and the span is cut off right
after the digits, so the reading is ambiguous. It's still the most interesting single moment
in the run: it reverses the usual worry, with the latent right and the text wrong.

**3. Real arithmetic does happen in latent mode.** In the 3599 problem, latent spans carry out
*"3+5=8, 8+9=17, 17+9=26"* and *"359 − (2×9) = 359 − 18 = 341"* correctly. On the
simpler problems, latent spans were mostly connective steps ("let me set up an
equation", "subtract both sides") and the numbers surfaced in text.

## What it hides

**4. Content nouns decode badly; function words and digits decode well.** Steps with a
nearest-token cosine below 0.5 are nearly all nouns:

| Meant | Read as (cosine) | Runners-up |
|---|---|---|
| widget(s) | *tiles, digit, cookies, units* (~0.25) | stickers, items |
| machines | *cars* (0.31) | machines 0.31, trees 0.31 |
| tunnel | **𝄕** (0.31) | ספטמ, tangent, tunnel 0.29 |
| lily pads | *l y cards ניוזל* | patch, plant |
| bat | *ball* (0.33) | bat 0.32 |
| leap (year) | *le* (0.26) | length |

The model clearly still distinguishes these things, because it solves the problems. So the
information is in the vector. But the nearest single token is a poor label for it. Digits,
punctuation and words like *the/of/is* sit at cosine > 0.95.

**5. Readability ≠ faithfulness.** The mean readability score is the average nearest-token
cosine, so it is dominated by those easy tokens. The highest-scoring span in the whole run
(0.97) is a degenerate loop. While checking 25!, the model wrote out its real digits and then
emitted "0" for 200 steps until it hit the step cap. Meanwhile, the train problem's spans say
*"the length of the train + the length of the train"* with "tunnel" decoded as "train" or
junk. The text, meanwhile, correctly adds tunnel + train.

**6. Exit markers that don't exit.** Several latent steps decode as `</latent>` (cosine
≈ 0.97–0.99) but generation stays in latent mode. I haven't looked at how the engine handles that.

## Summary

The pseudo-text is a lossy shadow of the latent state: honest where it's legible, but its
legibility is mostly the easy tokens. Of the two interesting events, neither is "the latent
corrects an error before the text does". One shows an error forming in latent first; the
other shows a correct latent being overridden by text.

## Running it yourself

You need a Hugging Face account with Jobs enabled (pay-as-you-go). With the `hf` CLI:

```bash
hf jobs run --flavor l4x1 --timeout 40m -e SET=traps python:3.11 \
  /bin/sh -lc "curl -sSL https://raw.githubusercontent.com/SoulFireMage/hybridagent-latent-probe/main/probe.sh | sh"
hf jobs logs <job-id>
```

`SET=maths` runs the other question file. Add your own `questions/<name>.txt` (one question
per line). An 8-question run takes about 12 minutes including the 16.5 GB checkpoint download.

### Gotchas I hit, so you don't

- **`pip install -r requirements.txt` aborts on the `deepspeed` pin** (needed only for
  training), and pip then installs *nothing*. That leaves `transformers` and `tavily`
  missing. `probe.sh` installs pins one by one and skips deepspeed.
- **The inspector imports `tavily` at start-up**, even if you never search. Any non-empty
  `TAVILY_API_KEY` lets it load; searches then just fail. The model tried to search twice
  and carried on without it.
- **Hugging Face Jobs redacts the *values* of env vars in logs.** Setting
  `PYTHONUNBUFFERED=1` turned every `1` in the output into `<redacted>`, which is
  catastrophic for maths traces. Don't pass short env values.
- **The HTML report dies with the container.** `dump_trace.py` prints a condensed text
  version to the log instead.
- The 8B model needs ~16.5 GB in bf16. That fits an L4 (24 GB), not a 10 GB consumer card,
  without quantisation, which I haven't tried with this engine.

## Files

- `probe.sh` – the whole job: install, run the inspector per question, dump traces.
- `dump_trace.py` – HTML report → condensed text (keeps only low-confidence readout steps).
- `questions/` – question sets used here.

Probe and write-up by Richard (SoulFireMage), with Claude.
