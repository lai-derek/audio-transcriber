# Tests and decisions

Most measurements below were recorded September 30–October 4, 2026. They predate
the Audio Transcriber rename, Python 3.14 update, and current AI prompts. The
November 17 speech-model comparison ran on the current app; the November 18
comparison used a separate sherpa-onnx script on the same audio and scorer.

Current defaults: speech uses NVIDIA Parakeet TDT 0.6B v2 (int8, sherpa-onnx, CPU,
two threads) for file imports and live dictation, with Whisper Small and Base as
alternatives; cleanup uses AUTO (GPU, then CPU). New file imports
use Medium cleanup with Mixed notes; live dictation uses Medium with Mostly prose.
The desktop keeps selected models loaded until it closes. Current prompts are
`cleanup-v4`, `format-v6`, and `summary-v6`; the historical quality and timing
results below do not validate the newer notes, summary styles, or overview wording.

The project began as an attempt to use an NPU for local speech transcription
and cleanup while leaving the computer usable. Tests led to different device
choices: **CPU for speech; GPU with CPU fallback for cleanup**. NPU stays optional.

Measured September 30–October 4, 2026 on one Windows x64 host: Core Ultra 5 335,
Intel integrated graphics, Intel AI Boost NPU, 8 logical processors, 31.51 GiB
usable RAM. Initial runs used OpenVINO 2026.3.0 / GenAI 2026.3.0.0;
October 3–4 runs used 2026.4.0 / 2026.4.0.0. The linked reports contain
methods, pinned models, and numerical results. Results apply to this computer
and these workloads; power consumption was not measured.

## Speech: model and device

The same 100 NPTEL clips from 100 lecture videos (838.42 seconds, 1,889 reference
words) ran on CPU. Word error rate counts word insertions, deletions, and
substitutions divided by reference words; lower is better.

| Model | Word edits | Pooled word error rate | Chunk time / audio time |
|---|---:|---:|---:|
| Whisper tiny.en INT4 | 455 | 24.09% | 0.1355 |
| Whisper base.en INT4 | 373 | 19.75% | 0.1803 |

**Decision:** tiny remains the small first-use model; base is the accuracy option.
Base improved pooled error by 4.34 percentage points at greater runtime cost.
The published annotations were used as supplied. Full-lecture accuracy remains
unmeasured.
[Clip selection, sources, and per-case results](../host-validation/evidence/2026-09-30/lecture-gold-pilot/README.md).

A 118.76-second MIT excerpt then ran twice per model/device. Below, recorded
chunk time divided by audio duration includes model loading but excludes app
startup; a ratio below 1 means faster than real time.

| Model | CPU: initial / repeat | GPU: initial / repeat | NPU: initial / repeat |
|---|---:|---:|---:|
| tiny.en | 0.0509 / 0.0206 | 0.1273 / 0.1135 | 0.4712 / 0.6255 |
| base.en | 0.0330 / 0.0320 | 0.0583 / 0.1174 | 1.1788 / 1.5591 |

**Decision:** start speech on CPU, which was fastest in these runs. The correct
NPU path rebuilds Whisper each chunk because reuse produced stale, truncated
text. Base on NPU was slower than real time here. Runtime binding succeeded on
all three devices; utilization was not independently measured.
[Repeat results](../host-validation/evidence/2026-09-30/lecture-gold-pilot/README.md#matched-lecture-device-repeat),
[four-clip check and NPU defect](../host-validation/evidence/2026-09-30/device-lecture-evaluation/README.md).

## Speech: larger models and Orukeet (November 17, 2026)

Five local models ran through the app's own pipeline (30-second chunks, CPU,
OpenVINO 2026.4) on two sets:

- **TED talk** (Craig Venter, TED-LIUM release 3, 15.1 minutes, 2,398
  human-transcribed words): true word error rate.
- **15 of the author's phone recordings** (30.9 minutes of unscripted speech;
  not published): no human transcript yet, so each model is compared with a
  hosted Oruk Resonance 1 transcript. This is disagreement, not error: hosted
  Oruk Orukeet and Resonance 1 disagree on 7.1% of words, the noise floor.
  Resonance 1 is the reference because local Orukeet shares hosted Orukeet's
  model family.

Both sides were normalized the same way: lowercase, no punctuation, numbers
written out, fillers dropped. Time is processing time divided by audio length,
including decoding.

| Model | TED word error | Disagreement, own recordings | Time / audio | Download |
|---|---:|---:|---:|---:|
| Whisper tiny.en INT4 | 11.8% | 29.7% | 0.040 | 41 MB |
| Whisper base.en INT4 (default) | 5.5% | 23.3% | 0.068 | 64 MB |
| Whisper small.en INT4 | 3.8% | 13.8% | 0.161 | 162 MB |
| Whisper large-v3-turbo INT4 | 3.8% | 9.8% | 0.486 | 478 MB |
| Orukeet INT8 (sherpa-onnx, 2 threads) | 3.3% | 10.7% | 0.100 | 487 MB |
| Oruk Orukeet, hosted (reference check) | 3.0% | 7.1% | — | — |
| Oruk Resonance 1, hosted (reference) | 3.2% | — | — | — |

Clean TED speech understated the default's error on real recordings: Base's
5.5% on TED became about 23% disagreement on unscripted phone audio. Word
counts matched across models, so the gap is misheard words, not dropped
speech. Larger models close most of it. Oruk's Spectra-2 returned the same
transcripts as hosted Orukeet and is not listed separately.

On November 17, Base stayed the default while sherpa-onnx models were checked;
local Orukeet beat Small.en on TED and speed but needed a new runtime and a
licence review. The hosted drafts came from a private script that is not part of
this repository; a local scorer computed the numbers. The audio and transcripts stay
private.

### November 18: sherpa-onnx models, and the new default

Four more models ran with sherpa-onnx 1.13.4 on the CPU (two threads), on the
same audio with the same scorer, cut at the quietest 100 ms before each
28-second limit. Speed is relative to local Orukeet in a controlled check
(180 s of TED, two rounds, power throttling off), measured on battery. The
silence check fed four 10-second inputs: digital silence, pink noise at
−50 dBFS, and room tone from the recordings at its own level and at −45 dBFS.

| Model (int8) | TED word error | Disagreement, own recordings | Speed vs Orukeet | Token times | Text on silence or noise |
|---|---:|---:|---:|---|---|
| Fluid Speech Pro | 2.8% | 6.4% | 4.6× slower | yes | 0 of 4 |
| Rumble-2-Flash | 2.9% | 9.5% | 3.3× slower | no | 4 of 4 |
| Parakeet TDT 0.6B v2 | 3.0% | 6.8% | 1.4× slower | yes | 0 of 4 |
| Qwen3-ASR 0.6B | 3.6% | 10.5% | 5.0× slower | no | 4 of 4 |

On AC power, Parakeet v2 processed the 180 s of TED in 0.050 of real time with two
threads and 0.053 with four; eight threads was slowest. A 10-second clip took
0.47 s with two threads and 0.66 s with four. The app uses two threads.

The app then ran Parakeet v2 through its own pipeline and resident worker, on AC
power, and the same scorer measured the result. The first try used pause-v2 cuts:
on TED it found a qualifying pause at only 1 of 32 cuts, cut mid-speech
elsewhere, and dropped words there (3.3% word error). With pause-v1 cuts at most
28 seconds, the app matched the harness:

| Run | TED word error | Disagreement, own recordings |
|---|---:|---:|
| Harness (quietest 100 ms before 28 s) | 3.0% (71 errors) | 6.8% |
| App, pause-v2 cuts | 3.3% (79) | 6.8% |
| App, pause-v1 cuts (default) | 3.0% (73) | 6.7% |

The app processed the 46 minutes in 0.051 of real time, decoding included (0.046
for recognition alone). An 8-second clip took 0.43 s in the loaded worker.
Subtitle cues were in order, with a median length of 4.1 s and a maximum of 13 s.

**Decision:** Parakeet TDT 0.6B v2 is the default speech model for file imports
and live dictation. Whisper Small and Whisper Base stay as selectable alternatives;
Whisper Tiny remains for tests and the command line but is not offered in the app.
Reasons:

- **Accuracy:** 3.0% word error on TED against Small's 3.8% and Base's 5.5%, and
  6.8% disagreement on the recordings against 13.8% and 23.3%, below the 7.1%
  disagreement between the two hosted drafts.
- **Speed:** 0.051 of real time in the app, against 0.068 for Base and 0.161 for
  Small on November 17. The test report lists those runs as on battery, so the
  comparison is approximate. It needs no device benchmark.
- **Timestamps:** every token has a start time and duration, so the Original tab,
  seeking, and subtitles get sentence-level times.
- **Silence:** it produced no text on any of the silence or noise inputs.
- **Licence:** CC BY 4.0, which needs attribution but has no share-alike clause.

Fluid Speech Pro was as accurate but 4.6× slower than Orukeet in its int8 build.
Rumble-2-Flash and Qwen3-ASR were ruled out: no timestamps, and invented text
on every silence or noise input.

**Caveats:**

- No hand-corrected reference exists yet for the author's recordings; the
  disagreement numbers measure agreement with one hosted draft, whose errors the
  Parakeet-family models may share. One TED talk is the only true error figure.
- Quiet lecture-hall audio was not tested; the model card warns about very quiet
  input.
- English only, like the Whisper `.en` models it replaces.

## Cleanup: quality, speed, and resource cost

An initial 1.5B model lost a corrected date without triggering the checks. The
chosen Qwen2.5 7B INT4 model handled targeted pilot examples better, at about
4.5 GB of downloads and several GiB of RAM. The model-size choice rests on this
small pilot. [Pilot](../host-validation/evidence/2026-10-02/ai-cleanup/README.md).

The 7B model ran 25 matched configurations twice per device with the cleanup-v2
prompt. One ambiguous case was excluded. Warmed timing excludes loading and
warmup. Later cleanup prompts have not had this three-device comparison.

| Device | Median request | Mean 25-request batch | All targeted checks passed | Final content checks passed |
|---|---:|---:|---:|---:|
| CPU | 3.78 s | 194.05 s | 40/48 | 46/48 |
| GPU | 1.09 s | 49.55 s | 42/48 | 46/48 |
| NPU | 2.99 s | 106.48 s | 40/48 | 46/48 |

GPU was faster and all devices passed 46 of 48 final-content checks. Literal
commands failed on every device; some warned outputs retained the source text.
[Inputs, outputs, startup times, and limitations](../host-validation/evidence/2026-10-02/ai-cleanup-comparison/README.md).

A separate fixed short request ran alongside moderate hardware graphics activity:

| Cleanup device | Warmed request | Worker CPU capacity used | Resident worker RAM |
|---|---:|---:|---:|
| GPU | 1.14 s | 10.1% | 4.67 GiB |
| NPU | 3.06 s | 0.96% | 5.46 GiB |
| CPU | 2.64 s | 27.9% | 8.48 GiB |

CPU percentages are the worker's share of whole-CPU capacity. RAM includes the
resident speech and cleanup pipelines, excluding the browser and loading. The
desktop used separate speech and cleanup workers in this comparison. With heavier graphics work:

| Cleanup device | Idle → cleanup graphics | Approximate slowdown | Cleanup request |
|---|---:|---:|---:|
| GPU | 59.3 → 55.4 fps | 7% | 7.78 s |
| NPU | 58.6 → 38.9 fps | 34% | 5.25 s |
| CPU | 59.1 → 53.4 fps | 10% | 7.30 s |

**Decision:** cleanup AUTO tries an available OpenVINO GPU, then CPU if it fails;
explicit device choices remain strict. GPU offered the measured balance of speed,
RAM, and graphics interference. NPU used least CPU but caused most graphics
slowdown; the cause remains unknown. These were synthetic browser workloads.
GPU counters above 100% were discarded, leaving GPU-capacity usage unresolved. [Resource measurements and method](../host-validation/evidence/2026-10-02/cleanup-resource-cost/README.md).

Cached GPU loading was 1.67–3.63 seconds in the matched comparison; first-ever
NPU compilation in the earlier pilot took 169.7 seconds. At that checkpoint, each desktop action
started a new worker; long cleanup reused its model across blocks. AUTO does not
measure foreground load or impose resource caps. Off skips the cleanup model.

## Text model speed: batching and prompt lookup (November 18, 2026)

With Parakeet, speech took about 3% of a lecture's processing time; cleanup, notes, and
summaries took the rest. A full background set for one lecture (three cleanup levels,
their notes and summaries) took about 2.4 times the recording's length on the GPU.

**Method.** Every text-model request the app made for two recordings was logged:
a 6.5-minute phone recording and the 15-minute TED talk (cleanup at all three levels,
Mixed and Structured notes, summaries in all three styles; 123 requests). The same
requests were then replayed under each option on the GPU (Core Ultra 5 335,
Intel Graphics, AC power). Accuracy checks:

- cleanup word error against the talk's human transcript, and against Resonance 1 for
  the phone recording;
- blocks the app's own checks reject;
- summary number flags.

The same request on the GPU does not always give the same text: two runs of one
pipeline matched on 39 of 41 requests, and different pipeline types matched on about 3
of 41. Differences of that size are noise.

| Option (GPU) | 6.5-min recording | TED talk | Stray "!" tokens |
|---|---:|---:|---|
| One request at a time (before) | 422 s | 987 s | none |
| Speculative decoding, Qwen2.5 1.5B draft | 346 s | — | none |
| Speculative decoding, Qwen2.5-Coder 0.5B draft | 277 s | — | none |
| Prompt lookup | 253 s | — | in one of three runs |
| Batching (blocks of a job sent together) | 172 s | 306 s | in one notes request |
| Batching + prompt lookup | 156 s | 248 s | thousands |
| Batching, 16-bit key/value cache | 192 s | 273 s | none |
| **Batching + prompt lookup, 16-bit key/value cache** | **140 s** | **247 s** | **none** |
| The same in the app, summary parts reused | — | 186 s | none |

Prompt lookup drafts the next few tokens from the request itself. Cleanup mostly copies
its input, so cleanup gained the most: about 70 s down to 9–11 s per level on the phone
recording. The stray tokens ("!", token 0, such as "24,0!0") came from the GPU's default
8-bit key/value cache. With a 16-bit cache, four full runs and two app runs had none.

| TED talk (raw speech: 3.0%) | Light | Medium | Heavy | Cleanup blocks rejected | Notes blocks rejected |
|---|---:|---:|---:|---:|---:|
| Before, app run | 6.5% | 7.0% | 12.4% | 1/36 | 21/34 |
| Before, second pipeline type | 5.2% | 8.0% | 12.4% | 2/36 | 20/34 |
| Chosen | 5.2% | 5.5% | 10.6% | 3/36 | 16/34 |

Word error is against the human transcript after cleanup; lower means closer. The three
rejected cleanup blocks changed how a number was written (for example "a hundred. 142
pages" to "a hundred and forty-two pages"). The app then keeps those blocks' source
text. Summary flags were numbers written as digits ("nine billion" as "9 billion") plus
one or two content errors, as in the runs before.

Memory, TED's largest notes job: peak use above idle was 5.6 GB before, and 6.5 GB with
a 1 GB key/value cache. Caches of 2 GB and 4 GB used 7.5 GB and 9.4 GB at the same speed.
On the CPU (fallback), one cleanup level of the talk took 190 s with no stray tokens.
Stop takes effect within 2–4 s, even in the middle of a batch.

**Decision:** on CPU and GPU, the app sends each job's blocks together (notes also send
likely retries together), uses prompt lookup (5 tokens, 3-token match), a 16-bit
key/value cache on the GPU, and a 1 GB cache. It reuses an answer to a repeated request,
such as a summary's part notes for each style. The NPU still sends one request at a
time; it was not measured here.

Not adopted:

- Draft models: slower than prompt lookup, and an extra download.
- A smaller text model: not tried, because accuracy came first.
- Fewer background versions: a feature change, not a speed fix.

Notes were rejected for about half of all blocks, both before and after this change;
see the next section.

## Notes: condensed, with unsaid numbers marked (November 19, 2026)

**Role.** Raw (with timestamps) is for specifics, the cleanup is the text to read or
use as context, and the summary gives the main points. Notes condense the whole
recording into topics and points for reading less than the full text.

**Why the old notes fell back.** On the TED talk, 16 of 34 notes requests failed
checks, and 15 of the 16 had real number problems: dropped details ("roughly 20 amino
acids", "born in 1946") or invented figures ("9 billion by 2040"). The whole block then
fell back to the source text. Two fixes were tried and dropped:

- A repair round that sent each failed block back with its problems recovered 2 of 6
  Mixed blocks on TED and no structured ones, at 9–14 s more per job.
- A layout-only version that copied every word and added headings passed 8 of 9
  blocks, but it was as long as the cleanup; notes are meant to be shorter.

**Condensing approaches tested** on TED and recording 150 (GPU, same model):

| Approach | Length vs cleanup (TED / 150) | TED numbers kept | Time, both | Result |
|---|---:|---:|---:|---|
| Paragraph notes: 1–2 bullets per laid-out paragraph, checked per paragraph | 49% / 25% | 23/24 | 46 s | Distortions; 4 of 39 paragraphs fell back as raw text; lost a section to "None" |
| Key sentences only (model picks sentence numbers) | 45% / 35% | 13/24 | 32 s | Kept sentences lose their referents; still spoken; copying by text failed (model paraphrased) |
| Key sentences, then rewrite | 42% / 15% | 13/24 | 45 s | 5 of 9 rewrites failed checks; new misattributions |
| Abstractive notes per block, unsaid numbers flagged | 24% / 25% | 18/24 | 24 s | Shortest and most readable; distortions comparable |

Judged against a reading of both full transcripts (12 main points for TED, 9 for
recording 150), paragraph and abstractive notes both carried the main ideas: TED 11
and 10 of 12 fully, recording 150 6 and 7 of 9. Both made 2–3 distortions per
recording that no number check catches, such as attributing the speaker's goal to
someone else, stating a question as fact, or reading a remark backwards.

**Decision.** Abstractive notes, with rules added for attribution and for keeping
questions as questions, and at least three points per heading. Dropped details are
allowed. A line giving a number the source never said, in digits or words ("5-7k" for
"five to seven thousand" passes), is marked "⚠ check this number" and the block is
flagged; only an empty, runaway, or overlong answer falls back to the source. Result
in the app (GPU):

| Session, style | Length vs cleanup | Lines marked | Fell back | Time |
|---|---:|---:|---:|---:|
| TED, structured | 33% | 2 (invented "by 2050"; a range, a false mark) | 0/6 | 20 s |
| TED, mixed | 53% | 3 (invented years "2046", "2050") | 0/6 | 29 s |
| Recording 150, structured | 32% | 0 | 0/3 | 15 s |
| Recording 150, mixed | 40% | 0 | 0/3 | 16 s |

The attribution rule fixed a person's background that recording 150's notes had
contradicted, and the TED questions stayed questions; one remark in recording 150 was
still read backwards. The range
false mark ("5,000 to 72,000" for "five to seven thousand") was fixed after this run.
Headings remain frequent (TED structured: 34 for 755 words).

## Audio cuts: fixed 30 seconds, zero overlap

After an initial Yale pilot, three new continuous passages (23:52, 3,583 reference
words) compared zero and one-second overlap twice per model on CPU. Zero overlap
improved all six model/passage comparisons.

| Model | Pooled word error: 1 s → 0 s overlap | Warm batch time: 1 s → 0 s |
|---|---:|---:|
| tiny.en | 12.84% → 8.71% | 66.76 → 53.41 s |
| base.en | 10.41% → 7.51% | 88.46 → 81.11 s |

**Decision:** zero overlap avoids many partial repetitions, but lost eight checked
boundary words that overlap retained. Pause-based cuts had mixed held-out results;
the newer variant worsened base on all three new passages and increased checked
seam deletions from 3 to 13 compared with the earlier variant. Keep fixed cuts
as default and pause variants opt-in. Caption references remain provisional.
These results were for Whisper. Since November 18, the default Parakeet v2 uses
pause-v1 cuts of at most 28 seconds (see the speech section above); the Whisper
models keep fixed 30-second cuts.
[Overlap holdout](../host-validation/evidence/2026-10-01/overlap-holdout/README.md),
[pause holdout](../host-validation/evidence/2026-10-02/pause-refinement/README.md).

## Earlier Whisper evaluations (September 30–October 2, 2026)

Before Parakeet became the default, the Whisper tiny.en and base.en models were
measured on CPU unless noted. Base was more accurate on every set:

| Set | tiny.en | base.en |
|---|---:|---:|
| AMI meetings, 4 clips (8 min) | 38.10% | 28.05% |
| TED-LIUM talks, 3 talks (15.2 min) | 14.55% | 12.35% |
| NPTEL lecture clips, 100 clips | 24.09% | 19.75% |

Two two-hour synthetic CPU runs finished with no sustained memory growth after
warm-up. NPU Whisper needed its pipeline refreshed between cuts to avoid repeated
text, and base.en on NPU ran slower than real time on a matched clip. References
for the AMI, TED-LIUM, and NPTEL sets are published human transcripts that were
not checked by listening. Evidence:
[human-reference pilots](../host-validation/evidence/2026-09-30/manual-reference-pilots/README.md),
[NPTEL pilot](../host-validation/evidence/2026-09-30/lecture-gold-pilot/README.md),
[device comparison](../host-validation/evidence/2026-09-30/device-lecture-evaluation/README.md),
[two-hour endurance](../host-validation/evidence/2026-09-30/two-hour-endurance/README.md),
[continuous Yale passages](../host-validation/evidence/2026-10-01/yale-continuous-pilot/README.md),
[pause pilot](../host-validation/evidence/2026-10-01/pause-pilot/README.md).

## October 3: initial file checkpoint

File transcription, selected automatic cleanup, unedited Raw, and separate
formatted study notes are included. Light requests minimal wording changes;
Medium makes more
clarity edits; Summary selects source passages into headings and bullets,
intentionally omitting detail and treating excerpts independently.
A new freeform summary pilot invented details. This checkpoint instead validates
model-selected passage indices and formats the source text itself. A 72-second
lecture produced 101 words of notes from 192 recognition words, preserving source
and cleanup hashes. Summary completeness and accuracy remain unmeasured.
[Checkpoint checks and limits](../host-validation/evidence/2026-10-03/file-cleanup-checkpoint/report.json)
include real CPU/GPU runs, source hashes, and the installed-wheel check.
At that checkpoint, microphone capture, hotkeys, insertion, and an installer
were deferred. Later dictation and installer checks are linked below.
[User guide](USER_GUIDE.md), [software checks](../DEVELOPMENT.md#checks).


## October 3: dependencies and separate formatting

The October 3 environment used OpenVINO 2026.4.0, GenAI 2026.4.0.0, and
PySide6 6.11.2.

Real CPU/GPU/NPU speech and cleanup paths were exercised on this host. The repeated
72-second CPU speech → GPU cleanup → summary workflow produced identical text to
the earlier checkpoint and preserved source hashes. The short NPU cleanup load
took about 209 seconds, including first compilation. The full device comparison
above used OpenVINO 2026.3; it has not been repeated with 2026.4.

**Decision:** save formatting separately from Raw and AI, with Summary kept as a
separate, intentionally shorter output. Mostly prose uses no model. Mixed and
Mostly structured request layouts from the cleanup model and reuse its loaded
pipeline when chained. The program renders source passages rather than generated
replacement text, verifies complete coverage/order, and falls back to prose on
invalid plans. Tables require explicit comparable source rows; steps require
source step markers. Requests are capped at eight passages after a larger request
pilot omitted passages and fell back to prose.

On the same 72-second transcript, prose formatting took about **0.05 seconds**;
Mixed took **11.88 seconds** over four requests, including model loading; Mostly
structured took **12.96 seconds** with that model already loaded. Mixed accepted
three layouts and fell back on one; Mostly structured accepted two and fell back
on two. Source and cleanup hashes stayed unchanged. Seven of eight additional
short model requests produced valid layouts, including a comparison table and
numbered procedure; one structured comparison returned invalid JSON and became
prose. Layout consistency remains uncertain; headings and grouping need review.
Mostly prose was the default for this comparison.

[Dependency, desktop, installed-package, and formatting checks](../host-validation/evidence/2026-10-03/validation-formatting/report.json)
record the inputs, versions, checks, and limits without personal library paths.


## Full class lecture and final cleanup checks

A supplied **47:20 class video** ran through the complete file workflow on the
same host with OpenVINO 2026.4. Browser and other desktop apps stayed open.
Each speech model ran once in a fixed order.

| Stage | Measured time | Worker CPU, mean / peak | Peak private working memory |
|---|---:|---:|---:|
| tiny speech, CPU | 93 s | 21% / 26% | 0.23 GiB |
| base speech, CPU | 146 s | 22% / 28% | 0.28 GiB |
| Final Light cleanup + prose, GPU | 459 s | 10% / 15% | 4.53 GiB |
| Mixed layout on the initial cleanup | 145 s | 11% / 12% | 4.65 GiB |

Speech times include loading and audio preparation. Cleanup includes loading;
Mixed reused its loaded cleanup model. Memory is the sampled peak across worker
processes; GPU/driver allocations are not fully captured. The base workflow took about
**10.1 minutes** in these separate stage runs;
Mostly prose added no model pass and its complete formatting step took less
than 0.1 seconds. Other apps stayed open, but typing and page-loading response
were not measured. GPU counters again exceeded 100% and were unusable.

**Decisions from this run:** keep CPU speech and AUTO cleanup. Prose added no
model pass. Base improved
several inspected terms and reduced conspicuous repeated text at a cost of
about 52 extra seconds; use it for lectures when the larger download is
acceptable. Without a checked audio reference, the full-lecture error rate
remains unknown. Both models still made recognition errors.

The full run exposed invented sentence endings, an added interpretation/label,
and a lost uncertainty phrase that passed the earlier checks. The final cleanup
prompt and conservative checks cover those observed cases. **9/31 cleanup
blocks retained source text with warnings** in the final run. Cleanup can still
change meaning or retain unclear recognition. Raw stays available. Mixed fell back to prose on **33/78 requests**;
all 45 accepted layouts were paragraphs, so it offered little layout variation
in this lecture despite its added runtime.

Original media and Raw/Balanced hashes were preserved; formatting retained all
cleaned words in order. Raw, AI, and Formatted exports succeeded in text,
Markdown, and JSON. Course text and media remain in ignored local storage, excluded from Git.
[Full lecture measurements and limits](../host-validation/evidence/2026-10-03/full-lecture/report.json).


## Live dictation: latency and idle memory

Dictation reuses CPU speech and GPU-to-CPU cleanup selection. At this checkpoint,
its worker exited after each recording. On a ten-second
clip of a provided speech recording, Off took **2.22 s**. Light cleanup ran twice
per device; the repeat used existing compilation caches:

| Cleanup device | First job | Repeat job |
|---|---:|---:|
| GPU | 19.28 s | 14.47 s |
| CPU | 45.91 s | 53.59 s |
| NPU | 202.50 s | 21.13 s |

These times include conversion, CPU transcription, cleanup setup/generation,
and saving text. NPU's first job spent **190.54 s** in model setup. All six Light
outputs matched. GPU remains the cleanup default with CPU fallback; Off provides
faster dictation when cleanup is unnecessary. The full 266-second file also
completed on CPU: **10.64 s with tiny**, **16.70 s with base**.

These runs used one computer with normal background applications open. WER for
this recording is unmeasured. The
[recording report](../host-validation/evidence/2026-10-03/live-dictation/user-recording-report.json)
contains device timings and insertion checks; the
[initial report](../host-validation/evidence/2026-10-03/live-dictation/report.json)
retains the earlier generated sample and unresolved laptop microphone check.

## Dictation: loaded models and alternative engines

On October 4, three clips (10, 10, and 30 seconds) from the same provided
recording compared fresh workers with workers that kept their models loaded.
Each loaded worker processed the clips three times. Times below cover normalized
audio through CPU tiny transcription and Light cleanup; app startup, recovery
saving, and model checksum checks are excluded. Fresh workers reused compilation
caches. RAM is the worker's working set after its last job.

| Cleanup setup | Fresh worker, 10 s clip | Loaded worker, 10 s clip | Loaded worker, 30 s clip | RAM held | CPU during repeat jobs |
|---|---:|---:|---:|---:|---:|
| Qwen 7B, GPU | 6.79 s | 2.73 s | 6.88 s | 4.77 GiB | 12% |
| Qwen 1.5B, GPU | 3.99 s | 1.15 s | 2.98 s | 1.55 GiB | 15% |
| Qwen 1.5B, CPU | 12.06 s | 2.51 s | 4.66 s | 2.18 GiB | 24% |

These are medians across the relevant clips/calls. Loaded idle CPU was 0–0.1%
over two-second samples. Keeping 7B loaded held about 15% of this computer's
usable RAM. The first job with a new GPU compilation cache took 32.19 s for 7B
and 12.73 s for 1.5B.

The smaller GPU model introduced unsupported interpretations and changed tone
in inspected output. Its CPU run rejected all three versions of one passage
and kept the raw text. The 7B GPU run also retained raw text on two rejected
edits. Repeated GPU cleanup sometimes varied despite identical raw input.
These checks compare edits with the raw text; the recording has no checked
reference for measuring speech accuracy.

For speech alone, loaded OpenVINO tiny INT4 took **0.33 s** on the ten-second
clips and **0.89 s** on the thirty-second clip, holding **0.28 GiB**.
Faster-whisper tiny INT8 with two CPU threads and beam size 1 took **0.49 s**
and **1.09 s**, holding **0.15 GiB**. Beam size 5 took **0.56 s** and **1.44 s**.
Typical worker CPU during repeat jobs was 25% for OpenVINO and 22% for
faster-whisper. Their text differed, so these runs establish no accuracy ranking.
Faster-whisper's file decoder failed with the installed PyAV version; the
comparison used normalized PCM samples for both engines.

The same fixed hardware graphics workload ran twice, with the device order
reversed on the second pass. Its frame rate and the warmed ten-second jobs were:

| Cleanup setup | Graphics before → during jobs | Frame-rate loss | Job time with graphics load |
|---|---:|---:|---:|
| Qwen 7B, GPU | 60.0 → 57.2 fps | 4.6% | 18.36 s |
| Qwen 1.5B, GPU | 59.8 → 58.9 fps | 1.6% | 6.84 s |
| Qwen 1.5B, CPU | 59.6 → 57.2 fps | 4.1% | 4.00 s |

The graphics workload slowed cleanup substantially while losing a few frames
per second itself. Loaded idle workers returned roughly 60 fps. This measures
interference with one graphics workload; it does not give a GPU utilization
percentage or establish responsiveness in other apps.

**Decisions:** retain OpenVINO speech and 7B cleanup. Loaded models offer a
substantial dictation latency reduction at the cost of retained RAM. These
measurements informed later work on the resident worker. The 1.5B model needs better fidelity
before becoming a default.

[Measurements, model revisions, and method](../host-validation/evidence/2026-10-04/dictation-efficiency/report.json).

## Workflow defaults

The October 4 installer used Light cleanup and Mostly structured formatting for
imports, and Medium cleanup with Mostly prose for dictation. The
[installed-app checks](../host-validation/evidence/2026-10-04/installer-checkpoint/README.md#defaults-rebuild)
verified those historical defaults, processing, exports, reinstall, and uninstall.
Current defaults are listed at the top of this page; saved user choices persist.
