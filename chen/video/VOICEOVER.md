# KEVexploit demo video: voice-over

Runs 1:52. Read at a calm pace, about 150 words a minute. Each line fits its window with a little room to spare. Times are positions in the finished video (`kevexploit_demo.mp4`), so you can record in one take against it, or line by line and drop each clip at its start time.

| Start | End | On screen | Voice-over |
|---|---|---|---|
| 0:00 | 0:05 | A galaxy of points bursts out; the counter runs to 72,670 | "In 2026, seventy-two thousand new vulnerabilities were published." |
| 0:05 | 0:09 | "Attackers will use a few dozen. Which ones?" | "Attackers will use a few dozen of them. Which ones?" |
| 0:09 | 0:14 | Points rise by risk and turn hot | "We score every bug using only what was public the month it came out." |
| 0:14 | 0:22 | Everything else sinks; the 500 pull into a red cluster | "Then we pick the five hundred a security team should patch first." |
| 0:22 | 0:28 | The whole pipeline, with data flowing along the wires | "Here's the machine that does it: seven stages, from raw records to a ranked list." |
| 0:28 | 0:36 | Slide 2: point-in-time practice data | "We train on older bugs exactly as they looked when they came out, so the model never sees the future." |
| 0:36 | 0:45 | Slide 3: five kinds of clues; IBM Granite 30B with its questions orbiting | "For each bug we collect exploit signals, the description and the publisher's track record. IBM Granite reads it like a security analyst." |
| 0:45 | 0:53 | Slide 4: three judges and the Venn | "Three different models rank every bug, including a fine-tuned Granite, and we average their votes." |
| 0:53 | 1:00 | Slide 5: two final filters | "Two rules clean up the list: drop bugs nobody would bother attacking, and cap any one vendor." |
| 1:00 | 1:08 | Slide 6: the proof, with watsonx.governance | "On held-out years it finds sixty-five percent more exploited bugs than the standard ranking, and watsonx.governance watches it for drift." |
| 1:08 | 1:15 | Slide 7: the Top 500 treemap | "The result: five hundred bugs, each with a plain-language reason." |
| 1:15 | 1:22 | The camera drops into the live scanner | "And because every CVE has a score, it works on your own code." |
| 1:22 | 1:37 | A real GitHub repo is scanned step by step | "Point it at a GitHub repo. It reads your lockfiles, finds every known vulnerability in your exact versions, and scores each one with our model." |
| 1:37 | 1:45 | Results: an exploited Pillow flagged at the top, then the ranked list | "Already-exploited bugs go straight to the top, then the rest in the order to patch." |
| 1:45 | 1:52 | End card | "KEVexploit. Patch these five hundred first." |

## Notes

- **Stressed words:** *seventy-two thousand*, *Which ones?*, *sixty-five percent*, *your own code*.
- **Numbers on screen:** say "five hundred" and "sixty-five percent" in full; the screen shows the digits.
- **Pronunciation:** KEVexploit is "kev-exploit" (KEV like Kevin); CVE is "C-V-E"; Granite is just "Granite".
- **If a line runs long:** each row's end time is a soft limit, and the visuals hold for about a second past it.
