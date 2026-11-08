# KEVexploit demo video: voice-over

Runs 1:52. Read at a calm, confident pace. Each line fits its window with a little room to spare. Times are positions in the finished video (`kevexploit_demo.mp4`), so you can record in one take against it, or line by line and drop each clip at its start time.

| Start | End | On screen | Voice-over |
|---|---|---|---|
| 0:00 | 0:05 | A galaxy of points bursts out; the counter runs to 72,670 | "The problem is not finding vulnerabilities. The problem is deciding what deserves attention before it becomes a breach." |
| 0:05 | 0:09 | "Attackers will use a few dozen. Which ones?" | "With thousands of CVEs arriving every year, a severity score alone leaves security teams buried in noise." |
| 0:09 | 0:14 | Points rise by risk and turn hot | "KEVexploit is our early-warning and prioritization engine: it turns that noise into an evidence-based patch queue." |
| 0:14 | 0:22 | Everything else sinks; the 500 pull into a red cluster | "Across more than seventy-two thousand vulnerabilities, it identifies the five hundred most likely to matter to attackers." |
| 0:22 | 0:28 | The whole pipeline, with data flowing along the wires | "The key is not one magic model. It is a full machine-learning pipeline, built to be accurate, explainable, and useful in production." |
| 0:28 | 0:36 | Point-in-time practice data | "We use point-in-time features and strict data-leakage controls, so the model only sees information that would have existed when a CVE was published." |
| 0:36 | 0:45 | Five kinds of clues; IBM Granite 30B with its questions orbiting | "That matters because without it, a model can overfit to the future and look brilliant in a demo. We use regularization, feature engineering, and IBM Granite to capture exploitability, technical language, and attacker appeal." |
| 0:45 | 0:53 | Three judges and the Venn | "Purged, time-blocked cross-validation tests whether those signals generalize. Then a rank ensemble combines tree models, logistic regression, and a fine-tuned Granite voter instead of trusting a single algorithm." |
| 0:53 | 1:00 | Two final filters | "Finally, Granite-informed filters remove low-impact noise, while a diversity cap prevents one vendor or assigner from taking over the list." |
| 1:00 | 1:08 | Proof, with watsonx.governance | "On held-out years, KEVexploit finds sixty-five percent more exploited vulnerabilities than the standard ranking. With watsonx.governance, the system is also designed for monitoring, drift detection, and accountable AI." |
| 1:08 | 1:15 | Top 500 treemap | "So the output is not a black box. It is a defensible Top 500, with evidence and plain-language reasons behind every priority." |
| 1:15 | 1:22 | The camera drops into the live scanner | "And this same intelligence moves from the benchmark into the software teams actually run." |
| 1:22 | 1:30 | A real GitHub repo is scanned step by step | "Connect a GitHub repository, resolve its exact dependency versions, map advisories to CVEs, and enrich each finding with exploit and threat intelligence." |
| 1:30 | 1:45 | Results: an exploited Pillow flagged at the top, then a glide down the ranked list and the upgrade plan | "Here, an exploited Pillow dependency rises immediately. Instead of handing developers another alert dump, KEVexploit shows the likely impact and the upgrade path, so remediation starts with the risk that matters most." |
| 1:45 | 1:52 | End card | "KEVexploit: turning vulnerability overload into explainable, intelligence-led action—before the breach." |

## Notes

- **Tone:** Sound direct, confident, and human, like explaining a product to a security leader. This is one continuous story, not a list of slide captions. Let the transitions carry the pitch: problem, credibility, differentiation, proof, and impact.
- **Stressed words:** *before it becomes a breach*, *data leakage*, *overfit*, *generalize*, *sixty-five percent*, *explainable*, *before the breach*.
- **Numbers on screen:** say "five hundred" and "sixty-five percent" in full; the screen shows the digits.
- **Pronunciation:** KEVexploit is "kev-exploit" (KEV like Kevin); CVE is "C-V-E"; Granite is just "Granite".
- **If a line runs long:** each row's end time is a soft limit, and the visuals hold for about a second past it.
