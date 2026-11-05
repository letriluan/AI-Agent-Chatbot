# Project 1 completion audit — 5 November 2026

All six agreed roadmap deliverables have evidence. The second schema uses the user-approved synthetic transit fixture. Completion describes the portfolio deliverables, not a production uptime or general accuracy guarantee.

| Deliverable | Evidence and limits |
| --- | --- |
| CI-gated deployment | [Render rollout](../render-fixed-deployment-2026-11-05/REPORT.md): main CI passed before Auto-Deploy of runtime commit `5df7de9`; Blueprint Auto Sync paused. No deliberately failing main commit was introduced. |
| Five holdout runs | [Fixed study](../reliability-fixed-live-2026-11-05/REPORT.md): 80/80 SQL, all 16 cases pass all five runs; 20/20 chart checks. Repeated exposed synthetic cases do not establish general accuracy. |
| Second schema | [Fixed transit study](../transit-fixed-live-2026-11-05/REPORT.md): 24/24 SQL, 4/4 charts; S4 lookup returns R2. Synthetic data, not a real Adelaide Metro evaluation. |
| Numeric support | Holdout 71/75 scored (94.7%); transit 16/17 (94.1%). Five context flags reviewed; 12 observations unscored. Full narrative factuality remains unmeasured. |
| Uptime monitoring and cold start | External automatic detection and recovery verified below; original controlled cold start **32.85 s**, independently confirmed with Render lifecycle logs. |
| Dated cost estimates | [Fixed cost estimates](../cost-estimates-fixed-2026-11-05.json): known usage subtotal USD 0.024925050 across 99/104 observations. Five recovered requests have missing usage; no full-suite mean or measured bill is claimed. |

## Automatic monitoring verified

UptimeRobot Free monitor `804558877`, **SQL Agent demo - health**, checks `https://ai-agent-chatbot-demo.onrender.com/health` using GET and the case-sensitive keyword `"status":"ok"`. Missing keyword starts an incident. Saved interval is **30 minutes**, timeout **60 seconds**. The user explicitly chose 30 minutes to conserve Render Free hours. Default account email alerts remain enabled; no paid upgrade was made. These checks never call Groq.

The provider's [first incident](https://dashboard.uptimerobot.com/incidents/376635194058224841) was detected automatically at **18:03:02**, confirmed at **18:03:47**, and **resolved by Europe at 18:08:52** (5m5s confirmed duration), on 5 November 2026, Australia/Adelaide (UTC+10:30). Its activity log records successful incident and recovery email dispatch. No manual probe or incident resolution intervened in that first recovery. See the archived recovery screenshot and text.

A second incident (`376644232005708366`) started at 18:39:42 and resolved at 20:56:51 (2h17m9s). A local no-model diagnostic at **20:51:54.977** obtained healthy JSON after **32,907.231 ms**, then ready JSON in **297.625 ms**. That diagnostic woke the service before the provider's subsequent recovery; this second recovery is wake-assisted, not independent evidence of unassisted recovery. Its raw `cold_start_confirmed: false` is preserved because no independent platform lifecycle logs were collected for this later probe.

The dashboard was subsequently observed **Up**, with both resolved incidents retained. The short-window dashboard uptime was **18.840%**; it is not a sustained availability result. Recorded 503s included `X-Render-Routing: hibernate-wake-error`. Thirty-minute checks allow the Free service to sleep and can expose wake failures. The monitor is operational and records these failures; always-on reliability has not been established. Longer history remains useful.

## GitHub diagnostic role

GitHub runs `41976127859` and `41979530248` both succeeded, but both were manually dispatched. No scheduled event was observed, including after the cron refresh. UptimeRobot owns the external automatic schedule. The GitHub workflow now retains only `workflow_dispatch` for on-demand `/health` and `/ready` diagnosis, avoiding duplicate scheduled traffic. Historical reports remain unchanged.

## Evidence integrity and scope

`manifest.json` records SHA-256 hashes of every other file in this archive. Settings text excludes the account email address. Screenshots and provider activity text preserve the observed automatic incident history. No API keys or passwords are included. This monitoring follow-up made **zero Groq calls** and did not change application runtime code or the deployed Render credential. All model studies used the user-authorized Free-only collection protocol; earlier baseline and quota-stopped results remain separate.
