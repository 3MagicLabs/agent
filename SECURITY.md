# Security Policy

## Reporting a vulnerability

Report privately through
[GitHub Security Advisories](https://github.com/3MagicLabs/agent/security/advisories/new).
Please do not open a public issue for a vulnerability.

We aim to acknowledge within 3 business days and to ship a fix or a mitigation
plan within 14 days.

## Supported versions

`main` is the only supported branch while the project is pre-1.0.

## Threat model

This project runs a language model that chooses which tools to call. The model's
output is untrusted input, and so is anything it fetches from the web.

| Surface | Risk | Mitigation |
|---|---|---|
| `python_repl` | Model-authored code execution | Runs only in an E2B microVM, never on the host. Disabled entirely when `E2B_API_KEY` is unset. |
| `scrape_webpage` | SSRF into internal networks | Scheme allowlist (`http`/`https` only), fixed timeout, no redirect to non-HTTP schemes. |
| `read_file` | Path traversal out of the download directory | Paths are reduced to their basename and re-resolved under the download directory; anything else is refused. |
| `download_task_file` | Writing outside the sandbox | Filenames are derived from the task ID, never from attacker-controlled headers. |
| Prompt injection | Scraped content instructing the model | Budgets bound blast radius; tools are capability-scoped. **Not fully mitigated** — see below. |
| Credentials | Leaking keys into logs or traces | Keys live only in `Settings`; `agent doctor` prints booleans, never values. |

### Known limitations

- **Prompt injection is not solved.** A malicious page can influence the
  model's next tool call. Do not point this agent at untrusted content while
  holding credentials you care about.
- **The sandbox is only as strong as E2B.** Treat any code the model writes as
  hostile.
- **No egress filtering.** The agent can reach any public URL.

## Secrets

Never commit credentials. Required secrets are listed in `.env.example` and are
read from the environment only. Secret scanning and push protection are enabled
on this repository; if you believe a secret was exposed, rotate it first and
report second.
