# Chemical EMI Designer — Evaluation guide

Start with the smallest path that exercises the project. Distinguish source inspection,
syntax/build checks, functional behavior and domain validation when recording a result.

## Guided reading and demonstration

1. **Compose the input.** Choose elements and quantities or a molecular preset. Inspect the
material-property assumptions used for the assembled formula.

2. **Set the physical conditions.** Frequency, thickness, conductivity, permeability and
permittivity determine the modeled response. Units and material provenance are part of the input,
not cosmetic labels.

3. **Calculate shielding terms.** The physics module calculates impedance, propagation, skin depth
and shielding components. Read each term before interpreting total shielding.

4. **Compare and challenge.** Use frequency or thickness exploration to see how the model changes.
Compare against measured material properties and laboratory data before drawing an engineering
conclusion.

## Declared checks

These commands/checks describe the intended verification path. Their presence in this
guide does not claim that they passed. See the dated evidence below and the README for setup.

```text
python -m compileall -q src streamlit_app
```

## Evidence levels

| Level | What it establishes | What it does not establish |
| --- | --- | --- |
| Source review | A path exists in tracked code | Successful runtime behavior |
| Syntax/build | Parser/compiler accepts that path | End-to-end correctness |
| Behavioral check | A specific input/output case passed | Generalization beyond cases |
| Domain evaluation | Performance on a stated target setting | Other users/data/environments |

## What to record

- Commit, environment, dependency versions and date.
- Input provenance and whether data is synthetic, public or privately supplied.
- Absolute pass/fail/skip counts; keep failed cases and their root causes.
- Whether external services, hardware or a production deployment were actually exercised.
- Expected output and an artifact showing the observation.

## Review scenarios

- **Separate property and physics layers:** Formula handling does not hide the physical inputs to
shielding calculations.

- **Expose shielding components:** Reflection, absorption and multiple-reflection terms can be
inspected independently.

- **Explicit model limits:** A weighted property estimate cannot replace measured behavior of a
synthesized material.

## Documentation inspection — 7 October 2026

The documentation was traced to committed source and checked for local links, balanced
code fences and supported implementation claims. Historical notebook outputs remain labeled
as historical. Live provider access, private databases and hardware behavior are not inferred
from configuration or dependency files. Any fresh run is recorded separately in the README.

## Next evidence to collect

- Add measured material-property provenance.
- Benchmark equations against independent reference cases.
- Validate numerical limits and experimentally measured shielding.
