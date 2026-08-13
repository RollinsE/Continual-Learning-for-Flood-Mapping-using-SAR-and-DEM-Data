# CLI logging and progress

Flood Extent Mapping uses one immediate logging policy for preprocessing, training,
continual training, evaluation, deployment and audit commands.

## Direct CLI execution

Run pipeline operations directly with `floodmap`:

```bash
floodmap <command> [command options] --plain-progress
```

Every command logs its resolved command line, startup, meaningful progress,
completion time, and failures with tracebacks. Console output is unbuffered in
Colab and the same run also writes a persistent log where an output location is
available.

## Clean Colab progress

`--plain-progress` is recommended in Colab. A finite training or validation
pass prints:

- a start record;
- roughly one progress record per 10% of the pass;
- a completion record;
- the full epoch metrics and checkpoint/early-stopping decision.

The progress record includes the latest loss and learning rate. It no longer
prints every 2–3% of a long epoch.

`--dynamic-progress` retains a live carriage-return tqdm bar for a suitable
interactive terminal.

## Heartbeats

`--heartbeat-seconds N` controls liveness messages during genuinely quiet work.
A heartbeat is suppressed whenever a batch update, validation update, warning,
or other meaningful log record has appeared within the last `N` seconds. This
prevents `Command running` lines from being interleaved with already-active
training progress.

Set `--heartbeat-seconds 0` to disable command heartbeats entirely.

## Threshold sweeps

During training, evaluation and ensemble evaluation, the console prints one
compact best-threshold line containing F1, IoU, precision, recall, MCC,
empty-scene false-positive rate, and non-empty tile recall.

The complete threshold table is still written to the persistent `output.log`
for auditability.

## Persistent logs

Default locations include:

```text
train / continual-train     <artifacts-dir>/<run-id>/output.log
preprocess                  <processed-data-dir>/preprocess.log
audits with --output-dir    <output-dir>/output.log
commands with --output-file <output-file>.log
```

An existing log receives a `New command session` separator before appended
records, which makes reconnects and resumed runs easy to distinguish.

## Training records retained

Training still shows:

- startup and resolved configuration;
- dataset and sampler preparation;
- train and validation progress;
- non-finite/AMP recovery warnings;
- end-of-epoch loss, F1, IoU, precision, recall and MCC;
- best-threshold summary;
- checkpoint improvements and saved paths;
- early stopping and resume state;
- interruption and completion messages.
