"""Best-effort W&B logging; local training logs and metric JSONs remain primary."""

from pathlib import Path


class SafeWandbRun:
    def __init__(self, run):
        self.run = run
        self.enabled = True

    def log(self, data, **kwargs):
        if not self.enabled:
            return
        try:
            self.run.log(data, **kwargs)
        except Exception as exc:
            self.enabled = False
            print(f"W&B logging failed ({type(exc).__name__}); local logs are retained.", flush=True)

    def finish(self):
        try:
            self.run.finish()
        except Exception as exc:
            print(f"W&B finish failed ({type(exc).__name__}); local logs are retained.", flush=True)


def start_wandb(output, project, name, config=None, *, evaluation=False):
    try:
        import wandb
    except ImportError:
        print("W&B is unavailable; continuing with local logs.", flush=True)
        return None
    output = Path(output)
    id_path = output / ("wandb_eval_run_id.txt" if evaluation else "wandb_run_id.txt")
    run_id = id_path.read_text().strip() if id_path.is_file() else wandb.util.generate_id()
    id_path.write_text(run_id + "\n")
    kwargs = dict(project=project, name=name, dir=str(output), id=run_id,
                  config=config, allow_val_change=True,
                  settings=wandb.Settings(init_timeout=30))
    try:
        run = wandb.init(**kwargs, resume="allow")
    except Exception as exc:
        print(f"W&B online init failed ({type(exc).__name__}); trying offline logging.", flush=True)
        try:
            run = wandb.init(**kwargs, mode="offline")
        except Exception as offline_exc:
            print(f"W&B offline init failed ({type(offline_exc).__name__}); continuing with local logs.", flush=True)
            return None
    if evaluation:
        try:
            run.define_metric("eval/*", step_metric="eval/epoch")
        except Exception as exc:
            print(f"W&B metric setup failed ({type(exc).__name__}).", flush=True)
    return SafeWandbRun(run)
