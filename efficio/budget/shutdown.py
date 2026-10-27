"""Release the GPU runtime when training is done: Run(shutdown_when_done=True).

Only Colab can be stopped from code. Efficio releases it only after the final checkpoint is
confirmed in storage; the Run class checks that before calling release_colab_runtime().
"""

NOT_SUPPORTED = {
    "kaggle": ("Kaggle sessions can't be stopped from code. Stop the session yourself when "
               "training ends, or run the notebook with Save Version, which ends by itself."),
    "local": "nothing to shut down on a local machine.",
}


def startup_notice(platform, has_storage):
    """The line logged when the Run is created."""
    if platform != "colab":
        reason = NOT_SUPPORTED.get(platform, NOT_SUPPORTED["local"])
        return f"[efficio] shutdown_when_done: {reason}"
    if not has_storage:
        return ("[efficio] shutdown_when_done: no storage is set, so the runtime will NOT be "
                "released (the checkpoint would be lost with it). Set storage= to use this.")
    return ("[efficio] shutdown_when_done: after run.report() the Colab runtime is released, "
            "but only if the final checkpoint was uploaded. Code after run.report() won't run.")


def release_colab_runtime(log=print):
    """Disconnect and delete the Colab runtime, so it stops using GPU quota."""
    from google.colab import runtime
    log("[efficio] shutdown_when_done: releasing the Colab runtime now")
    runtime.unassign()
