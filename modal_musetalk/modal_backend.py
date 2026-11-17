"""Real Modal client. Importing this module does not import Modal.

`build_backend()` imports Modal and modal_app. Call it only after the spend
gate is open. It deploys the named app only when that app is missing, then
spawns through the returned object. Recovery uses FunctionCall.from_id and
does not spawn.
"""

from __future__ import annotations

from modal_musetalk.spec import APP_NAME, FUNCTION_NAME


class ModalBackend:
    def __init__(self, modal, app, function) -> None:
        self._modal = modal
        self._app = app
        self._function_object = function

    def ensure_function(self):
        try:
            function = self._modal.Function.from_name(APP_NAME, FUNCTION_NAME)
            function.hydrate()
            return function
        except self._modal.exception.NotFoundError:
            with self._modal.enable_output():
                self._app.deploy()
            function = self._modal.Function.from_name(APP_NAME, FUNCTION_NAME)
            function.hydrate()
            return function

    def spawn(self, function, spawn_args: tuple):
        return function.spawn(*spawn_args)

    def from_id(self, call_id: str):
        return self._modal.FunctionCall.from_id(call_id)


def build_backend() -> ModalBackend:
    import modal
    import modal_app

    return ModalBackend(modal, modal_app.app, modal_app.infer_once)
