"""Deployment operations for the runtime artifact directory.

``scripts/deploy_decision_policy.py`` is the command-line entry point; this
package holds the logic it dispatches to. Modules, from the bottom up:

- ``errors`` / ``paths``: ``DeployError``, the deployment file names and the
  small file helpers (``display``, ``sha256``, ``read_json``, ``now_utc``).
- ``identity``: what a model and a policy are -- manifest interpretation,
  checkpoint fit, class-order, temperature and policy validation, the adapters
  to the backend's own readers, and producer-evidence comparison.
- ``records``: the deployment record's shape. Builds records from values it is
  given and reads historical ones; it never touches a target directory or
  loads a model.
- ``install``: staging, backup, atomic ``os.replace`` install, the post-install
  callback and rollback. It does not know what a model or a policy is.
- ``policy``, ``promote``, ``restore``: one operation each. Each decides what
  is installed and composes ``identity``, ``records`` and ``install``.
- ``verify_live``: checks a running API against a target. It imports no write
  operation.

The backend (``app.backend``) never imports this package; this package reads
the target through the backend's readers so it checks what the runtime reads.

Failure-injection seams: the operations call ``identity`` functions through the
module (``identity.verify_through_backend(...)``), so a test that patches
``app.deployment.identity.verify_through_backend``, ``verify_model_files`` or
``build_model`` reaches every operation. The backup timestamp is read through
``app.deployment.install.datetime``.
"""
