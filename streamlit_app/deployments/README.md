# Deployment catalogue

Place each portable bundle exported by `floodmap export-deployment` in its own subdirectory:

```text
streamlit_app/deployments/<model-name>/deployment_manifest.yaml
```

Keep the complete exported `assets/` tree beside the manifest. Deployment checkpoints under this directory are configured for Git LFS; verify them with `git lfs ls-files` before pushing a public Streamlit deployment.

No trained checkpoint is included in the source release.
