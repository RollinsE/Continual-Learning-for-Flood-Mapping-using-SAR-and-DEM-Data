# Flood Extent Mapping deployment bundle

This directory is self-contained: the manifest refers only to files inside this
bundle.  Move or copy the entire directory together.

Run prediction after installing the matching Flood Extent Mapping package:

```bash
floodmap predict-scene   --manifest deployment_manifest.yaml   --sar-path /path/to/vv_vh_scene.tif   --output-dir outputs   --write-probability   --write-html-report
```

The `deployment_bundle.json` inventory records file sizes and SHA-256 hashes for
the bundled model assets.
