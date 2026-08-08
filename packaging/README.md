# Realtime Windows packages

Build the event-driven recognition-only OCR package:

```powershell
.\packaging\build_realtime.ps1 -Variant OCR -ModelPath .\best.pt -SourcePath .\test.mp4
```

Build the YOLO-only performance package:

```powershell
.\packaging\build_realtime.ps1 -Variant YOLO -ModelPath .\best.pt -SourcePath .\test.mp4
```

The OCR package embeds only `en_PP-OCRv5_mobile_rec` from
`$HOME\.paddlex\official_models` by default. OCR runs in a low-priority child
process after event evidence is saved; the package does not include the server
text-detection model. Override the model root with `-OcrModelsPath` when
preparing an offline target-computer package.
