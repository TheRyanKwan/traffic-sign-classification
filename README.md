| Week | Main goal | What you should finish |
| :-- | :-- | :-- |
| 1 | Colab startup and data verification | Create the Colab notebook structure, connect a GPU runtime, mount Drive or link GitHub, download GTSRB, inspect file layout, verify labels, and display sample images from multiple classes. |
| 2 | Data pipeline and preprocessing | Build the dataloader, resize images to a fixed size, normalize inputs, test histogram equalization or CLAHE, and add basic augmentation such as rotation, brightness shift, and translation. |
| 3 | Baseline CNN | Train a small CNN end to end in Colab, save checkpoints to Drive, track training and validation metrics, and establish your first reproducible baseline. |
| 4 | Strong backbone | Train ResNet-18 or ResNet-34, compare pretrained versus training from scratch, and report accuracy, precision, recall, and F1-score. |
| 5 | STN integration | Add a Spatial Transformer Network before the backbone and verify that the module trains correctly under the same preprocessing and optimizer settings as the plain ResNet. |
| 6 | Ablation study | Run controlled comparisons for full image versus crop, equalization versus none, augmentation versus none, and ResNet versus ResNet+STN. |
| 7 | Interpretation and error analysis | Generate a confusion matrix, collect common failure cases, and create Grad-CAM or saliency-based visualizations to explain where the model is attending. |
| 8 | Writing and presentation | Finalize the report, clean up figures and tables, prepare slides, summarize findings, and package the final notebook and saved models for submission. |