### **1. Dexter: Data Pipeline & Augmentation Lead**

- **Focus**: Data loading, class mapping, and preprocessing.
- **Key Responsibilities**:
  - Load and inspect the **Imagewoof dataset** and confirm class order against `classes.txt`.
  - Implement data augmentation pipelines (e.g., random flips, rotations, color jitter, normalization) specifically suited for fine-grained dog breed features like coat texture and ear shape.
  - Format inputs properly into `(N, 128, 128, 3)` arrays for validation testing.

---

### **2. Chase: Model Architecture & Design Lead**

- **Focus**: CNN network structure and layer design.
- **Key Responsibilities**:
  - Construct the custom CNN PyTorch/Keras class from scratch.
  - Start with a baseline architecture (e.g., 4–6 conv layers, stacking \\(3 \times 3\\) convolutions before pooling).
  - Iterate on model depth, spatial dimensions, channel counts, batch normalization, and residual/skip connections to boost feature extraction.

---

### **3. Lance: Training Loop & Optimization Lead**

- **Focus**: Training, hyperparameter tuning, and logging.
- **Key Responsibilities**:
  - Build the main training and validation loop to monitor loss and accuracy over epochs.
  - Implement and log **fixed random seeds** to ensure strict reproducibility of your final model.
  - Experiment with optimizers (e.g., AdamW, SGD with momentum), learning rate schedulers, and weight decay to maximize validation accuracy without overfitting.

---

### **4. Tyler: Integration, Testing & Submission Lead**

- **Focus**: Interface validation, file packaging, and live evaluation prep.
- **Key Responsibilities**:
  - Implement and strictly test the required **`load_model`** and **`predict`** interface functions.
  - Verify that `load_model` loads `model.pth` seamlessly across both **CPU and GPU** environments.
  - Package the final `GroupX_Assignment.zip`, test that `predict` reproduces validation accuracy, and lead the preparation for the Q&A.
