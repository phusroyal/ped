# Add conda

### **Step 1: Add Miniconda to Your Path**
You need to add the custom installation directory to your `PATH` environment variable.

1. Open your shell configuration file (`.bashrc`, `.zshrc`, or equivalent):
   ```bash
   nano ~/.bashrc
   ```
   Or for Zsh:
   ```bash
   nano ~/.zshrc
   ```

2. Add the following line at the end of the file:
   ```bash
   export PATH="/home/ubuntu/phusr/miniconda3/bin:$PATH"
   ```

3. Save and exit the editor:
   - In `nano`: Press `Ctrl + O`, then `Enter`, and `Ctrl + X` to exit.

4. Reload the shell configuration to apply the changes:
   ```bash
   source ~/.bashrc
   ```
   Or for Zsh:
   ```bash
   source ~/.zshrc
   ```

---

### **Step 2: Initialize Conda**
Run the following command to initialize Miniconda for your shell:
```bash
/home/ubuntu/phusr/miniconda3/bin/conda init
```

Restart your terminal or reload your shell configuration:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

---

### **Step 3: Verify the Installation**
Check that Conda is properly installed and working:
```bash
conda --version
```

Example output:
```plaintext
conda 23.1.0
```

---

### **Step 4: Update Conda (Optional)**
Make sure you have the latest version of Conda:
```bash
conda update conda
```

---

### **Optional: Clean Up Installation File**
If you haven't removed the installer yet, you can delete it to save space:
```bash
rm ~/Miniconda3-latest-*.sh
```

# Huggingface
pip install -U "huggingface_hub[cli]"# test

hi