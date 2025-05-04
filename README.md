
# Sampling Bohemian Matrices

This project explores the structure and spectral behavior of **Bohemian matrices** — matrices with bounded integer (or otherwise simple) entries — and visualizes how their eigenvalues evolve under interpolation and random perturbation.

The result is a set of beautiful, structure-rich visualizations that combine linear algebra, randomness, and look nice.

---

## 🌀 What it Does

- Samples and interpolates between hand-crafted Bohemian matrices
- Perturbs specific entries with random complex phases (sampled from the unit circle)
- Visualizes the resulting eigenvalue clouds as still images or high-resolution video
- Supports 4K and 8K frame generation

---

## 📸 Example Output

![Eigenvalue ring cloud](output/frame_001.png)

---

## 🧠 Why?

This project sits at the intersection of:

- **Mathematics**: Bohemian matrices and spectral theory
- **Simulation**: High-volume sampling of random structures
- **Visual aesthetics**: The resulting plots are visually striking and reveal algebraic patterns

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
numpy
matplotlib
opencv-python
joblib
```

---

## 🚀 Usage

### ▶ Generate a static scatterplot:

```bash
python bohemian_matrices_interpolation_w_sampling.py
```

### ▶ Generate a high-resolution frames and video (8K supported):

```bash
python generate_bohemian_video.py
```

Output is saved to:

```
./output/bohemian.mp4
```

Each individual frame is also saved as a `.png` file in the same folder.

---

## 📁 Folder Structure

```
/output            # Rendered frames and final video
generate_frames_and_video.py
requirements.txt
README.md
```

---

## 📚 References

- Bohemian matrices: https://en.wikipedia.org/wiki/Bohemian_matrices
- Eigenvalue visualization techniques


---

## 🧑‍💻 Author

**Lukas Hondrich**  
[Website](https://sites.google.com/view/lukashondrich) · [GitHub](https://github.com/lukashondrich)

---

## 📜 License

MIT License
