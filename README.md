# Tugas Besar 1 IF3270 - Feedforward Neural Network From Scratch

Repository ini berisi implementasi FFNN from scratch untuk Tugas Besar 1 IF3270 Pembelajaran Mesin.

## Struktur Repository

- `src/activation_function.py`: implementasi fungsi aktivasi dan turunannya.
- `src/loss_function.py`: implementasi loss function dan turunannya.
- `src/initialization.py`: metode inisialisasi bobot.
- `src/layer.py`: komponen Dense layer.
- `src/rmsnorm.py`: implementasi bonus RMSNorm.
- `src/model.py`: implementasi kelas FFNN (forward, backward, train, save/load, plotting distribusi).
- `src/ffnn_model.py`: compatibility import untuk kelas `FFNN`.
- `src/notebook.ipynb`: notebook eksplorasi/eksperimen.
- `src/pengujian.ipynb`: notebook pengujian utama sesuai spesifikasi.
- `data/datasetml_2026.csv`: dataset yang digunakan.

## Cara Menjalankan

1. Buat environment Python (disarankan Python 3.10+).
2. Install dependency:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

3. Jalankan notebook:

```bash
jupyter notebook
```

4. Buka `src/pengujian.ipynb` untuk menjalankan seluruh eksperimen.

## Fitur Utama FFNN

- Arsitektur fleksibel berdasarkan `layer_sizes`.
- Fungsi aktivasi:
	- `linear`
	- `relu`
	- `sigmoid`
	- `tanh`
	- `softmax`
	- bonus: `elu`, `swish`
- Loss function:
	- `mean_squared_error`
	- `binary_cross_entropy`
	- `categorical_cross_entropy`
- Inisialisasi bobot:
	- `zero`
	- `uniform` (dengan lower/upper bound, seed)
	- `normal` (dengan mean/variance, seed)
	- bonus: `xavier`, `he`
- Mendukung training batch.
- Backpropagation menggunakan chain rule.
- Regularisasi `L1` dan `L2`.
- Update bobot dengan gradient descent.
- Menyimpan histori `train_loss` dan `val_loss` per epoch.
- Visualisasi distribusi bobot dan gradien bobot per layer.
- Save/load model dengan pickle.

## Checklist Eksperimen Wajib

Gunakan `src/pengujian.ipynb` untuk menunjukkan:

1. Pengaruh depth dan width (3 variasi width, 3 variasi depth).
2. Pengaruh fungsi aktivasi hidden layer (kecuali softmax).
3. Pengaruh learning rate (3 variasi).
4. Pengaruh regularisasi (tanpa regularisasi, L1, L2).
5. Perbandingan dengan sklearn MLP menggunakan hyperparameter yang sama.

Untuk setiap eksperimen (selain uji perbandingan sklearn), tampilkan:

- hasil akhir prediksi,
- grafik training loss dan validation loss,
- distribusi bobot dan gradien bobot beberapa layer.