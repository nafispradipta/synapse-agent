# Dokumentasi Pengguna

Dokumen ini menjelaskan `synapse-cortex` tanpa istilah teknis yang rumit. Cocok untuk pemilik produk, tim operasional, atau siapa pun yang ingin memahami manfaatnya sebelum memakai.

## Singkatnya

Synapse adalah buku ingatan jangka panjang untuk asisten AI.

Asisten AI biasanya lupa semua yang terjadi setelah percakapan selesai. Synapse menyimpan hal-hal penting ke dalam satu file di komputer Anda, lalu mengambilnya kembali saat dibutuhkan. Hasilnya, asisten bisa mengingat keputusan, aturan kerja, pengalaman gagal, dan preferensi Anda dari sesi ke sesi.

## Masalah yang diselesaikan

Bayangkan tiga situasi ini:

1. Anda sudah memberi tahu asisten bahwa server produksi memakai port tertentu. Besoknya asisten bertanya lagi.
2. Catatan lama mengatakan tugas "sedang dikerjakan", catatan baru mengatakan "selesai". Sistem pencarian biasa bisa mengambil keduanya sekaligus dan membuat asisten salah paham.
3. Anda menempelkan seluruh riwayat percakapan ke prompt. Biayanya mahal dan perhatian model jadi terpecah.

Synapse menjawab ketiganya: menyimpan fakta secara ringkas, menandai fakta lama sebagai tidak berlaku, dan hanya mengirim konteks yang muat di anggaran token.

## Apa yang bisa disimpan

- Fakta dan aturan, misalnya arsitektur sistem atau standar operasional.
- Keputusan, misalnya pilihan database atau alasan memilih sebuah pendekatan.
- Preferensi, misalnya gaya penulisan atau pengaturan favorit.
- Insiden dan pelajaran, misalnya apa yang menyebabkan gangguan dan cara memperbaikinya.
- Pengetahuan gabungan, misalnya ringkasan topik yang terus diperbarui.
- Catatan riset dari web yang dikarantina sampai Anda setujui.

## Cara kerja dalam bahasa sederhana

Saat informasi disimpan, Synapse membuat tiga hal: ringkasan satu baris, ringkasan agak panjang, dan catatan lengkap. Synapse juga membuat "benang penghubung" ke informasi lain yang mirip atau terkait.

Saat informasi dicari, Synapse memakai beberapa cara sekaligus: mencocokkan kata, mencocokkan makna, menelusuri benang penghubung, dan memprioritaskan yang masih relevan. Hasilnya dirapikan agar muat di dalam batas panjang yang Anda tentukan.

Secara berkala, Synapse merapikan diri: memperkuat ingatan yang sering dipakai, melemahkan yang sudah lama tidak dipakai, dan menggabungkan catatan lama menjadi ringkasan.

## Cara memulai

### Jika Anda memakai asisten yang mendukung MCP

Minta bantuan pengembang atau admin untuk menambahkan konfigurasi berikut ke aplikasi asisten Anda:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "synapse-mcp",
      "env": {
        "SYNAPSE_DB_PATH": "/lokasi/ingatan/synapse.db"
      }
    }
  }
}
```

Setelah aktif, asisten bisa menyimpan dan mencari ingatan melalui empat perintah: mencari ingatan, menyimpan ingatan, mencatat pengalaman, dan melihat statistik.

### Jika Anda memakai Python

Instal paketnya:

```bash
pip install synapse-cortex
```

Versi yang tersedia di PyPI bisa tertinggal dari versi terbaru di repositori. Jika fitur yang Anda butuhkan belum ada di PyPI, pasang dari sumber.

Lalu jalankan contoh sederhana yang tidak membutuhkan koneksi internet atau kunci API:

```bash
python examples/quickstart.py
```

## Apakah perlu koneksi internet atau kunci API

Tidak untuk fungsi dasar. Menyimpan dan mencari ingatan bekerja sepenuhnya di komputer Anda.

Koneksi internet hanya diperlukan jika Anda ingin:

- Memakai model bahasa untuk merangkum dan memperbaiki ingatan secara lebih pintar.
- Memakai layanan embedding daring agar pencarian makna lebih akurat.
- Mengambil catatan riset dari situs yang sudah diizinkan.

Semua fitur itu opsional. Jika tidak tersedia, Synapse memakai cara bawaan yang lebih sederhana tetapi tetap berjalan.

## Privasi dan keamanan

- Data disimpan di file SQLite di komputer Anda. Tidak ada server pusat.
- Jika Anda mengatur layanan model bahasa daring, isi ingatan yang dikirim untuk diproses akan keluar dari komputer Anda. Gunakan model lokal jika data tidak boleh keluar.
- Database dan catatan Markdown tidak dienkripsi secara bawaan. Lindungi dengan izin folder, kata sandi komputer, atau enkripsi disk.
- Secara bawaan, semua proyek di komputer yang sama bisa berbagi satu database di `~/.synapse/synapse.db`. Server MCP memakai lokasi `./data/synapse.db` lebih dulu jika file itu sudah ada. Gunakan lokasi terpisah jika ingin memisahkan ingatan antar proyek.
- Catatan hasil riset web disimpan dalam status karantina dan tidak ikut muncul di pencarian sampai Anda menyetujuinya.
- Jangan menyimpan kata sandi, kunci API, atau data pribadi sensitif ke dalam ingatan.

## Batasan yang perlu diketahui

- Synapse bukan mesin kebenaran. Ia menyimpan apa yang diberikan, ditambah penilaian keyakinan dan deteksi pertentangan.
- Skala yang nyaman adalah sekitar sepuluh ribu catatan aktif. Ini bukan pengganti database vektor skala besar.
- Pencarian makna bawaan hanya cocok untuk kemiripan kata. Untuk parafrase dan beda bahasa, pasang model embedding.
- Database hanya nyaman untuk satu penulis pada satu waktu. Penulisan bersamaan akan menunggu antrean.
- Kualitas ingatan bergantung pada apa yang dipilih untuk disimpan.

## Pertanyaan yang sering muncul

**Apakah ingatan saya otomatis dikirim ke internet?**
Tidak, kecuali Anda mengonfigurasi layanan model atau embedding daring.

**Bisakah saya menghapus semua ingatan?**
Bisa. Hapus file database dan salinannya. Catatan Markdown di folder vault juga perlu dihapus terpisah.

**Bisakah beberapa asisten AI berbagi ingatan yang sama?**
Bisa, jika mereka menunjuk ke file database yang sama. Gunakan file berbeda untuk memisahkan proyek.

**Apa yang terjadi jika tidak ada kunci API?**
Fungsi utama tetap berjalan. Perangkuman tingkat lanjut memakai aturan bawaan.

**Apakah data aman jika komputer dipakai bersama?**
Database tidak terenkripsi. Batasi izin folder atau gunakan enkripsi disk.

**Apakah Synapse bisa dijalankan di Windows?**
Bisa. Pustaka dan server MCP berjalan di Windows. Skrip konsolidasi malam hari lebih cocok untuk Linux karena memanggil interpreter dengan jalur tetap.

## Status saat ini

Versi paket: `1.1.1`. Rincian teknis dan riwayat perubahan ada di [CHANGELOG.md](../CHANGELOG.md) serta [Status.md](Status.md).
