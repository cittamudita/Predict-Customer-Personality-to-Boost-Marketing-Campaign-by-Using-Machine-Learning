## Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning
Sebuah perusahaan dapat berkembang dengan pesat saat mengetahui perilaku customer personality nya, sehingga dapat memberikan layanan serta manfaat lebih baik kepada customers yang berpotensi menjadi loyal customers. Dengan mengolah data historical marketing campaign guna menaikkan performa dan menyasar customers yang tepat agar dapat bertransaksi di platform perusahaan, dari insight data tersebut fokus kita adalah membuat sebuah model prediksi kluster sehingga memudahkan perusahaan dalam membuat keputusan

#### Conversion rate analysis based on income, spending and age
Analisis conversion rate merupakan suatu pencarian insight data persentase pengunjung website serta tindakan apa saja yang mereka lakukan selama mengunjungi situs, dan apakah tindakan mereka menghasilkan transaksi pembelian atau tidak selama berkunjung di website tersebut, hal ini dapat dilakukan dengan melakukan feature engineering pada variable data yang tersaji, sehingga dapat menghasilkan satu kolom baru yaitu Conversion rate.

![1](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-1.jpeg)
- Pendapatan yang lebih tinggi, menunjukkan tingkat konversi yang lebih tinggi. Dengan tingkat konversi tinggi maka total purchases juga tinggi. Dengan demikian kita bisa menargetkan kampanye kepada pelanggan dengan pendapatan yang lebih tinggi.
- Untuk pelanggan dengan pendapatan lebih rendah, kita dapat menjelajah lebih jauh dengan menggunakan teknik pengelompokan untuk mengelompokkan mereka ke dalam cluster dan memberikan rekomendasi yang disesuaikan dengan masing-masing cluster, yang berpotensi meningkatkan jumlah transaksi di platform kita.
![2](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-2.jpeg)

- Semakin tinggi pendapatan seorang pelanggan, semakin tinggi pula tingkat konversinya. Temuan ini didukung insight lain, di mana peningkatan pendapatan berhubungan dengan total pengeluaran yang lebih tinggi di platform, dan terdapat hubungan linier antara total pengeluaran dan tingkat konversi.
- Namun pada feature umur, baik untuk grup umur young adult, adult, dan senior adult tidak memiliki hubungan yang spesifik dengan conversion. Tidak ada umur spesifik yang memiliki conversion rate yang tinggi.



#### Data Cleaning & Preprocessing
Data Cleaning & Preprocessing Flow
![3](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-3.jpeg)

#### Data Modeling
#####  Elbow Method menggunakan K-Means Clustering
![4](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-4.jpeg)

-  Dalam gambar di atas, kita dapat melihat bahwa ketika n_cluster = 4, skor inersia tidak mengalami perubahan yang signifikan, sehingga kita akan menggunakan n_cluster = 4. n=4 dimana perbedaan inertia dengan menambah cluster baru berkurang drastis

##### Evaluasi menggunakan Silhouette Score
![5](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-5.jpeg)

-  Dari gambar di atas, kita dapat melihat bahwa ketika n_cluster = 4, skor silhouette adalah 0.41.
- Ini menunjukkan bahwa, secara umum, pengamatan dalam kelompok yang sama memiliki jarak yang cukup seragam satu sama lain dan juga cukup terpisah dari kelompok lain.
- Nilai 0.41 mengindikasikan bahwa pengelompokan ini mungkin memadai, tetapi masih memiliki ruang untuk perbaikan. Semakin tinggi nilai Silhouette Score, semakin baik pengelompokan data tersebut.

##### Clustering Analysis
![6](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-6.jpeg)

####  Customer Personality Analysis for Marketing Retargeting
![7](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-7.jpeg)
![8](https://github.com/cittamudita/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/3bce80f74e670826ae22d328c93dcb49bb1b5668/M3-8.jpeg)
Cluster 0
Risk of Churn
- Kelompok ini memiliki pendapatan dan pengeluaran terendah. Mereka sering mengunjungi situs web kita, namun jarang melakukan transaksi atau menggunakan promo-promo di platform kami. Mereka lebih cenderung bertransaksi secara organik.

Cluster 1
Mid Spender
- Kelompok ini memiliki pendapatan dan pengeluaran terbesar kedua setelah pengeluar besar. Meskipun mereka cukup jarang mengunjungi platform, mereka merespons kampanye dengan baik. Ciri khasnya adalah penggunaan promo yang lebih tinggi daripada pengeluar besar. Kami perlu memahami preferensi pembelian kelompok ini agar dapat mengelola promosi dengan lebih efisien dan mengurangi biaya kami.

Cluster 2
Low Spender
- Kelompok ini memiliki pengeluaran lebih rendah daripada pengeluar besar dan pengeluar menengah. Mereka cukup sering mengunjungi situs web dan mencari promo, tetapi mereka tidak mengaplikasikan promo se sering kelompok pengeluar menengah.

Cluster 3
High Spender
- Kelompok pelanggan ini jumlahnya lebih sedikit jika dibandingkan dengan total midspender ,total low spender dan risk of churn, tetapi mereka adalah pengeluar besar dengan pendapatan dan pengeluaran yang tinggi di platform. Mereka tidak sering mengunjungi platform kita, tetapi ketika mereka melakukannya, mereka cenderung bertransaksi. Kelompok ini merupakan target yang sangat menarik karena memiliki tingkat konversi yang tinggi.

##### Recommendation
Cluster 0
Risk of Churn
- Melihat perilaku mereka saat berhadapan dengan produk, apakah produk yang ditawarkan kurang sesuai, atau apakah sensitivitas terhadap harga memengaruhi mereka untuk tidak melakukan transaksi di platform kita.

Cluster 1
Mid Spender
- Lakukan analisis lebih mendalam untuk memahami cara meningkatkan frekuensi transaksi dalam kelompok ini. Berikan rekomendasi yang lebih personal dan melakukan analisis lebih detail tentang bagaimana mengoptimalkan promosi untuk kelompok ini. Meskipun kami menawarkan promosi lebih sedikit, kelompok ini tetap berbelanja di platform kita.

Cluster 2
Low Spender
- Berikan insentif kepada pelanggan dalam kelompok ini untuk mengaktifkan kembali aktivitas berbelanja mereka, misalnya dengan menawarkan diskon khusus atau promosi untuk pembelian pertama setelah jangka waktu yang lama.

Cluster 3
High Spender
- Memantau transaksi dan retensi dari kelompok ini. Fokus pada peningkatan pelayanan agar kelompok ini tetap loyal dan tidak melakukan churn.

##### Potential Impact GMV
**Risk of Churn**
GMV Risk of Churn: 38259000

**Mid Spender**
GMV Low Spender: 41176000
  
**Low Spender**
GMV Mid Spender: 614752000
  
**High Spender**
GMV High Spender: 662801000

- Jika kita fokus untuk memfokuskan pada high spender, kita bisa mendapatkan gmv_high_spender sebesar 662801000, sedangkan pada mid_spender sebesar 614752000.
