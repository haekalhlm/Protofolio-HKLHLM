# **Laporan Proyek Machine Learning - Haekal**

## **Project Overview**
### **Latar Belakang**
Banyaknya film-film animasi jepang atau yang disebut dengan *anime* bermunculan 2 tahun terakhir ini, meningkatnya minat nonton terhadap anime mulai disukai oleh banyak orang. Hal ini terjadi bersamaan dengan naiknya kecepatan anime dirilis sehingga banyak orang awam yang mencoba untuk mulai menonton anime, tetapi terhambat dengan pengetahuan tentang anime itu sendiri. Dari banyaknya judul anime mengakibatkan kebingungan untuk memilih.

Oleh karena itu sebuah sistem yang dapat memberikan rekomendasi kepada pengguna berdasarkan genre anime maupun berdasarkan rating antar user yang mereka berikan ke anime yang sudah mereka tonton sebelumnya.
## **Business Understanding**
### **Problem Statement**
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:

* Bagaimana cara melakukan pengelolahan data sehingga dapat menghasilkan rekomendasi yang baik dan relevan bagi pengguna?
* Bagaimana cara membangun sistem untuk merekomendasikan film anime yang yang sesuai dengan preferensi pengguna?
### **Goals**
Tujuan dibuatnya proyek ini adalah sebagai berikut:

* Dapat lakukan pengolahan data yang baik agar dapat digunakan dalam membangun model sistem rekomendasi yang baik.
* Membangun model *machine learning* untuk merekomendasikan sebuah film anime yang kemungkinan disukai oleh pengguna..
### **Solution Approach**
Solusi yang dapat diterapkan agar goals diatas terpenuhi adalah sebagai berikut:

* Melakukan analisa pada data untuk memahami dan Memeriksa missing value dan duplikasi data.
* Melakukan proses Normalisasi data rating agar data dapat dengan mudah di proses oleh model.
* Membangun sistem rekomendasi menggunakan 2 teknik yang umum digunakan yaitu: Content-Based dan Collaborative Filtering. 

## **Data Understanding**
Dataset yang digunakan "Anime Recommendations Database" untuk link sumbernya https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database. Dataset tersebut berisi anime.csv memiliki 12,294  jumlah data dan rating.csv memiliki 7,813,737 jumlah data. Di dalam dataset ini terdapat dua dataframe, yaitu:

**anime.csv**
- anime_id : id unik anime
- name : nama anime
- genre : genre anime
- type : tipe siaran anime (TV, Movie, etc)
- episodes : jumlah episode anime
- rating : rating anime
- members : total user

  | #| Column      | Non-null Count  | Dtype |
  |-- | ---| ----| ----- |
  |0| anime_id     | 12294 non-null | int64 |
  |1| name         | 12294 non-nul  | object |
  |2| genre      |12243 non-null   |    object     |
  |3| type     | 12269 non-null | object |
  |4| episodes   | 12294 non-null |  object      |
  |5| rating  | 12064 non-null |  float64     |
  |6| members     | 12294 non-null  | int64 |


**rating.csv**
- user_id : id unik user
- anime_id : id unik anime
- rating : rating yang diberikan oleh user

  | #| Column     | Dtype |
  | ------ | ----- | ----- |
  |0| user_id     |  int64 |
  |1| anime_id         | int64 |
  |2| rating      |int64     |
  

Berikut beberapa visualisasi hasil Exploratory Data Analysis(EDA) dari data anime dan data rating.
![anime.csv](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/17.png)
Terlihat dari tipe siaran anime paling terbanyak yaitu tipe siaran TV dan yang paling sedikit tipe siaran Music.
![rating.csv](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/18.png)
Dari grafik diatas dapat dilihat jumlah rating 7 paling banyak diberikan oleh uer dan anime berating 10 paling sedikit diberikan oleh user.
## **Data Preparation**
### Data Cleaning

- Memeriksa missing value dari table anime. Menggunakan fungsi isnull dari library pandas. Lalu terdapat 3 fitur yang memiliki misisng value yaitu genre, rating dan type. Selanjutnya data tersebut dibersihkan menggunakan fungsi dorpna.

### Data Transform

- Menyandikan (encode) fitur ‘user_id’ dan ‘anime_id’ ke dalam indeks integer. Hal ini dilakukan karena model machine learning tidak bisa menerima data tipe objek, sehingga kita mengubahnya ke numerik. 
- Memetakan ‘user_id’ dan ‘anime_id’ ke dataframe yang berkaitan. Hal ini dilakukan untuk memberikan informasi yang jelas dan akan mudah jika kita ingin memakai informasi ini.
- Melakukan proses normalisasi terhadap nilai rating. Hal ini dilakukan agar hasil model kita nanti akan lebih akurat.

### Feature Engineering

- Data kemudian dibagi menjadi data train dan data test sebanyak 80:20. Data untuk Training dan Validasi untuk Collaborative Based Filtering. Hal ini dilakukan agar model kita menghindari masalah seperti overfitting dan underfitting.



## **Modelling & Result**     

### **Content-Based Filtering**
Pada Content Based menggunakan TF-IDF Vectorizer untuk membangun sistem rekomendasi berdasarkan genre anime. karena untuk menemukan representasi fitur penting dari setiap genre anime. Selanjutnya merubah vektor tf-idf dalam bentuk matriks dengan fungsi todense(). Lalu menghitung derajat kesamaan (similarity degree) antar anime dengan teknik cosine similarity. Tahap terakhir membuat fungsi anime_recommendations dengan beberapa parameter sebagai berikut:
- nama_anime : Nama anime
- similarity_data : Dataframe mencocokan similarity
- items : nama fitur untuk menyamakan kemiripan 
- k : Banyaknya rekomendasi yang ingin diberikan
    
    Berikut adalah hasil dari model yang sudah dibuat:

  | | name     | genre |
  | ------ | ----- | ----- |
  |0| One Piece: Episode of Sabo - 3 Kyoudai no Kizu...     |  Action, Adventure, Comedy, Drama, Fantasy, Sho... |
  |1| One Piece: Episode of Nami - Koukaishi no Nami...         | Action, Adventure, Comedy, Drama, Fantasy, Sho... |
  |2| One Piece: Episode of Merry - Mou Hitori no Na...      |Action, Adventure, Comedy, Drama, Fantasy, Sho...     |
  |3| One Piece: Oounabara ni Hirake! Dekkai Dekkai ...         | Action, Adventure, Comedy, Fantasy, Shounen, S... |
  |4| One Piece: Adventure of Nebulandia      |Action, Adventure, Comedy, Fantasy, Shounen, S...     |
  


### **Collaborative Filtering**
 Pada model Collabirative menggunakan RecommenderNet. Setelah itu me-compile model menggunakan Binary Crossentropy untuk menghitung loss function, menggunakan Adam (Adaptive Moment Estimation) sebagai optimizer dan root mean squared error (RMSE) sebagai metrics evaluation. Selanjutnya melatih model dengan batch size = 8, dan epoch = 100. Untuk mendapatkan hasil rekomendasi yang optimal. Lalu membuat fungsi untuk mendapatkan anime yang belum pernah ditonton oleh user dengan menyamakan anime_id yang berada di anime.csv dan rating.csv . 
    Berikut adalah hasil rekomendasi berdassarkan rating::
### showing recommendations for users: 7
 - Anime with high ratings from user

  |  | name     | genre |
  | ------ | ----- | ----- |
  |0| No Game No Life     |  Adventure, Comedy, Ecchi, Fantasy, Game, Supernatural |
  |1| Neon Genesis Evangelion  | Action, Dementia, Drama, Mecha, Psychological, Sci-Fi |
  |2|Pokemon: The Origin   |Action, Adventure, Comedy, Fantasy, Kids     |
  |3|Umineko no Naku Koro ni  | Horror, Mystery, Psychological, Supernatural |
  |4| Hayate no Gotoku! Cant Take My Eyes Off You    | Comedy, Harem, Parody, Shounen   |
  
 - Top 10 anime recommendation
 
  |  | name     | genre |
  | ------ | ----- | ----- |
  |0| Haikyuu!! Second Season     |  Comedy, Drama, School, Shounen, Sports |
  |1| Sen to Chihiro no Kamikakushi   | Adventure, Drama, Supernatural |
  |2|Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai.  | Drama, Slice of Life, Supernatural   |
  |3| Magi |The Kingdom of Magic : Action, Adventure, Fantasy, Magic, Shounen|
  |4| Sakamichi no Apollon   | Drama, Josei, Music, Romance, School |
  |5| Fate/stay night   | Unlimited Blade Works : Action, Fantasy, Magic, Shounen, Supernatural |
  |6|Hajime no Ippo: Boxer no Kobushi   | Comedy, Drama, Shounen, Sports |
  |7| Kara no Kyoukai 3 : Tsuukaku Zanryuu   | Action, Drama, Mystery, Supernatural, Thriller |
  |8| Yowamushi Pedal   | Comedy, Drama, Shounen, Sports |
  |9| Byousoku 5 Centimeter   | Drama, Romance, Slice of Life |
 

Darri hasil di atas menampilkan genre anime yang direkomendasikan memiliki persamaan dengan anime yang sudah ditonton oleh user.
    

### Evaluation
#### Content-Based
Pada teknik *content-based*, metrik evaluasi yang digunakan yaitu metrik presisi. Dari metrik ini, akan dihitung berdasarkan rekomendasi item anime yang memiliki genre yang sesuai dengan anime.  Berikut rumus dari metrik presisi:

![rumus metrik presisi](https://hasty.ai/media/pages/docs/mp-wiki/metrics/accuracy/fcbf093d04-1653642321/11.png)

Yang dimana:
- *Accuracy*: Nilai akurasi
- *Number of correct predictions*: Banyaknya jumlah data yang benar
- *Total number of predictions*: Banyaknya jumlah data yang diprediksi

Dari hasil yang diberikan pada tahapan *Content-Based*, 5 dari 5 data anime yang direkomendasikan memiliki genre yang sama, dan dilakukan sebanyak 2 kali pada film yang berbeda. Sehingga hal ini dapat disimpulkan bahwa akurasi yang dihasilkan yaitu sebesar 80%.

#### Collaborative Filtering
Berikut metrik evaluasi *Root Mean Squared Error* (RMSE) yang berfungsi untuk mengukur tingkat akurasi perkiraan dari suatu model. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi. Berikut rumus dari RMSE:

$$RMSE = \sqrt {\frac{1}{N} \sum_{i=1}^{N} (\hat{y_{i}} - y_{i})^2}$$



Yang dimana:
- Predicted(i) = Nilai prediksi
- Actual(i) = Nilai sebenarnya
- N = jumlah yang diobservasi

Berikut merupakan hasil RMSE dari model dengan *collaborative filtering*.
![rmse](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/21.png)
Dari hasil pelatihan yang dilakukan. Proses training model cukup smooth dan model konvergen pada epochs sekitar 20. Dapat dilihat bahwa nilai konvergen metrik RMSE berada di sekitar 0.16untuk training dan disekitar 0.23 untuk validasi. Nilai tersebut cukup baik untuk sistem rekomendasi. 

### Kesimpulan
Dari hasil pembuatan proyek sistem rekomendasi film anime di atas. Dapat disimpulkan bahwa model memiliki performa yang sangat baik dalam memberikan rekomendasi, baik pada model yang menggunakan Content-Based maupun Collaborative Filtering. Hal ini didukung dengan hasil rekomendasi yang diberikan dari kedua model menghasilkan rekomendasi yang cukup relevan kepada pengguna dengan nilai RMSE yang diperoleh sebesar 0.1627 untuk data latih dan 0.2134 untuk data uji
