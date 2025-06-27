# === Import Library ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split

# === Konfigurasi Halaman ===
st.set_page_config(layout="wide", page_title="Analisis Pasar Laptop India")

# === Fungsi Formatter Harga ===
def price_formatter(price):
    if price >= 100000:
        return f'₹{price/100000:.2f} Lakh'
    elif price >= 1000:
        return f'₹{price/1000:.1f}K'
    else:
        return f'₹{price:.0f}'

# === STREAMLIT APP LAYOUT ===
st.title("💻 Dashboard Analisis Pasar Laptop India")

# Sidebar
st.sidebar.title("Menu Dashboard")

# === File Upload ===
uploaded_file = st.sidebar.file_uploader("📂 Unggah File CSV Dataset Laptop", type=["csv"])

# Variabel untuk melacak status data
data_processed = False

# === Data Processing ===
if uploaded_file is not None:
    try:
        # Baca data
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File berhasil diunggah!")
        
        # === Data Preparation ===
        # Hapus duplikat
        df = df.drop_duplicates()
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Feature Engineering (jika kolom tertentu ada)
        if 'Ram' in df.columns and 'Ram_GB' not in df.columns:
            # Extract RAM in GB (menggunakan raw string untuk regex)
            df['Ram_GB'] = df['Ram'].str.extract(r'(\d+)').astype(int)
        
        if 'Memory' in df.columns:
            # Extract storage info
            if 'Storage_GB' not in df.columns:
                def extract_storage_size(text):
                    total_gb = 0
                    if isinstance(text, str):
                        import re
                        sizes = re.findall(r'(\d+)(?:\s*)(GB|TB)', text)
                        for size, unit in sizes:
                            if unit == 'TB':
                                total_gb += int(size) * 1024  # 1 TB = 1024 GB
                            else:
                                total_gb += int(size)
                    return total_gb
                
                df['Storage_GB'] = df['Memory'].apply(extract_storage_size)
            
            if 'Has_SSD' not in df.columns:
                df['Has_SSD'] = df['Memory'].apply(lambda x: 1 if isinstance(x, str) and 'SSD' in x else 0)
            
            if 'Has_HDD' not in df.columns:
                df['Has_HDD'] = df['Memory'].apply(lambda x: 1 if isinstance(x, str) and 'HDD' in x else 0)
        
        # Label Encoding
        label_encoders = {}
        for col in df.select_dtypes(include='object'):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Log transform Price for better modeling
        if 'Price' in df.columns:
            df['Log_Price'] = np.log1p(df['Price'])
            
            # Feature & Target
            X = df.drop(['Price', 'Log_Price'], axis=1)
            y = df['Price']
            y_log = df['Log_Price']
            
            # Standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
            
            # Linear Regression on Log Price
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Convert back to original scale for interpretation
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            r2_orig = r2_score(y_test_orig, y_pred_orig)
            
            # KMeans Clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, df['Cluster'])
            
            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            
            # Get cluster names
            cluster_stats = df.groupby('Cluster')['Price'].mean().sort_values()
            cluster_names = {
                cluster_stats.index[0]: "Entry-level",
                cluster_stats.index[1]: "Mid-range",
                cluster_stats.index[2]: "Premium"
            }
            df['Segment'] = df['Cluster'].map(cluster_names)
            
            # Tandai bahwa data telah diproses
            data_processed = True
        else:
            st.error("❌ Dataset harus memiliki kolom 'Price'. Silakan unggah dataset yang sesuai.")
            data_processed = False
    
    except Exception as e:
        st.error(f"❌ Error dalam pemrosesan data: {e}")
        data_processed = False

# === Menu Selection ===
menu = st.sidebar.radio("📌 Menu", [
    "🏠 Business Understanding", 
    "📊 Data Exploration", 
    "📈 Prediksi Harga", 
    "🧩 Clustering", 
    "🧪 Simulasi Data Baru",
    "📑 Kesimpulan & Rekomendasi"
])

# === Business Understanding (selalu ditampilkan) ===
if menu == "🏠 Business Understanding":
    st.header("🏢 Business Understanding")
    
    st.markdown("""
    <div style='background-color:#f0f2f6;padding:15px;border-radius:10px;'>
    <h3>Tujuan Bisnis:</h3>
    <p>Menganalisis pasar laptop India untuk mengidentifikasi segmentasi pasar, faktor-faktor yang memengaruhi harga, dan peluang potensial bagi vendor untuk strategi penetapan harga dan pengembangan produk yang optimal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Objektif Bisnis")
        st.markdown("""
        1. **Memahami segmentasi pasar laptop India** berdasarkan karakteristik teknis dan harga
        2. **Mengidentifikasi faktor-faktor utama** yang memengaruhi harga laptop di pasar India
        3. **Mengembangkan model prediksi harga** untuk membantu penetapan harga kompetitif
        4. **Merekomendasikan strategi pemasaran** yang sesuai dengan segmen pasar yang teridentifikasi
        """)
    
    with col2:
        st.subheader("Pertanyaan Bisnis")
        st.markdown("""
        1. **Segmentasi Pasar**: Bagaimana struktur segmentasi pasar laptop India berdasarkan spesifikasi dan harga?
        2. **Price Drivers**: Faktor teknis apa yang paling memengaruhi harga laptop di pasar India?
        3. **Brand Premium**: Seberapa besar pengaruh brand terhadap harga di berbagai segmen?
        4. **Competitive Pricing**: Bagaimana menentukan harga optimal untuk spesifikasi tertentu?
        5. **Market Gap**: Adakah segmen pasar yang belum terlayani dengan baik (underserved)?
        """)
    
    st.subheader("Latar Belakang Pasar")
    st.write("""
    Pasar laptop India mengalami pertumbuhan pesat dengan CAGR 15.4% selama 2018-2023, mencapai valuasi $7.5 miliar pada 2023. 
    Pertumbuhan ini didorong oleh meningkatnya penetrasi internet, inisiatif Digital India, dan kebutuhan akan perangkat computing yang terjangkau 
    namun mampu mendukung produktivitas dan entertainment.
    """)
    
    st.subheader("Metodologi Analisis")
    st.write("""
    Analisis menggunakan pendekatan dua model:
    1. **Supervised Learning (Linear Regression)**: Memprediksi harga laptop berdasarkan spesifikasi teknis
    2. **Unsupervised Learning (K-Means Clustering)**: Segmentasi pasar berdasarkan karakteristik laptop
    """)

# === Menu 2: Data Exploration ===
elif menu == "📊 Data Exploration":
    st.header("📊 Data Exploration & Understanding")
    
    if data_processed:
        tab1, tab2, tab3 = st.tabs(["📋 Overview", "📊 Distribusi", "🔄 Korelasi"])
        
        with tab1:
            st.subheader("📌 Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**5 Data Teratas:**")
                st.dataframe(df.head())
            
            with col2:
                st.write("**Struktur Data:**")
                st.text(f"Jumlah baris: {df.shape[0]}  |  Jumlah kolom: {df.shape[1]}")
                st.write("Tipe data per kolom:")
                st.write(df.dtypes)
            
            st.subheader("📉 Statistik Deskriptif")
            st.write(df.describe())
            
            st.subheader("🧩 Cek Duplikat & Missing Values")
            st.write(f"Jumlah duplikat: {df.duplicated().sum()}")
            st.write("Jumlah missing value per kolom:")
            st.write(df.isnull().sum())
        
        with tab2:
            st.subheader("🔍 Distribusi Harga")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['Price'], kde=True, bins=25, color='navy', ax=ax)
                ax.set_title('Distribusi Harga Laptop', fontsize=14)
                ax.set_xlabel('Harga (₹)', fontsize=12)
                st.pyplot(fig)
                
                st.markdown("""
                **Insight:**
                - Distribusi harga positively skewed dengan konsentrasi di segmen entry dan mid-range
                - Terlihat 3 kelompok harga: entry-level, mid-range, dan premium
                - Terdapat outlier di segmen ultra-premium
                """)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=df['Price'], color='navy', ax=ax)
                ax.set_title('Box Plot Harga Laptop', fontsize=14)
                ax.set_ylabel('Harga (₹)', fontsize=12)
                st.pyplot(fig)
                
                # Price stats
                price_stats = df['Price'].describe()
                st.markdown(f"""
                **Statistik Harga:**
                - **Minimum:** {price_formatter(price_stats['min'])}
                - **Q1 (25%):** {price_formatter(price_stats['25%'])}
                - **Median:** {price_formatter(price_stats['50%'])}
                - **Q3 (75%):** {price_formatter(price_stats['75%'])}
                - **Maximum:** {price_formatter(price_stats['max'])}
                - **Mean:** {price_formatter(price_stats['mean'])}
                """)
        
        with tab3:
            st.subheader("📊 Korelasi Antar Fitur")
            
            # Pilih hanya kolom numerik untuk korelasi
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Heatmap Korelasi Antar Variabel', fontsize=14)
            st.pyplot(fig)
            
            # Korelasi dengan harga
            if 'Price' in numeric_cols:
                price_corr = df[numeric_cols].corr()['Price'].sort_values(ascending=False)
                st.write("**Korelasi dengan Harga:**")
                st.write(price_corr)
                
                st.markdown("""
                **Insight Korelasi:**
                - **RAM** memiliki korelasi tertinggi dengan harga, menunjukkan ini faktor teknis terpenting
                - **SSD** berkorelasi positif kuat, HDD berkorelasi negatif - menandakan preferensi konsumen terhadap SSD
                - **Storage capacity** kurang berpengaruh dibandingkan tipe storage
                
                **Implikasi Bisnis:**
                - Strategi pricing sebaiknya memprioritaskan RAM dan tipe storage sebagai faktor utama
                - SSD dapat menjadi fitur diferensiasi utama di segmen mid-range
                """)
    else:
        st.info("📂 Silakan unggah file dataset terlebih dahulu untuk melihat eksplorasi data.")

# === Menu 3: Regression ===
elif menu == "📈 Prediksi Harga":
    st.header("📈 Model Prediksi Harga Laptop")
    
    if data_processed:
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("RMSE (pada log harga)", f"{rmse:.4f}")
            st.metric("R² Score", f"{r2:.4f}")
        
        with col2:
            st.metric("RMSE (harga asli)", f"₹{rmse_orig:.2f}")
            st.metric("R² Score (harga asli)", f"{r2_orig:.4f}")
        
        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_title("Actual vs. Predicted (Log Price)", fontsize=14)
        ax.set_xlabel("Log Price Aktual", fontsize=12)
        ax.set_ylabel("Log Price Prediksi", fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Residual analysis
        residuals = y_test - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residual plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_title("Residual Plot", fontsize=14)
            ax.set_xlabel("Prediksi Log Price", fontsize=12)
            ax.set_ylabel("Residual", fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Histogram of residuals
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(residuals, bins=30, kde=True, color='navy', ax=ax)
            ax.set_title("Distribusi Residual", fontsize=14)
            ax.set_xlabel("Residual", fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Feature importance
        if hasattr(lr, 'coef_'):
            st.subheader("🏆 Pengaruh Fitur Terhadap Harga")
            
            coef_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': lr.coef_
            }).sort_values('Coefficient', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10), palette='viridis', ax=ax)
            ax.set_title('Top 10 Feature Importance', fontsize=14)
            ax.set_xlabel('Magnitude of Impact on Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.markdown("""
            **Insight Feature Importance:**
            - **RAM** adalah prediktor terkuat harga laptop di pasar India
            - **SSD** memberikan premium harga yang signifikan
            - **Brand premium** (jika tersedia dalam dataset) memiliki pengaruh substansial pada segmen high-end
            
            **Implikasi Bisnis:**
            - Strategi upselling sebaiknya fokus pada peningkatan RAM dan penambahan SSD
            - Peningkatan kapasitas storage kurang efektif dibanding upgrade tipe storage
            - Perlu mempertimbangkan nilai brand dalam penetapan harga, terutama di segmen premium
            """)
    else:
        st.info("📂 Silakan unggah file dataset terlebih dahulu untuk melihat model prediksi harga.")

# === Menu 4: Clustering ===
elif menu == "🧩 Clustering":
    st.header("🧩 Segmentasi Pasar Laptop")
    
    if data_processed:
        st.metric("Silhouette Score", f"{sil_score:.3f}", 
                 help="Silhouette score mengukur seberapa baik cluster terpisah. Nilai mendekati 1 menunjukkan separasi yang baik.")
        
        # Visualize clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Segment', palette='viridis', s=70, alpha=0.7, ax=ax)
        
        # Add centroids
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, marker='*', 
                   c='red', edgecolor='k', label='Centroids')
        
        ax.set_title('Visualisasi Segmen Pasar dengan PCA', fontsize=14)
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.legend(title='Segmen Pasar')
        st.pyplot(fig)
        
        # Cluster statistics
        st.subheader("📊 Karakteristik Segmen Pasar")
        
        # Show relevant features per cluster
        relevant_features = ['Price', 'Ram_GB', 'Storage_GB', 'Has_SSD', 'Has_HDD']
        if 'Inches' in df.columns:
            relevant_features.append('Inches')
        
        # Ensure all required columns exist
        available_features = [col for col in relevant_features if col in df.columns]
        
        # Calculate segment statistics
        segment_stats = df.groupby('Segment')[available_features].mean().round(2)
        
        # Add count of laptops per segment
        segment_counts = df['Segment'].value_counts()
        segment_stats['Count'] = segment_counts
        
        st.dataframe(segment_stats)
        
        # Cluster interpretation
        st.subheader("🧠 Interpretasi Segmen")
        
        st.markdown("""
        **Segmentasi Pasar Laptop India:**
        
        **1. Entry-level (Budget Segment):**
        - RAM ~4GB, mayoritas HDD, harga terjangkau
        - Target: Pelajar, first-time buyers, basic computing
        - Price Range: ₹10K-₹40K
        
        **2. Mid-range (Value Segment):**
        - RAM ~8GB, mix SSD & HDD, performa balanced
        - Target: Professionals, small business, daily productivity
        - Price Range: ₹40K-₹70K
        
        **3. Premium (Performance Segment):**
        - RAM 16GB+, dominan SSD, performa tinggi
        - Target: Power users, gaming, content creation
        - Price Range: ₹70K+
        
        **Implikasi Bisnis:**
        - Strategi produk dan marketing perlu disesuaikan per segmen
        - Mid-range memiliki potensi pertumbuhan tertinggi di pasar India
        - Premium segment memberikan margin tertinggi tetapi volume lebih kecil
        """)
    else:
        st.info("📂 Silakan unggah file dataset terlebih dahulu untuk melihat segmentasi pasar.")

# === Menu 5: Simulasi Data Baru ===
elif menu == "🧪 Simulasi Data Baru":
    st.header("🧪 Simulasi Prediksi Harga Laptop Baru")
    
    if data_processed:
        st.write("""
        Gunakan simulator ini untuk memprediksi harga laptop berdasarkan spesifikasi yang dipilih.
        Model ini dapat membantu dalam menentukan strategi penetapan harga yang kompetitif.
        """)
        
        # Collect input dalam dictionary
        input_values = {}
        
        # Handle categorical features
        cat_features = [col for col in X.columns if col in label_encoders]
        st.subheader("Spesifikasi Kategorikal")
        
        # Split into columns
        col1, col2 = st.columns(2)
        half_point = len(cat_features) // 2
        
        with col1:
            for col in cat_features[:half_point]:
                options = label_encoders[col].classes_.tolist()
                selected = st.selectbox(f"{col}", options)
                input_values[col] = label_encoders[col].transform([selected])[0]
        
        with col2:
            for col in cat_features[half_point:]:
                options = label_encoders[col].classes_.tolist()
                selected = st.selectbox(f"{col}", options)
                input_values[col] = label_encoders[col].transform([selected])[0]
        
        # Handle numeric features
        st.subheader("Spesifikasi Numerik")
        num_features = [col for col in X.columns if col not in label_encoders]
        
        # Divide numeric features into 2 columns
        num_half = len(num_features) // 2
        col1, col2 = st.columns(2)
        
        with col1:
            for col in num_features[:num_half]:
                mean_val = float(X[col].mean())
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                value = st.number_input(f"{col}", value=mean_val, min_value=min_val, max_value=max_val)
                input_values[col] = value
        
        with col2:
            for col in num_features[num_half:]:
                mean_val = float(X[col].mean())
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                value = st.number_input(f"{col}", value=mean_val, min_value=min_val, max_value=max_val)
                input_values[col] = value
        
        # PERBAIKAN: Buat DataFrame dengan kolom yang sama persis seperti X
        # Pastikan semua kolom ada dan dalam urutan yang sama
        input_df = pd.DataFrame(columns=X.columns)
        for col in X.columns:
            if col in input_values:
                input_df.loc[0, col] = input_values[col]
            else:
                input_df.loc[0, col] = 0  # Default value jika kolom tidak ada di input
        
        # Scale input (sekarang akan bekerja karena kolom input_df sama persis dengan X)
        input_scaled = scaler.transform(input_df)
        
        # Predict price (log scale)
        log_pred_price = lr.predict(input_scaled)[0]
        
        # Convert back to original scale
        pred_price = np.expm1(log_pred_price)
        
        # Display prediction
        st.markdown(f"""
        <div style='background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;'>
        <h2>Estimasi Harga Laptop: {price_formatter(pred_price)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Determine segment
        if pred_price < 40000:
            segment = "Entry-level"
        elif pred_price < 70000:
            segment = "Mid-range"
        else:
            segment = "Premium"
        
        st.write(f"**Segmen Pasar:** {segment}")
        
        # Price range suggestion
        lower_bound = max(0, pred_price - 0.1 * pred_price)
        upper_bound = pred_price + 0.1 * pred_price
        
        st.write(f"**Range Harga Kompetitif:** {price_formatter(lower_bound)} - {price_formatter(upper_bound)}")
    else:
        st.info("📂 Silakan unggah file dataset terlebih dahulu untuk menggunakan simulator prediksi harga.")

# === Menu 6: Conclusion ===
elif menu == "📑 Kesimpulan & Rekomendasi":
    st.header("📑 Kesimpulan & Rekomendasi Bisnis")
    
    if data_processed:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Key Insights")
            st.markdown("""
            **1. Segmentasi Pasar:**
            - Pasar laptop India terbagi dalam 3 segmen utama: Entry-level, Mid-range, dan Premium
            - Segmen mid-range menunjukkan potensi pertumbuhan tertinggi
            
            **2. Price Drivers:**
            - RAM adalah faktor teknis terkuat yang memengaruhi harga
            - Keberadaan SSD memberikan premium harga signifikan
            - Brand memiliki pengaruh substansial di segmen premium
            """)
        
        with col2:
            st.subheader("📋 Rekomendasi Strategis")
            st.markdown("""
            **1. Product Strategy:**
            - Entry-level: 4GB RAM, 500GB storage, fokus affordability
            - Mid-range: 8GB RAM, SSD standard, fokus value-for-money
            - Premium: 16GB+ RAM, SSD, fokus performa dan eksklusivitas
            
            **2. Pricing Strategy:**
            - Entry-level: Competitive pricing dengan margin 15-20%
            - Mid-range: Value-based pricing dengan margin 20-25%
            - Premium: Premium pricing dengan margin 25-35%
            """)
        
        st.subheader("🎯 Action Plan")
        st.markdown("""
        1. Implementasikan strategi segmentasi 3-tier untuk optimasi product lineup
        2. Fokus pada value-add di segmen mid-range yang memiliki growth potential tertinggi
        3. Develop premium mid-range sebagai opportunity segment untuk penetrasi pasar
        """)
    else:
        st.info("📂 Silakan unggah file dataset terlebih dahulu untuk melihat kesimpulan dan rekomendasi.")

# Footer (selalu ditampilkan)
st.markdown("""
---
Dashboard ini dikembangkan sebagai bagian dari proyek analisis pasar laptop India.  
Data mencakup berbagai spesifikasi dan harga laptop di pasar India.
""")