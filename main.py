import numpy as np
import pandas as pd
import streamlit as st
import os
from joblib import load
import matplotlib.pyplot as plt

# Sayfa Ayarları
st.set_page_config(
    page_title="Mantar Sınıflandırıcı",
    page_icon="images/msh.png",
    menu_items={
        "Get help": "mailto:betulnur.demirhan@turktelekom.com.tr",
        "About": "Daha Fazla Bilgi İçin\n" + "https://github.com/bnurdemirhan/datascience"
    }
)

# Arka Plan Resmi
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("images/doga.jpg");  # Arka plan resmini güncelledik
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Başlık Ekleme
st.title("Mantar Sınıflandırma Projesi")
st.image("images/mushrooms.png")

# Senaryo İçeriği
st.subheader("Bu Mantar Zararlı mı Yararlı mı?")
st.markdown(""" 
Bir grup doğa meraklısı, ormanda keşfe çıkmış ve çeşitli mantar türlerini incelemeye karar vermiştir. Ancak, mantarların bazı türlerinin zehirli olduğunu ve sağlığı tehdit edebileceğini unutmamak gerekir. Bu nedenle, mantarları doğru bir şekilde tanıyabilmek, grup için hayati önem taşımaktadır.

Mantarların yenilebilir olup olmadığını anlamak için belirli özelliklerin göz önünde bulundurulması gerekir. Uygulamamız, kullanıcıların bu özellikleri girerek mantarların zararlı mı yoksa yararlı mı olduğunu öğrenmelerine yardımcı olacaktır.
""")

st.subheader("Proje Amacı")
st.markdown(""" 
Bu proje, kullanıcıların mantarları sınıflandırarak yenilebilir ve zehirli olanları ayırt etmelerine yardımcı olmayı amaçlar. Kullanıcılar, mantarın fiziksel özelliklerini (örneğin, solungaç boyutu, rengi, kap şekli) girecek ve modelimiz bu bilgilerle mantarın yenilebilir mi yoksa zehirli mi olduğunu tahmin edecektir.
""")

st.subheader("Kullanıcı Deneyimi")
st.markdown(""" 
Uygulama arayüzü, kullanıcıların mantar özelliklerini seçmesine olanak tanır. Kullanıcılar, aşağıdaki özellikleri belirleyerek mantarlarını tanımlayabilir:
- **Solungaç Boyutu**: Solungaçların genişliği, mantarın türünü belirlemek için önemli bir göstergedir.
- **Solungaç Rengi**: Solungaçların rengi, zehirli ve yenilebilir türler arasında farklılık gösterir.
- **Kap Şekli**: Mantarın üst kısmının şekli, türlerini ayırt etmek için kullanılan önemli bir özelliktir.
- **Kap Rengi**: Mantarın kap renginin belirli türlerle ilişkili olup olmadığını belirlemek için bir başka önemli faktördür.
- **Bruises**: Mantarın zarar görüp görmediği bilgisi, bazı türler için önemli bir belirleyicidir.
- **Odor**: Mantarın kokusu, bazı zehirli türler için belirgin bir ipucu verebilir.
- **Gill Attachment**: Solungaçların mantar sapına nasıl bağlandığı, türler arasında farklılık gösterebilir.
""")

st.subheader("Bilgilendirme ve Eğitim")
st.markdown(""" 
Mantar sınıflandırma modeli, kullanıcıya her tahmin sonucunda mantar hakkında bilgi vererek, hangi özelliklerin belirli bir mantar türünün yenilebilirliği ile ilişkili olduğunu açıklayacaktır. Kullanıcılar, bu uygulama sayesinde mantarları tanımayı ve sağlıklı seçimler yapmayı öğreneceklerdir.
""")

# Dosya yolunu dinamik olarak oluştur
base_dir = os.path.dirname(__file__)  # Mevcut dosyanın dizinini al
df_path = os.path.join(base_dir, 'mushrooms.csv')  # 'mushrooms.csv' ile birleştir

# Pandasla veri setini okuyalım
df_mushrooms = pd.read_csv(df_path)

# Label Encoding
df_encoded = df_mushrooms.apply(lambda col: pd.factorize(col)[0])

# Tablo Ekleme
st.dataframe(df_encoded.sample(5))  # 5 rastgele örneği göster

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Sonucu görmek için aşağıdaki özellikleri seçin!**")

# Gill Size ve Gill Color için kullanıcıdan girdi alma
gill_size_options = {0: "Dar", 1: "Geniş"}  # 0 ve 1 yerine anlamlı isimler
gill_size = st.sidebar.selectbox("Solungaç Boyutu", options=list(gill_size_options.keys()),
                                 format_func=lambda x: gill_size_options[x], help="Solungaçların genişliğini seçin.")

# Renk seçeneklerini oluştur
gill_color_options = {0: "Siyah", 1: "Kahverengi", 2: "Açık Kahverengi", 3: "Çikolata", 4: "Gri", 5: "Yeşil",
                      6: "Pembe", 7: "Mor", 8: "Kırmızı", 9: "Beyaz", 10: "Sarı"}
gill_color = st.sidebar.selectbox("Solungaç Rengi", options=list(gill_color_options.keys()),
                                  format_func=lambda x: gill_color_options[x], help="Solungaçların rengini seçin.")

# Modeli yükleme
logreg_model = load('logreg_model.pkl')
std_scale = load('scaler.pkl')

# Kullanıcı girdilerini DataFrame'e dönüştürme
input_df = pd.DataFrame({
    'gill-size': [gill_size],
    'gill-color': [gill_color],
})

# Verileri ölçeklendirme
scaled_input_df = std_scale.transform(input_df)

# Tahmin yapma
pred = logreg_model.predict(scaled_input_df)
pred_probability = np.round(logreg_model.predict_proba(scaled_input_df), 2)

# Sonuç Ekranı
if st.sidebar.button("Gönder", key="submit_button"):
    st.info("Sonuç aşağıda bulunabilir.")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
        'Solungaç Boyutu': [gill_size_options[gill_size]],
        'Solungaç Rengi': [gill_color_options[gill_color]],
        'Tahmin': [pred],
        'Yenilebilir Olasılığı': [pred_probability[0][1]],
        'Zehirli Olasılığı': [pred_probability[0][0]]
    })

    results_df["Tahmin"] = results_df["Tahmin"].apply(lambda x: "Zehirli" if x == 0 else "Yenilebilir")
    st.table(results_df)

    # Görsel sonuç
    if pred == 0:
        st.image("images/zehir.png", caption='Zehirli Mantar')
    else:
        st.image("images/ye.png", caption='Yenilebilir Mantar')

    # Mantar Dağılım Grafiği
    st.subheader("Mantar Dağılım Grafiği")
    plt.figure(figsize=(10, 5))
    df_mushrooms['class'].value_counts().plot(kind='bar', color=['red', 'green'])
    plt.title("Mantar Türlerine Göre Dağılım")
    plt.xlabel("Mantar Türü")
    plt.ylabel("Sayısı")
    st.pyplot(plt)

    # Geri Bildirim Formu
    st.subheader("Geri Bildirim")
    feedback = st.text_area("Lütfen geri bildiriminizi buraya yazın.")
    if st.button("Geri Bildirim Gönder", key="feedback_button"):
        st.success("Geri bildiriminiz için teşekkürler!")

    # Sık Sorulan Sorular
    st.subheader("Sık Sorulan Sorular")
    with st.expander("SSS"):
        st.markdown(""" 
        - **Mantar nasıl sınıflandırılır?** 
          Mantarlar, fiziksel özelliklerine göre sınıflandırılır.

        - **Zehirli mantarları nasıl tanıyabilirim?**
          Belirli özellikler, zehirli türlerin tanımlanmasında yardımcı olabilir.

        - **Yenilebilir mantarlar nelerdir?**
          Kullanıcı, belirli özellikleri girerek bu mantarları öğrenebilir.
        """)

else:
    st.markdown("Lütfen *Gönder* butonuna tıklayın!")
