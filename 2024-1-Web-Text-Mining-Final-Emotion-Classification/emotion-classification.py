# Converted from the original Jupyter notebook.
# Markdown cells and saved text outputs are preserved as comments.

# %%
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
# %% [markdown]
# Saved output
# Reading package lists... Done
# Building dependency tree... Done
# Reading state information... Done
# The following NEW packages will be installed:
#   fonts-nanum
# 0 upgraded, 1 newly installed, 0 to remove and 45 not upgraded.
# Need to get 10.3 MB of archives.
# After this operation, 34.1 MB of additional disk space will be used.
# Get:1 http://archive.ubuntu.com/ubuntu jammy/universe amd64 fonts-nanum all 20200506-1 [10.3 MB]
# Fetched 10.3 MB in 1s (15.5 MB/s)
# debconf: unable to initialize frontend: Dialog
# debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 1.)
# debconf: falling back to frontend: Readline
# debconf: unable to initialize frontend: Readline
# debconf: (This frontend requires a controlling tty.)
# debconf: falling back to frontend: Teletype
# dpkg-preconfigure: unable to re-open stdin: 
# Selecting previously unselected package fonts-nanum.
# (Reading database ... 121925 files and directories currently installed.)
# Preparing to unpack .../fonts-nanum_20200506-1_all.deb ...
# Unpacking fonts-nanum (20200506-1) ...
# Setting up fonts-nanum (20200506-1) ...
# Processing triggers for fontconfig (2.13.1-4.2ubuntu5) ...
# /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
# /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
# /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
# /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
# /usr/share/fonts/truetype/nanum: caching, new cache contents: 12 fonts, 0 dirs
# /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
# /root/.local/share/fonts: skipping, no such directory
# /root/.fonts: skipping, no such directory
# /usr/share/fonts/truetype: skipping, looped directory detected
# /usr/share/fonts/truetype/humor-sans: skipping, looped directory detected
# /usr/share/fonts/truetype/liberation: skipping, looped directory detected
# /usr/share/fonts/truetype/nanum: skipping, looped directory detected
# /var/cache/fontconfig: cleaning cache directory
# /root/.cache/fontconfig: not cleaning non-existent cache directory
# /root/.fontconfig: not cleaning non-existent cache directory
# fc-cache: succeeded
# 
#
# %%
import  matplotlib
import  matplotlib.font_manager  as fm
import  matplotlib.pyplot  as plt

sys_font  = fm.findSystemFonts ( )

[ font  for  font  in  sys_font  if  "Nanum"  in font ]
# %% [markdown]
# Saved output
# ['/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumSquareR.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumGothicCodingBold.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumSquareB.ttf',
#  '/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf']
#
# %%
font_path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

font_name  = fm.FontProperties(fname=font_path, size=12).get_name( )

print("???고듃 ?대쫫 : ",font_name)

plt.rc("font", family= font_name)
# %% [markdown]
# Saved output
# ???고듃 ?대쫫 :  NanumGothic
# 
#
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
# %%
from google.colab import drive
drive.mount('/content/drive')
# %% [markdown]
# Saved output
# Mounted at /content/drive
# 
#
# %% [markdown]
# 
# **?곗씠?곗뀑 異쒖쿂 : https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets/code**
#
# %%
import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/2024-1 ?밴낵 ?띿뒪?몃쭏?대떇媛쒕줎/emotion'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# %% [markdown]
# Saved output
# /content/drive/MyDrive/2024-1 ?밴낵 ?띿뒪?몃쭏?대떇媛쒕줎/emotion/sentiment-emotion-labelled_Dell_tweets.csv
# /content/drive/MyDrive/2024-1 ?밴낵 ?띿뒪?몃쭏?대떇媛쒕줎/emotion/tweet_emotions.csv
# 
#
# %%
df = pd.read_csv("/content/drive/MyDrive/2024-1 ?밴낵 ?띿뒪?몃쭏?대떇媛쒕줎/emotion/sentiment-emotion-labelled_Dell_tweets.csv")
# %% [markdown]
# 
# ## **1.?곗씠??遺꾩꽍?섍린**
#
# %% [markdown]
# 
# ### **1-1. ?곗씠???뺤씤 諛??꾩쿂由?*
#
# %%
df
# %% [markdown]
# Saved output
#        Unnamed: 0                   Datetime             Tweet Id  \
# 0               0  2022-09-30 23:29:15+00:00  1575991191170342912   
# 1               1  2022-09-30 21:46:35+00:00  1575965354425131008   
# 2               2  2022-09-30 21:18:02+00:00  1575958171423752203   
# 3               3  2022-09-30 20:05:24+00:00  1575939891485032450   
# 4               4  2022-09-30 20:03:17+00:00  1575939359160750080   
# ...           ...                        ...                  ...   
# 24965       24965  2022-01-01 02:02:04+00:00  1477097760931336198   
# 24966       24966  2022-01-01 01:57:34+00:00  1477096631300415496   
# 24967       24967  2022-01-01 01:36:36+00:00  1477091355629432833   
# 24968       24968  2022-01-01 01:31:30+00:00  1477090070830141442   
# 24969       24969  2022-01-01 00:59:37+00:00  1477082048900726784   
# 
#                                                     Text        Username  \
# 0      @Logitech @apple @Google @Microsoft @Dell @Len...  ManjuSreedaran   
# 1      @MK_habit_addict @official_stier @MortalKombat...      MiKeMcDnet   
# 2      As혻@CRN혻celebrates its 40th anniversary,혻Bob F...        jfollett   
# 3      @dell your customer service is horrible especi...       daveccarr   
# 4      @zacokalo @Dell @DellCares @Dell give the man ...      heycamella   
# ...                                                  ...             ...   
# 24965  @ElDarkAngel2 @GamersNexus @Dell I wouldn't ev...          Eodart   
# 24966  @kite_real @GamersNexus @Dell I didn't really ...          Eodart   
# 24967  Hey @JoshTheFixer here it is....27 4K UHD USB-...     Corleone250   
# 24968  @bravadogaming @thewolfpena @Alienware @intel ...      MrTwistyyy   
# 24969  @rabia_ejaz @Dell Stopped buying windows lapto...   IDevourNehari   
# 
#       sentiment  sentiment_score       emotion  emotion_score  
# 0       neutral         0.853283  anticipation       0.587121  
# 1       neutral         0.519470           joy       0.886913  
# 2      positive         0.763791           joy       0.960347  
# 3      negative         0.954023         anger       0.983203  
# 4       neutral         0.529170         anger       0.776124  
# ...         ...              ...           ...            ...  
# 24965  negative         0.682981         anger       0.906309  
# 24966  positive         0.743940           joy       0.951701  
# 24967   neutral         0.654463  anticipation       0.471185  
# 24968   neutral         0.794049  anticipation       0.747014  
# 24969  positive         0.733861           joy       0.958346  
# 
# [24970 rows x 9 columns]
#
# %% [markdown]
# 
# Datatime : ?몄쐵 ?묒꽦???좎쭨&?쒓컙
# 
# Tweet ID : ?몄쐵 ?묒꽦???꾩씠??# 
# Text : ?몄쐵 ?댁슜
# 
# Username : ?몄쐞???ъ슜???대쫫
# 
# sentiment : 媛먯젙 (湲띿젙/以묐┰/遺??
# 
# sentiment_score : 媛먯젙 ?먯닔(媛먯젙 媛뺣룄瑜??섏튂濡??섑???
# 
# emotion : 媛먯젙 ?좏삎
# 
# emotion_score : 媛먯젙 ?좏삎 ?먯닔
#
# %%
df = df.sample(n=1000, random_state=42)
# %% [markdown]
# 
# ### **1-2. ?몄쐵 ?묒꽦???꾩씠??諛??ъ슜??遺꾩꽍**
#
# %%
tweet_id_counts = df['Tweet Id'].value_counts()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
tweet_id_counts.plot(kind='bar', color='blue')

plt.title('Tweet Id Distribution')
plt.xlabel('Tweet Id')
plt.ylabel('Frequency')
plt.xticks([])

username_counts = df['Username'].value_counts()
plt.subplot(1, 2, 2)
username_counts.plot(kind='bar', color='red')
plt.title('Usernames Distribution')
plt.xlabel('Username')
plt.ylabel('Frequency')

plt.tight_layout()
plt.xticks([])
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1000x500 with 2 Axes>
#
# %% [markdown]
# 
# **?쒖そ?쇰줈 ?좊┛ ?곗씠?캸 -> ?섎??놁쓬 -> ?대떦 ????젣**
#
# %%
df.drop(columns=['Unnamed: 0','Tweet Id', 'Username'], inplace=True)
df
# %% [markdown]
# Saved output
#                         Datetime  \
# 6482   2022-07-19 14:13:30+00:00   
# 13625  2022-05-02 14:37:44+00:00   
# 21043  2022-02-16 05:34:17+00:00   
# 20108  2022-02-26 14:14:19+00:00   
# 2152   2022-09-02 22:39:32+00:00   
# ...                          ...   
# 13176  2022-05-05 08:40:40+00:00   
# 6649   2022-07-18 00:49:00+00:00   
# 10491  2022-06-04 19:39:43+00:00   
# 17346  2022-03-30 22:03:22+00:00   
# 20521  2022-02-23 00:06:10+00:00   
# 
#                                                     Text sentiment  \
# 6482   @AmanNagpal81 @SamsungIndia @SamsungMobile @Ap...   neutral   
# 13625  @jackoneill1984 @GamersNexus @Dell taking one ...   neutral   
# 21043  @MichaelDell @Dell @DellCaresPRO one more http...   neutral   
# 20108  @DellCares What should we do if your dealer ch...  negative   
# 2152   #facebookdown @TheRock @steveaustinBSR혻 @WWEUn...   neutral   
# ...                                                  ...       ...   
# 13176  Reminder 7:\n@Dell @DellCares = #Delldoesnotgi...  negative   
# 6649   Has @Dell support always been this terrible? A...  negative   
# 10491  The promise of tech: work from anywhere, anyti...  positive   
# 17346  Dear @Dell , first of all thanks to check my o...  negative   
# 20521  @AirspanNetworks @DellTech @Dell We're excited...  positive   
# 
#        sentiment_score       emotion  emotion_score  
# 6482          0.699980       sadness       0.927320  
# 13625         0.701479           joy       0.767457  
# 21043         0.743684  anticipation       0.234333  
# 20108         0.841557         anger       0.983523  
# 2152          0.901858  anticipation       0.550912  
# ...                ...           ...            ...  
# 13176         0.833563         anger       0.919111  
# 6649          0.901842       disgust       0.899949  
# 10491         0.816528      optimism       0.970569  
# 17346         0.877455         anger       0.969596  
# 20521         0.982404           joy       0.974803  
# 
# [1000 rows x 6 columns]
#
# %% [markdown]
# 
# ### **1-3. sentiment , emotion ?먭렇?섑봽濡??쒓컖??*
#
# %%
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')

emotion_counts = df['emotion'].value_counts()

plt.subplot(1, 2, 2)
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Emotion Distribution')

plt.tight_layout()
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1200x600 with 2 Axes>
#
# %% [markdown]
# 
# ### **1-4. emotion 留됯렇?섑봽濡??쒓컖??*
#
# %%
emotion_counts = df['emotion'].value_counts()

plt.figure(figsize=(10, 6))

bars = plt.bar(emotion_counts.index, emotion_counts.values)

plt.title('Emotion Distribution')
plt.xlabel('Emotion')
plt.ylabel('Frequency')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1000x600 with 1 Axes>
#
# %% [markdown]
# 
# ### **1-5. ?덊듃留듭쑝濡?sentiment? emotion 愿怨??쒓컖??*
#
# %%
pivot_table = pd.pivot_table(df, index='sentiment', columns='emotion', aggfunc='size', fill_value=0)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap='Blues', fmt='g')

plt.title('Relationship')
plt.xlabel('Emotion')
plt.ylabel('Sentiment')

plt.show()
# %% [markdown]
# Saved output
# <Figure size 800x600 with 2 Axes>
#
# %% [markdown]
# 
# ### **1-6. 留됰?洹몃옒?꾨줈 datatime怨?sentiment 愿怨??쒓컖??*
#
# %%
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# ?붾퀎 媛먯젙怨?sentiment 愿怨?吏묎퀎
sentiment_counts = df.groupby(df.index.to_period('M'))['sentiment'].value_counts().unstack().fillna(0)

# datetime ?뺤떇?쇰줈 ?몃뜳??蹂??sentiment_counts.index = sentiment_counts.index.to_timestamp()

# 洹몃옒??洹몃━湲?plt.figure(figsize=(10, 6))

plt.plot(sentiment_counts.index, sentiment_counts['neutral'], label='Neutral', color='green')
plt.plot(sentiment_counts.index, sentiment_counts['negative'], label='Negative', color='red')
plt.plot(sentiment_counts.index, sentiment_counts['positive'], label='Positive', color='blue')

plt.title('Monthly Sentiment Counts(2022)')
plt.xlabel('Month')
plt.ylabel('Count')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %% [markdown]
# Saved output
# <ipython-input-10-1ea495027c20>:5: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
#   sentiment_counts = df.groupby(df.index.to_period('M'))['sentiment'].value_counts().unstack().fillna(0)
# 
# <Figure size 1000x600 with 1 Axes>
#
# %% [markdown]
# 
# ### **1-7. 媛?媛먯젙蹂?text?먯꽌 媛??留롮씠 ?깆옣?섎뒗 ?⑥뼱 遺꾩꽍 - wordcloud**
#
# %%
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

pattern = r'@\w+'
def preprocess_text(text):
    text = re.sub(pattern, '', text)  # '@'濡??쒖옉?섎뒗 ?⑥뼱 ?쒓굅
    return text

df['clean_text'] = df['Text'].apply(preprocess_text)

emotions = df['emotion'].unique()

plt.figure(figsize=(15, 10))

for i, emotion in enumerate(emotions):
    filtered_text = ' '.join(df[df['emotion'] == emotion]['clean_text'])
    filtered_text_out = ' '.join(word for word in filtered_text.split() if word.lower() not in stop_words)
    word_counts_out = Counter(filtered_text_out.split())
    top_words_out = dict(word_counts_out.most_common(20))

    # WordCloud ?앹꽦
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(top_words_out)

    # subplot 異붽?
    plt.subplot(2, 4, i+1)  # 2??4?댁쓽 洹몃━?쒖뿉??i+1 踰덉㎏ subplot??洹몃┝
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{emotion.capitalize()} - Top 20 words')
    plt.axis('off')

plt.tight_layout(pad=2.0)  # subplot 媛?媛꾧꺽 ?볤쾶 ?ㅼ젙
plt.show()
# %% [markdown]
# Saved output
# [nltk_data] Downloading package stopwords to /root/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# 
# <Figure size 1500x1000 with 8 Axes>
#
# %% [markdown]
# 
# ## **2. ?곗씠???꾩쿂由?*
#
# %%
df
# %% [markdown]
# Saved output
#                                                                         Text  \
# Datetime                                                                       
# 2022-07-19 14:13:30+00:00  @AmanNagpal81 @SamsungIndia @SamsungMobile @Ap...   
# 2022-05-02 14:37:44+00:00  @jackoneill1984 @GamersNexus @Dell taking one ...   
# 2022-02-16 05:34:17+00:00  @MichaelDell @Dell @DellCaresPRO one more http...   
# 2022-02-26 14:14:19+00:00  @DellCares What should we do if your dealer ch...   
# 2022-09-02 22:39:32+00:00  #facebookdown @TheRock @steveaustinBSR혻 @WWEUn...   
# ...                                                                      ...   
# 2022-05-05 08:40:40+00:00  Reminder 7:\n@Dell @DellCares = #Delldoesnotgi...   
# 2022-07-18 00:49:00+00:00  Has @Dell support always been this terrible? A...   
# 2022-06-04 19:39:43+00:00  The promise of tech: work from anywhere, anyti...   
# 2022-03-30 22:03:22+00:00  Dear @Dell , first of all thanks to check my o...   
# 2022-02-23 00:06:10+00:00  @AirspanNetworks @DellTech @Dell We're excited...   
# 
#                           sentiment  sentiment_score       emotion  \
# Datetime                                                             
# 2022-07-19 14:13:30+00:00   neutral         0.699980       sadness   
# 2022-05-02 14:37:44+00:00   neutral         0.701479           joy   
# 2022-02-16 05:34:17+00:00   neutral         0.743684  anticipation   
# 2022-02-26 14:14:19+00:00  negative         0.841557         anger   
# 2022-09-02 22:39:32+00:00   neutral         0.901858  anticipation   
# ...                             ...              ...           ...   
# 2022-05-05 08:40:40+00:00  negative         0.833563         anger   
# 2022-07-18 00:49:00+00:00  negative         0.901842       disgust   
# 2022-06-04 19:39:43+00:00  positive         0.816528      optimism   
# 2022-03-30 22:03:22+00:00  negative         0.877455         anger   
# 2022-02-23 00:06:10+00:00  positive         0.982404           joy   
# 
#                            emotion_score  \
# Datetime                                   
# 2022-07-19 14:13:30+00:00       0.927320   
# 2022-05-02 14:37:44+00:00       0.767457   
# 2022-02-16 05:34:17+00:00       0.234333   
# 2022-02-26 14:14:19+00:00       0.983523   
# 2022-09-02 22:39:32+00:00       0.550912   
# ...                                  ...   
# 2022-05-05 08:40:40+00:00       0.919111   
# 2022-07-18 00:49:00+00:00       0.899949   
# 2022-06-04 19:39:43+00:00       0.970569   
# 2022-03-30 22:03:22+00:00       0.969596   
# 2022-02-23 00:06:10+00:00       0.974803   
# 
#                                                                   clean_text  
# Datetime                                                                      
# 2022-07-19 14:13:30+00:00                I need only leptop bro for my ed...  
# 2022-05-02 14:37:44+00:00                            taking one for the team  
# 2022-02-16 05:34:17+00:00                   one more https://t.co/4lKkc7djNR  
# 2022-02-26 14:14:19+00:00   What should we do if your dealer cheated rath...  
# 2022-09-02 22:39:32+00:00  #facebookdown  혻 혻  #psndown     혻혻  if u coul...  
# ...                                                                      ...  
# 2022-05-05 08:40:40+00:00  Reminder 7:\n  = #Delldoesnotgiveadamm\n\n#del...  
# 2022-07-18 00:49:00+00:00  Has  support always been this terrible? Asking...  
# 2022-06-04 19:39:43+00:00  The promise of tech: work from anywhere, anyti...  
# 2022-03-30 22:03:22+00:00  Dear  , first of all thanks to check my order,...  
# 2022-02-23 00:06:10+00:00     We're excited to support Airspan 5G #RAN fo...  
# 
# [1000 rows x 6 columns]
#
# %%
df.reset_index(inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%y-%m-%d')
df_clean = df.drop(['Text', 'sentiment_score', 'emotion_score'], axis=1)
# %%
df_clean
# %% [markdown]
# Saved output
#      Datetime sentiment       emotion  \
# 0    22-07-19   neutral       sadness   
# 1    22-05-02   neutral           joy   
# 2    22-02-16   neutral  anticipation   
# 3    22-02-26  negative         anger   
# 4    22-09-02   neutral  anticipation   
# ..        ...       ...           ...   
# 995  22-05-05  negative         anger   
# 996  22-07-18  negative       disgust   
# 997  22-06-04  positive      optimism   
# 998  22-03-30  negative         anger   
# 999  22-02-23  positive           joy   
# 
#                                             clean_text  
# 0                  I need only leptop bro for my ed...  
# 1                              taking one for the team  
# 2                     one more https://t.co/4lKkc7djNR  
# 3     What should we do if your dealer cheated rath...  
# 4    #facebookdown  혻 혻  #psndown     혻혻  if u coul...  
# ..                                                 ...  
# 995  Reminder 7:\n  = #Delldoesnotgiveadamm\n\n#del...  
# 996  Has  support always been this terrible? Asking...  
# 997  The promise of tech: work from anywhere, anyti...  
# 998  Dear  , first of all thanks to check my order,...  
# 999     We're excited to support Airspan 5G #RAN fo...  
# 
# [1000 rows x 4 columns]
#
# %%
# prompt: df_clean DataFrame ?ъ슜: clean_text瑜????⑹튇 all_text瑜?留뚮뱺 ?ㅼ쓬 媛?紐낆궗 ?⑥뼱??鍮덈룄?섎? 異붿텧?섎뒗 肄붾뱶瑜??묒꽦?댁＜?몄슂

import pandas as pd

# Combine all text into a single string
all_text = ' '.join(df_clean['clean_text'])

# Extract all nouns from the text
nouns = [word for word, pos in nltk.pos_tag(all_text.split()) if pos.startswith('N')]

# Count the frequency of each noun
noun_counts = pd.Series(nouns).value_counts()

# Print the noun counts
print(noun_counts)
# %% [markdown]
# 
# ## **3. Semantic Network Analysis(?뚮뱶?대씪?곕뱶 ?쒖쇅)**
#
# %%
!pip show squarify
!pip install squarify -q
!pip install konlpy -q
# %% [markdown]
# Saved output
# Name: squarify
# Version: 0.4.3
# Summary: Pure Python implementation of the squarify treemap layout algorithm
# Home-page: https://github.com/laserson/squarify
# Author: Uri Laserson
# Author-email: uri.laserson@gmail.com
# License: Apache v2
# Location: /usr/local/lib/python3.10/dist-packages
# Requires: 
# Required-by: 
# 
#
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import squarify

import requests
from bs4 import BeautifulSoup

import nltk
nltk.download("punkt")

from nltk import word_tokenize, bigrams
from nltk.util import ngrams
from nltk import ConditionalFreqDist, FreqDist

from konlpy.tag import Kkma
from konlpy.tag import Okt

from wordcloud import WordCloud
from collections import Counter

import time
import datetime

import warnings
warnings.filterwarnings('ignore')
# %% [markdown]
# Saved output
# [nltk_data] Downloading package punkt to /root/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# 
#
# %% [markdown]
# 
# ### **3-1. Treemap 洹몃━湲?*
#
# %%
from collections import Counter
from konlpy.tag import Okt
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

okt = Okt()

all_text = ' '.join(df_clean['clean_text'].tolist())
tokens = word_tokenize(all_text)
words = [word for word in tokens if len(word) > 1]
words_2 = [word for word in words if re.match(r'^[a-zA-Z]+$', word)]

def pos_tagging(words):
    tagged_words = pos_tag(words)
    return tagged_words

tagged_words = pos_tagging(words_2)
nouns = [word for word, pos in tagged_words if pos.startswith('NN')]


word_counts = Counter(nouns)
most_words = dict(word_counts.most_common(50))

cmap = matplotlib.cm.Set3
normalize = matplotlib.colors.Normalize(vmin=min(most_words.values()), vmax=max(most_words.values()))

colors = [cmap(normalize(value)) for value in most_words.values()]

plt.figure(figsize=(10, 10))
squarify.plot(label=most_words.keys(),
              sizes=most_words.values(),
              text_kwargs={'fontsize': 12},
              color=colors,
              ec='white',
              alpha=0.7)

plt.axis("off")
plt.show()
# %% [markdown]
# Saved output
# [nltk_data] Downloading package punkt to /root/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# [nltk_data] Downloading package averaged_perceptron_tagger to
# [nltk_data]     /root/nltk_data...
# [nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
# 
# <Figure size 1000x1000 with 1 Axes>
#
# %% [markdown]
# 
# ### **3-2. networkx 洹몃━湲?*
#
# %%
import networkx as nx
nltk.download('wordnet')
import itertools
import operator

most_freq_words = [word for word, count in word_counts.most_common(10)]

G = nx.Graph()

# ?몃뱶 異붽?
G.add_nodes_from(most_freq_words)

# 臾몄옣?먯꽌 紐낆궗 ?⑥뼱?ㅼ쓣 異붿텧?섍퀬 ?ㅽ듃?뚰겕???ｌ? 異붽?
article_sentences = df_clean['clean_text']
repeat_num = 1

for sentence in article_sentences:
    word_tokens = nltk.word_tokenize(sentence)
    tokens_pos = nltk.pos_tag(word_tokens)

    NN_words = [word for word, pos in tokens_pos if 'NN' in pos]

    wlem = nltk.WordNetLemmatizer()
    lemmatized_words = [wlem.lemmatize(word) for word in NN_words]

    selected_words = [word for word in lemmatized_words if word in most_freq_words]

    selected_words = list(set(selected_words))  # 以묐났 ?쒓굅

    repeat_num += 1

    # ?몃뱶 媛꾩쓽 議고빀???ｌ?濡?異붽?
    for pair in itertools.combinations(selected_words, 2):
        if G.has_edge(pair[0], pair[1]):
            G[pair[0]][pair[1]]['weight'] += 1
        else:
            G.add_edge(pair[0], pair[1], weight=1)

# 媛以묒튂 湲곗??쇰줈 ?뺣젹
edge_items = nx.get_edge_attributes(G, 'weight')
sorted_edge_items = sorted(edge_items.items(), key=operator.itemgetter(1), reverse=True)

color_map = []
for node in G:
    if G.degree(node) >= 2:   # ?덉떆濡?degree媛 2 ?댁긽??寃쎌슦 'pink' ?됱긽
        color_map.append('pink')
    else:
        color_map.append('beige')

# circular layout?쇰줈 ?쒓컖??plt.figure(figsize=(6, 6))
pos = nx.circular_layout(G, scale=0.2)

nx.draw_networkx(G, pos, node_color=color_map, edge_color='grey', with_labels=True, node_size=3000, font_size=10)

plt.axis('off')
plt.title('Network Graph with Node Color by Degree')
plt.show()
# %% [markdown]
# Saved output
# [nltk_data] Downloading package wordnet to /root/nltk_data...
# [nltk_data]   Package wordnet is already up-to-date!
# 
# <Figure size 600x600 with 1 Axes>
#
# %%
def get_node_size(node_values):
    node_sizes = np.array(list(node_values))
    node_sizes = 1000 * (node_sizes - min(node_sizes)) / (max(node_sizes) - min(node_sizes)) + 300
    return node_sizes

# ?ㅼ뼇??以묒떖??吏??怨꾩궛
dc = nx.degree_centrality(G)  # Degree Centrality
cc = nx.closeness_centrality(G)  # Closeness Centrality
bc = nx.betweenness_centrality(G)  # Betweenness Centrality
ec = nx.eigenvector_centrality(G, weight='weight')  # Eigenvector Centrality
pr = nx.pagerank(G)   # PageRank

def get_node_size(node_values):
    node_sizes = np.array(list(node_values.values()))
    node_sizes = 1000 * (node_sizes - min(node_sizes)) / (max(node_sizes) - min(node_sizes)) + 300
    return node_sizes

# ?몃뱶 ?ш린 ?ㅼ젙
node_sizes = get_node_size(dc)  # ?덉떆濡?Degree Centrality瑜?湲곗??쇰줈 ?ㅼ젙

# ?ㅽ듃?뚰겕 ?쒓컖??plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)  # ?덉씠?꾩썐 ?ㅼ젙
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='grey')
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', font_family='sans-serif')
plt.title('Network Visualization', fontsize=15)
plt.axis('off')
plt.show()
# %% [markdown]
# Saved output
# <Figure size 800x800 with 1 Axes>
#
# %%
print("Degree Centrality:")
for node, centrality in dc.items():
    print(f"{node}: {centrality}")

print("\nCloseness Centrality:")
for node, centrality in cc.items():
    print(f"{node}: {centrality}")

print("\nBetweenness Centrality:")
for node, centrality in bc.items():
    print(f"{node}: {centrality}")

print("\nEigenvector Centrality:")
for node, centrality in ec.items():
    print(f"{node}: {centrality}")

print("\nPageRank:")
for node, centrality in pr.items():
    print(f"{node}: {centrality}")
# %% [markdown]
# Saved output
# Degree Centrality:
# https: 0.0
# Dell: 0.8888888888888888
# service: 0.8888888888888888
# customer: 0.8888888888888888
# laptop: 0.8888888888888888
# dell: 0.8888888888888888
# time: 0.8888888888888888
# support: 0.8888888888888888
# product: 0.8888888888888888
# warranty: 0.8888888888888888
# 
# Closeness Centrality:
# https: 0.0
# Dell: 0.8888888888888888
# service: 0.8888888888888888
# customer: 0.8888888888888888
# laptop: 0.8888888888888888
# dell: 0.8888888888888888
# time: 0.8888888888888888
# support: 0.8888888888888888
# product: 0.8888888888888888
# warranty: 0.8888888888888888
# 
# Betweenness Centrality:
# https: 0.0
# Dell: 0.0
# service: 0.0
# customer: 0.0
# laptop: 0.0
# dell: 0.0
# time: 0.0
# support: 0.0
# product: 0.0
# warranty: 0.0
# 
# Eigenvector Centrality:
# https: 4.610662472835303e-22
# Dell: 0.3640511601568312
# service: 0.4604645442061795
# customer: 0.44044256204039023
# laptop: 0.3598871358950825
# dell: 0.2930034923852174
# time: 0.2493455168926666
# support: 0.22110871820652417
# product: 0.2701488836869898
# warranty: 0.24907235356283125
# 
# PageRank:
# https: 0.016393445303185497
# Dell: 0.1234496613931806
# service: 0.14983544964154777
# customer: 0.14217331275441564
# laptop: 0.12367572696377029
# dell: 0.10027137320735544
# time: 0.085909830089894
# support: 0.08129207688663832
# product: 0.0910171269987879
# warranty: 0.0859819967612246
# 
#
# %% [markdown]
# 
# ## **4. Topic Modeling 遺꾩꽍**
#
# %%
df = pd.read_csv("/content/drive/MyDrive/2024-1 ?밴낵 ?띿뒪?몃쭏?대떇媛쒕줎/emotion/sentiment-emotion-labelled_Dell_tweets.csv")
# %%
df = df.sample(n=1000, random_state=42)
# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Preprocess the text data
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['Text'])

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(X)

# Print topics and visualize
print("Top words for each topic:")
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    print(f"Topic #{topic_idx+1}: {' '.join(top_words)}")
# %% [markdown]
# Saved output
# Top words for each topic:
# Topic #1: dell laptop dellcares service https new customer warranty problem support
# Topic #2: dell https laptop just support new customer time work don
# Topic #3: dell https like really pc windows alienware xps order update
# Topic #4: dell michaeldell https twitter hp elonmusk emc dellcares don service
# Topic #5: dell https microsoft tech great intel learn amp starwars join
# 
#
# %%
df_Sadness = df[df['emotion'] == 'sadness']
df_Joy = df[df['emotion'] == 'joy']
df_Anticipation = df[df['emotion'] == 'anticipation']
df_Anger = df[df['emotion'] == 'anger']
df_Surprise = df[df['emotion'] == 'surprise']
df_Fear = df[df['emotion'] == 'fear']
df_Optimism = df[df['emotion'] == 'optimism']
df_Disgust = df[df['emotion'] == 'disgust']
# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# CountVectorizer 諛?LDA ?곸슜 ?⑥닔 ?뺤쓽
def apply_lda(df, emotion_label):
    vectorizer = CountVectorizer(max_features=2000)  # max_features 媛믪쓣 ?섎┝
    X = vectorizer.fit_transform(df['Text'])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    topics = lda.fit_transform(X)

    print(f"Top words for each topic in {emotion_label}:")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"Topic #{topic_idx+1}: {' '.join(top_words)}")
    print('----------------------')
# %%
apply_lda(df_Sadness, 'Sadness')
apply_lda(df_Joy, 'Joy')
apply_lda(df_Anticipation, 'Anticipation')
apply_lda(df_Anger, 'Anger')
apply_lda(df_Surprise, 'Surprise')
apply_lda(df_Fear, 'Fear')
apply_lda(df_Optimism, 'Optimism')
apply_lda(df_Disgust, 'Disgust')
# %% [markdown]
# Saved output
# Top words for each topic in Sadness:
# Topic #1: dell laptop is you for of with can or from
# Topic #2: dell the to my co https dellcares are no of
# Topic #3: dell to the and of is it my laptop get
# Topic #4: it dell the in for to and don is an
# Topic #5: acer_india apple only need my purpose samsungindia educational asus samsungmobile
# ----------------------
# Top words for each topic in Joy:
# Topic #1: dell and https co to the you is it with
# Topic #2: dell https co to the and of for on up
# Topic #3: dell co https the and to you on for get
# Topic #4: the dell https co to and of for it you
# Topic #5: dell to it the and https co for love its
# ----------------------
# Top words for each topic in Anticipation:
# Topic #1: dell the to co https for in you and on
# Topic #2: dell and co https the of you is like in
# Topic #3: dell the https co and to you in for of
# Topic #4: dell co https the to of and with is for
# Topic #5: dell the and https co on to of this well
# ----------------------
# Top words for each topic in Anger:
# Topic #1: dell to the my it and you for is laptop
# Topic #2: dell this the and they dellcares how https co is
# Topic #3: dell to the and is for of you service that
# Topic #4: the dell to it and is of laptop from my
# Topic #5: to dell my with of is your laptop they you
# ----------------------
# Top words for each topic in Surprise:
# Topic #1: dell one month laptop on new my more wow mail
# Topic #2: dell why this do long woah so legonas gaming keys
# Topic #3: the to and terms conditions can this do dell_in dellcares
# Topic #4: dell why this do long woah so legonas gaming keys
# Topic #5: why long woah so legonas dell this do gaming keys
# ----------------------
# Top words for each topic in Fear:
# Topic #1: dell get is my intel current phoronix carlonluca cannot on
# Topic #2: to through had dell have if get this exp them
# Topic #3: co https it you powerprotect from data what worst is
# Topic #4: co https you this us the dell have and about
# Topic #5: https co your to it dell with how unprotected would
# ----------------------
# Top words for each topic in Optimism:
# Topic #1: dell to co https the for and of on your
# Topic #2: co https dell with to your is it the one
# Topic #3: dell it and the to https co you are like
# Topic #4: dell and co https to for is the you with
# Topic #5: dell you my laptop to not buy it the from
# ----------------------
# Top words for each topic in Disgust:
# Topic #1: dell and is the of to it this my you
# Topic #2: dell the to it is with https co and on
# Topic #3: dell the to and have from they on is your
# Topic #4: dell the you this is co https computer in if
# Topic #5: the co https dell what you will 2000 say never
# ----------------------
# 
#
# %%
# CountVectorizer 諛?LDA ?곸슜 ?⑥닔 ?뺤쓽
def apply_lda(df, emotion_label, n_topics, max_iter):
    vectorizer = CountVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['Text'])

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=42)
    lda.fit(X)

    print(f"Top words for each topic in {emotion_label}:")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"Topic #{topic_idx + 1}: {' '.join(top_words)}")
    print('----------------------')

    return lda

# 媛먯젙蹂?LDA 紐⑤뜽 ?곸슜 諛?寃곌낵 異쒕젰
lda_anticipation = apply_lda(df[df['emotion'] == 'anticipation'], 'Anticipation', 10, 500)
lda_joy = apply_lda(df[df['emotion'] == 'joy'], 'Joy', 10, 500)
lda_anger = apply_lda(df[df['emotion'] == 'anger'], 'Anger', 10, 500)
lda_disgust = apply_lda(df[df['emotion'] == 'disgust'], 'Disgust', 10, 500)
# %% [markdown]
# Saved output
# Top words for each topic in Anticipation:
# Topic #1: dell the to co https in of it on that
# Topic #2: dell the https co in of and we michaeldell alienware
# Topic #3: dell the https co you with and for to your
# Topic #4: https co dell the of to and with is for
# Topic #5: dell of the on to you want what as will
# Topic #6: dell hp you this the time re but think should
# Topic #7: dell co https to the on and you microsoft it
# Topic #8: dell the to and is you for co https of
# Topic #9: to the dell for and on this in co https
# Topic #10: dell and it to the like with as that for
# ----------------------
# Top words for each topic in Joy:
# Topic #1: dell and the https co is to you it on
# Topic #2: dell https co and the to of for it that
# Topic #3: and dell the https co to on for of love
# Topic #4: the dell of https co to for and we you
# Topic #5: dell to it and for one be time the lol
# Topic #6: dell https co michaeldell the emc twitter elonmusk of you
# Topic #7: dell co https and the to with in from on
# Topic #8: dell co https to the and you for it your
# Topic #9: dell to https co the on delltech its great xps
# Topic #10: dell https co to and the it at this is
# ----------------------
# Top words for each topic in Anger:
# Topic #1: dell to the my it you and is of for
# Topic #2: dell to this and it dellcares the is can have
# Topic #3: dell and the is to for customer service of not
# Topic #4: the dell to it and is from laptop of they
# Topic #5: dell to and they my with your of you me
# Topic #6: is dell the my of you have to that service
# Topic #7: dell from your with different me same own is template
# Topic #8: dell to the of and laptop it for in with
# Topic #9: dell the to and for my is it service no
# Topic #10: dell hp on dellcares stop or of never whistling americanair
# ----------------------
# Top words for each topic in Disgust:
# Topic #1: dell and you it to of this co https which
# Topic #2: dell to it the on michaeldell for not laptop when
# Topic #3: to dell the from and on get my they in
# Topic #4: dell you the this https co it computer not microsoft
# Topic #5: dell slow clap albatross an tips spoon coaster purchase like
# Topic #6: dell what you https co service poor and to like
# Topic #7: dell the is my https co of with you windows
# Topic #8: dell the is of this and it to now my
# Topic #9: dell with and an up issue having ve adapter booting
# Topic #10: dell and is this bad your no you the support
# ----------------------
# 
#
# %% [markdown]
# 
# ## **5. Text classifiaction - BERT 紐⑤뜽 ?ъ슜**
#
# %%
import pandas as pd
import torch
import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
# %%
df_clf = df[['Text', 'emotion']][0:2000]
df_clf.columns = ['text', 'labels']
class_names = df_clf['labels'].unique()
print(class_names)
# %% [markdown]
# Saved output
# ['anticipation' 'joy' 'anger' 'sadness' 'fear' 'optimism' 'disgust'
#  'surprise']
# 
#
# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
# The secret `HF_TOKEN` does not exist in your Colab secrets.
# To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
# You will be able to reuse this secret in all of your notebooks.
# Please note that authentication is recommended but still optional to access public models or datasets.
#   warnings.warn(
# 
# tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]
# vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]
# tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]
# /usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
#   warnings.warn(
# 
# config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]
#
# %%
label_map = {label: idx for idx, label in enumerate(class_names)}
def map_label(label):
    return label_map[label]
# %%
class GPReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = map_label(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
# %%
train_dataset = GPReviewDataset(
    texts=df_clf['text'].values,
    labels=df_clf['labels'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
# %%
BATCH_SIZE = 16
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# %%
class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
# %%
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_names)
bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
model = BERTSentimentClassifier(bert_model, num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
# %% [markdown]
# Saved output
# model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]
#
# %%
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return correct_predictions.double() / n_examples, np.mean(losses), predictions, real_values
# %%
EPOCHS = 10
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# %%
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_dataset))
    print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')
# %% [markdown]
# Saved output
# Epoch 1/10
# ----------
# Train loss: 1.3870, accuracy: 0.5170
# Epoch 2/10
# ----------
# Train loss: 0.8817, accuracy: 0.6965
# Epoch 3/10
# ----------
# Train loss: 0.5977, accuracy: 0.7935
# Epoch 4/10
# ----------
# Train loss: 0.3879, accuracy: 0.8725
# Epoch 5/10
# ----------
# Train loss: 0.2490, accuracy: 0.9245
# Epoch 6/10
# ----------
# Train loss: 0.1842, accuracy: 0.9420
# Epoch 7/10
# ----------
# Train loss: 0.1168, accuracy: 0.9680
# Epoch 8/10
# ----------
# Train loss: 0.0811, accuracy: 0.9800
# Epoch 9/10
# ----------
# Train loss: 0.0463, accuracy: 0.9905
# Epoch 10/10
# ----------
# Train loss: 0.0384, accuracy: 0.9920
# 
#
# %%
df_clf_test = df[['Text', 'emotion']][2001:2500]
df_clf_test.columns = ['text', 'labels']
test_dataset = GPReviewDataset(
    texts=df_clf_test['text'].values,
    labels=df_clf_test['labels'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# %%
test_acc, test_loss, predictions, real_values = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(test_dataset)
)

# ?됯? 寃곌낵 異쒕젰
print(f'Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}')
# %% [markdown]
# Saved output
# Test loss: 1.3709, accuracy: 0.6854
# 
#
# %%
y_pred = predictions.numpy()
y_true = real_values.numpy()

# ?쇰룞?됰젹 ?앹꽦
conf_mat = confusion_matrix(y_true, y_pred)

# ?쇰룞?됰젹 ?쒓컖??plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1000x800 with 2 Axes>
#
# %%
# y_pred? y_true?먯꽌 ?ъ슜???대옒???뺤씤
unique_classes_pred = np.unique(y_pred)
unique_classes_true = np.unique(y_true)
# ?ㅼ젣 ?ъ슜???대옒?ㅼ쓽 媛쒖닔
num_classes_pred = len(unique_classes_pred)
num_classes_true = len(unique_classes_true)

# target_names瑜??ㅼ젣 ?ъ슜???대옒???대쫫?쇰줈 ?쒗븳
actual_class_names = [class_names[i] for i in unique_classes_true]
print(f"Actual class names: {actual_class_names}")

# ?ㅼ젣 ?ъ슜???대옒???몃뜳?ㅻ? labels濡??꾨떖
class_report = classification_report(y_true, y_pred, labels=unique_classes_true, target_names=class_names[unique_classes_true])

# 遺꾨쪟 蹂닿퀬??異쒕젰
print(class_report)
# %% [markdown]
# Saved output
# Actual class names: ['anticipation', 'joy', 'anger', 'sadness', 'fear', 'optimism', 'disgust']
#               precision    recall  f1-score   support
# 
# anticipation       0.54      0.66      0.60        93
#          joy       0.81      0.75      0.78       123
#        anger       0.88      0.85      0.86       141
#      sadness       0.51      0.54      0.53        39
#         fear       0.50      0.25      0.33         8
#     optimism       0.67      0.60      0.63        10
#      disgust       0.48      0.47      0.48        85
# 
#     accuracy                           0.69       499
#    macro avg       0.63      0.59      0.60       499
# weighted avg       0.69      0.69      0.69       499
# 
# 
#

