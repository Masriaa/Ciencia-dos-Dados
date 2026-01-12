import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem import RSLPStemmer
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

def analisar_desbalanceamento(df, coluna_alvo):
    """
    Calcula o Imbalance Ratio e plota a distribuição das classes.
    """
    contagem = df[coluna_alvo].value_counts()
    n_maioria = contagem.max()
    n_minoria = contagem.min()
    ir = n_maioria / n_minoria

    print(f"--- Análise de Desbalanceamento ---")
    print(f"Imbalance Ratio (IR): {ir:.2f}")

    if ir <= 2:
        classificacao = "Praticamente balanceado"
    elif ir <= 4:
        classificacao = "Levemente desbalanceado"
    else:
        classificacao = "Explicitamente desbalanceado"
    
    print(f"Classificação: {classificacao}")

    return ir

def clean_text_olist(text: str) -> str:
    """
    Função auxiliar para limpar strings individuais de avaliações da Olist.
    """
    # Garantindo que é string e converter para minúsculas
    text = str(text).lower()
    # Removendo números
    text = re.sub(r'\d+', '', text)
    # Removendo pontuação
    text = re.sub(r'[^\w\s]', '', text) 
    # Removendo múltiplos espaços e quebras de linha
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def limpeza(df, nome_coluna):
    """
    Aplica a limpeza em uma coluna específica do DataFrame.
    """
    df[nome_coluna] = df[nome_coluna].apply(clean_text_olist)
    return df

def stemming(df, coluna):
    """
    Reduz as palavras ao radical e converte a lista de tokens de volta para string.
    """
    stemmer = RSLPStemmer()
        
    # 1. Aplica o stemming (em cada palavra da lista de tokens)
    df[coluna] = df[coluna].apply(lambda lista: [stemmer.stem(p) for p in lista])
    
    # 2. Converte a lista de volta para uma string única (Join)
    df[coluna] = df[coluna].apply(lambda x: " ".join(x))
    
    return df

def vetorizar_dados(X_train, X_test):
    """
    Configura e aplica as vetorizações BoW e TF-IDF.
    """
    # Configuração BoW
    bow_vectorizer = CountVectorizer(
        ngram_range=(1, 2), 
        min_df=5,
        max_df=0.8,
        max_features=5000,
        binary=False
    )

    # Configuração TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=5000,
        sublinear_tf=True
    )

    # Aplicação
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"Vetorização concluída: {X_train_bow.shape[1]} features extraídas.")
    
    return X_train_bow, X_test_bow, X_train_tfidf, X_test_tfidf

def avaliar_modelo(modelo, nome_modelo, X_train_bow, X_test_bow, X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Treina e avalia um modelo usando BoW e TF-IDF, plotando as matrizes lado a lado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    
    experimentos = [
        ("BoW", X_train_bow, X_test_bow, axes[0]),
        ("TF-IDF", X_train_tfidf, X_test_tfidf, axes[1])
    ]
    
    for vector, X_tr, X_te, ax in experimentos:
        # Treinamento
        modelo.fit(X_tr, y_train)
        y_pred = modelo.predict(X_te)
        
        # Relatório de Texto
        print(f"\n===== {nome_modelo} + {vector} =====")
        print(classification_report(y_test, y_pred))
        
        # Plot da Matriz
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax)
        ax.set_title(f'Matriz: {nome_modelo} + {vector}')
        ax.set_ylabel('Real')
        ax.set_xlabel('Previsto')

    plt.tight_layout()
    plt.show()

def resultados(dados):
    """
    Processa os rankings e gera as visualizações para BoW e TF-IDF.
    """
    df = pd.DataFrame(dados)

    # --- RANKING BAG OF WORDS (BoW) ---
    ranking_bow = df[df['Vetorização'] == 'BoW'].sort_values(by='Acurácia', ascending=False)
    
    # --- RANKING TF-IDF ---
    ranking_tfidf = df[df['Vetorização'] == 'TF-IDF'].sort_values(by='Acurácia', ascending=False)

    # Configuração visual para os gráficos
    sns.set_theme(style="whitegrid")

    # Visualização BoW
    plt.figure(figsize=(5, 3))
    sns.barplot(
        x='Acurácia', 
        y='Modelo', 
        data=ranking_bow, 
        palette='RdPu', 
        hue='Modelo',    
        legend=False     
    )
    plt.title('Performance: Modelos com Bag of Words (BoW)')
    plt.xlim(0.8, 0.92) 
    plt.xlabel('Acurácia')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()

    # Visualização TF-IDF
    plt.figure(figsize=(5, 3))
    sns.barplot(
        x='Acurácia', 
        y='Modelo', 
        data=ranking_tfidf, 
        palette='RdPu', 
        hue='Modelo',    
        legend=False     
    )
    plt.title('Performance: Modelos com TF-IDF')
    plt.xlim(0.8, 0.92)
    plt.xlabel('Acurácia')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()

    return ranking_bow, ranking_tfidf

def nuvens_palavras(df, coluna_texto, coluna_label):
    """
    Gera e exibe as nuvens de palavras para sentimentos positivos e negativos lado a lado.
    """
    # 1. Separando os comentários por opinião
    positivos = " ".join(df[df[coluna_label] == 1][coluna_texto])
    negativos = " ".join(df[df[coluna_label] == 0][coluna_texto])

    def gerar_nuvem(texto, titulo, cor, ax):
        nuvem = WordCloud(width=800, height=400, 
                          background_color='black', 
                          colormap=cor,
                          max_words=100).generate(texto)
        
        ax.imshow(nuvem, interpolation='bilinear')
        ax.set_title(titulo, fontsize=20)
        ax.axis('off')

    # 2. Criando a figura com 2 colunas para ficarem lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    gerar_nuvem(positivos, "Avaliações Positivas", "Greens", axes[0])
    gerar_nuvem(negativos, "Avaliações Negativas", "Reds", axes[1])
    
    plt.tight_layout()
    plt.show()