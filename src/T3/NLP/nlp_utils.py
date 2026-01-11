import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def analisar_desbalanceamento(df, coluna_alvo):
    """
    Calcula o Imbalance Ratio e plota a distribuição das classes.
    """
    contagem = df[coluna_alvo].value_counts()
    n_maioria = contagem.max()
    n_minoria = contagem.min()
    ir = n_maioria / n_minoria

    print(f"--- Análise de Desbalanceamento ---")
    print(f"Classe Majoritária: {n_maioria}")
    print(f"Classe Minoritária: {n_minoria}")
    print(f"Imbalance Ratio (IR): {ir:.2f}")

    if ir <= 2:
        classificacao = "Praticamente balanceado"
    elif ir <= 4:
        classificacao = "Levemente desbalanceado"
    else:
        classificacao = "Explicitamente desbalanceado"
    
    print(f"Classificação: {classificacao}")

    return ir

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