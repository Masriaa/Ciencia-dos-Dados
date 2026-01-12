# Ciência dos Dados
Este guia descreve os passos necessários para configurar o ambiente de desenvolvimento e acessar os notebooks do projeto.

## Pré-requisitos
* Git
* Python

## Guia de instalação

### 1. Clonar o repositório
```
git clone https://github.com/Masriaa/Ciencia-dos-Dados.git
cd Ciencia-dos-Dados
```
### 2. Configurar o Ambiente Virtual

#### Windows
```powershell
python -m venv .venv
.venv\bin\Activate.ps1
```

#### Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as Dependências do Trabalho
O projeto é dividido em trabalhos (T1, T2 e T3.). Cada trabalho possui um arquivo de dependências específico dentro da pasta `requirements`. Instale as dependências referentes ao trabalho que deseja executar.
#### Exemplo (para o Trabalho 3):
```bash
pip install -r requirements/T3.txt
```

### 4. Acessar os notebooks
Com o ambiente ativado e as dependências instaladas, inicie o servidor do JupyterLab.
```bash
jupyter lab
```


## Estrutura de diretórios do projeto
```
.
|-- README.md            <- Arquivo README principal com as instruções do projeto.
|-- .gitignore           <- Especifica arquivos e pastas a serem ignorados pelo Git.
|-- data
|   |-- external         <- Dados de fontes externas.
|   |-- interim          <- Dados intermediários que foram transformados.
|   |-- processed        <- Datasets finais, prontos para modelagem.
|   `-- raw              <- A fonte de dados original e imutável.
|-- models               <- Modelos treinados e serializados.
|-- notebooks
|   |-- T1               <- Jupyter Notebooks referentes ao Trabalho 1.
|   |-- T2               <- Jupyter Notebooks referentes ao Trabalho 2.
|   `-- T3               <- Jupyter Notebooks referentes ao Trabalho 3.
|-- references
|   |-- T1               <- Materiais de referência para o Trabalho 1.
|   |-- T2               <- Materiais de referência para o Trabalho 2.
|   `-- T3               <- Materiais de referência para o Trabalho 3.
|-- reports
|   |-- figures          <- Gráficos e figuras geradas para os relatórios.
|   |-- T1.pdf           <- Relatório do Trabalho 1.
|   |-- T2.pdf           <- Relatório do Trabalho 2.
|   `-- T3.pdf           <- Relatório do Trabalho 3.
|-- requirements
|   |-- T1.txt           <- Arquivo de dependências Python para o Trabalho 1.
|   |-- T2.txt           <- Arquivo de dependências Python para o Trabalho 2.
|   `-- T3.txt           <- Arquivo de dependências Python para o Trabalho 3.
`-- src                  <- Código-fonte principal do projeto
```
