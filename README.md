# RAG Pipeline — Governança de Dados & LGPD

Pipeline de **Retrieval-Augmented Generation (RAG)** local em Python para consulta inteligente de documentos internos de Governança e LGPD. Desenvolvido como componente de portfólio de Engenharia de IA, demonstrando os pilares técnicos de Busca Semântica, Vetorização de Documentos e Limitação de Alucinação em LLMs de produção.

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         FASE DE INDEXAÇÃO                       │
│                         (executada 1x)                          │
│                                                                 │
│  governance_docs.json ──► document_loader.py ──► Chunks de     │
│                                                   Texto         │
│                                │                    │           │
│                                ▼                    ▼           │
│                     embeddings.py          Gemini               │
│                   build_vector_store()  gemini-embedding-001    │
│                                │                    │           │
│                                └──────────► Vector Store        │
│                                             (numpy arrays)      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      FASE DE CONSULTA (RAG loop)                │
│                                                                 │
│  Pergunta do usuário                                            │
│         │                                                       │
│         ▼                                                       │
│  Gemini gemini-embedding-001 ──► Query Vector                   │
│         │                                                       │
│         ▼                                                       │
│  Cosine Similarity vs. Vector Store ──► Top-K Documentos       │
│         │                                                       │
│         ▼                                                       │
│  Prompt Engineering (contexto injetado)                         │
│         │                                                       │
│         ▼                                                       │
│  Gemini 2.5 Flash ──► Resposta Fundamentada nos Documentos     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Conceitos de Engenharia de IA

### 1. Vetorização e Embeddings Semânticos

Documentos de texto não podem ser comparados diretamente por algoritmos matemáticos. A solução é convertê-los para o **espaço vetorial contínuo** — onde a proximidade geométrica entre dois vetores corresponde à similaridade semântica entre os textos que eles representam.

Este projeto usa o modelo **`gemini-embedding-001`** do Google Gemini, que projeta qualquer texto em um vetor denso de alta dimensionalidade. O processo de indexação converte os 12 documentos de governança em 12 vetores, formando o **vector store** em memória.

**Por que embeddings superam busca por palavras-chave (BM25)?**

| Critério | BM25 / Full-text | Embeddings Semânticos |
|---|---|---|
| "titular de dados" vs "pessoa cujos dados são tratados" | Sem match | Match alto |
| Sinônimos e paráfrases | Falha | Captura naturalmente |
| Contexto e intenção | Ignora | Codificado no vetor |
| Custo computacional | O(n) trivial | O(n·d) com vetorização prévia |

### 2. Busca Semântica com Similaridade de Cosseno

A função de busca (`semantic_search` em `embeddings.py`) implementa o algoritmo de **Approximate Nearest Neighbor** em sua forma exata (brute-force), adequada para bases com centenas de documentos.

A métrica de distância utilizada é a **Similaridade de Cosseno**:

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

O valor retornado varia de **-1 a +1**, onde:
- `1.0` → vetores idênticos (máxima relevância)
- `0.0` → vetores ortogonais (tópicos não relacionados)
- `-1.0` → vetores opostos (conceitos contrários)

A similaridade de cosseno é preferível à distância euclidiana para embeddings de texto porque **normaliza o comprimento dos vetores**, tornando-a insensível ao tamanho dos documentos.

**Para escalar a milhões de documentos**, essa busca brute-force seria substituída por índices ANN como FAISS (Meta) ou ScaNN (Google), que reduzem a complexidade de O(n) para O(log n).

### 3. Limitação de Alucinação de LLMs (Groundedness)

O maior risco em aplicações com LLMs em produção é a **alucinação** — geração de informações plausíveis mas factualmente incorretas. Em contextos de LGPD e compliance, um prazo errado ou uma obrigação inventada pode ter consequências legais reais.

O RAG resolve este problema na camada arquitetural, não na camada de prompting:

**Mecanismo de Grounding implementado:**

1. **Restrição de contexto**: O modelo só recebe como entrada os trechos recuperados pelo vector store. Não há acesso ao conhecimento paramétrico do modelo sobre LGPD.

2. **Instrução explícita no sistema**: O prompt instrui o modelo a responder **"Esta informação não está disponível nos documentos"** caso a resposta não esteja nos trechos fornecidos — evitando respostas inventadas.

3. **Citação de fontes**: O modelo é instruído a citar o ID de cada documento do qual extraiu informação (`[pol-001]`), criando **rastreabilidade auditável** — fundamental para compliance.

4. **Separação de responsabilidades**: O LLM não é responsável por *saber* — é responsável por *articular e sintetizar* o que os documentos dizem.

```
SEM RAG:  Pergunta ──► LLM (conhecimento paramétrico) ──► Resposta (pode alucinar)

COM RAG:  Pergunta ──► Retrieval ──► Contexto real ──► LLM ──► Resposta fundamentada
```

### 4. Prompt Engineering para RAG

O `_build_prompt` em `rag_pipeline.py` implementa um padrão de **Constrained Generation**:

- **Role definition**: define o modelo como especialista em LGPD da empresa
- **Context injection**: insere os documentos recuperados com metadados (ID, título, categoria)
- **Negative space instruction**: lista explicitamente o que o modelo NÃO deve fazer
- **Boundary enforcement**: limita o contexto a `_MAX_CONTEXT_CHARS` para evitar estouro de context window

---

## Estrutura do Projeto

```
Retrieval-Augmented Generation/
│
├── documents/
│   └── governance_docs.json        # Base de conhecimento (12 políticas de LGPD/Governança)
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py          # Carregamento e validação dos documentos JSON
│   ├── embeddings.py               # Geração de embeddings e busca semântica por cosseno
│   └── rag_pipeline.py             # Orquestrador do pipeline RAG (indexação + consulta)
│
├── main.py                         # CLI com modos interativo, demo e one-shot
├── requirements.txt
├── .env.example
└── README.md
```

---

## Stack Tecnológico

| Componente | Tecnologia | Justificativa |
|---|---|---|
| LLM (geração) | `gemini-2.5-flash` | Estado da arte em raciocínio com custo otimizado |
| Modelo de Embeddings | `gemini-embedding-001` | Modelo estável de embeddings do Gemini, mesma API do LLM |
| SDK Python | `google-genai >= 1.16` | SDK oficial unificado Google (substitui `google-generativeai`) |
| Álgebra vetorial | `numpy` | Produto interno e norma L2 para cosseno sem overhead |
| Gerenciamento de env | `python-dotenv` | 12-factor app: credenciais em `.env`, fora do código |
| Logging | `logging` (stdlib) | Rastreabilidade de cada etapa do pipeline sem dependência extra |

---

## Configuração e Execução

### Pré-requisitos

- Python 3.11+
- Chave de API do Google Gemini ([obtenha aqui no Google AI Studio](https://aistudio.google.com/app/apikey))

### Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd "Retrieval-Augmented Generation"

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure a chave de API
cp .env.example .env
# Edite .env e insira sua GEMINI_API_KEY
```

### Execução

```bash
# Modo interativo (chat com os documentos)
python main.py

# Pergunta única
python main.py --query "Quais são os direitos dos titulares de dados?"

# Pergunta única com output em JSON
python main.py --query "O que é o RIPD?" --output-json resultado.json

# Executar 5 perguntas de demonstração e exportar resultados
python main.py --demo

# Ajustar número de documentos recuperados (padrão: 3)
python main.py --top-k 5

# Logs detalhados (DEBUG)
python main.py --verbose
```

### Exemplo de Output

```
======================================================================
PERGUNTA: Qual é o prazo para comunicar um incidente de segurança à ANPD?
======================================================================

RESPOSTA:
Com base nos documentos de governança consultados, em caso de incidente de
segurança envolvendo dados pessoais, a empresa deve comunicar o ocorrido à
ANPD em um prazo máximo de **72 horas** após a ciência do incidente [pol-007].

A notificação deve conter: descrição da natureza dos dados afetados,
informações sobre os titulares envolvidos, medidas técnicas utilizadas,
riscos relacionados e medidas adotadas para mitigar os efeitos [pol-007].

FONTES RECUPERADAS (3 documento(s)):
  • [pol-007] Gestão de Incidentes de Segurança com Dados Pessoais (similaridade: 0.8821)
  • [pol-003] Direitos dos Titulares de Dados (similaridade: 0.7134)
  • [pol-001] Política Geral de Proteção de Dados Pessoais (similaridade: 0.6903)
======================================================================
```

---

## Base de Conhecimento

A base de dados em `documents/governance_docs.json` contém 12 documentos cobrindo:

| ID | Documento | Categoria |
|---|---|---|
| pol-001 | Política Geral de Proteção de Dados Pessoais | LGPD |
| pol-002 | Bases Legais para Tratamento de Dados | LGPD |
| pol-003 | Direitos dos Titulares de Dados | LGPD |
| pol-004 | Encarregado de Proteção de Dados (DPO) | Governança |
| pol-005 | Relatório de Impacto à Proteção de Dados (RIPD) | Governança |
| pol-006 | Política de Retenção e Descarte de Dados | Governança |
| pol-007 | Gestão de Incidentes de Segurança | Segurança |
| pol-008 | Transferência Internacional de Dados | LGPD |
| pol-009 | Privacy by Design e Privacy by Default | Governança |
| pol-010 | Compartilhamento com Terceiros e Operadores | Governança |
| pol-011 | Treinamento e Conscientização | Governança |
| pol-012 | Registro de Atividades de Tratamento (RAT/ROPA) | Governança |

---

## Limitações e Próximos Passos

**Limitações atuais (intencionais para este escopo):**
- Vector store em memória (sem persistência em disco): reindexa a cada execução
- Busca brute-force: O(n) — adequada para dezenas/centenas de documentos
- Chunking simples por documento completo (sem sliding window)

**Evoluções naturais para produção:**
- Persistência com **ChromaDB** ou **pgvector** (PostgreSQL) para eliminar reindexação
- Chunking com sobreposição (overlap) para capturar contexto em fronteiras de parágrafos
- **Reranker** cross-encoder (ex: Cohere Rerank) para refinar os top-K após cosine search
- Índice **FAISS** (IndexFlatIP) para escalar a milhões de documentos sem regressão de latência
- API REST com **FastAPI** para exposição como microserviço

---

